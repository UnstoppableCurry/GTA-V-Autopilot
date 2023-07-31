import time

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32ui
from PIL import ImageGrab
from visualization_by_pytorch import *
from util.control import *


def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def screen_record():
    last_time = time.time()
    while (True):
        # 800x600 windowed mode
        printscreen = np.array(ImageGrab.grab(bbox=(40, 40, 1280, 720)))
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def autopilot(frame, yolo, COLORS, time_backtracking, tracker, memory, cfg, net, row_anchor, cls_num_per_lane,
              img_h, img_w):
    yolo_predict_result = yolo_predict(frame, yolo, img_size=608)  # 目标检测-yolo算法
    # out_img = np.zeros((frame.shape[1], frame.shape[0], 1), np.uint8)
    # M, M_inverse = cal_perspective_params(out_img)  # 投射变换矩阵
    imgs = torch.tensor(process_data(cv2.resize(frame, (800, 288)))).cuda()
    out_j, col_sample_w, lane_dict = lane_predict(frame, imgs, net, row_anchor, cfg, cls_num_per_lane, img_h,
                                                  img_w)  # 车道线检测
    if len(lane_dict.keys()) > 1:
        #     # print(lane_dict)
        frame, index, loss = lane_expand_draw(lane_dict, frame, img_w, img_h, out_img, M)  # 车道线扩展功能与绘画
    #     if None != index != loss:
    #         if index == 0:  # right
    #             PressKey(A)
    #             ReleaseKey(A)
    #         elif index == 1:  # left
    #             PressKey(D)
    #             ReleaseKey(D)
    if yolo_predict_result:
        frame, memory, time_backtracking = sort(yolo_predict_result[0], yolo_predict_result[1], tracker, memory,
                                                yolo_predict_result[2], frame, time_backtracking,
                                                COLORS, yolo_predict_result[3])  # sort-卡尔曼滤波 多传感器融合
    return frame, memory, time_backtracking


def autopolot_agent():
    yolo, COLORS, time_backtracking, tracker, memory, category_index = get_yolo_parameter()
    cfg, net, row_anchor, cls_num_per_lane = get_fast_lane_detection_parmers()
    # cfg, net, row_anchor, cls_num_per_lane = (0, 0, 0, 0)
    yolo.eval()
    # cap = cv2.VideoCapture('./video/project_video.mp4')
    # 获取属性
    outv = cv2.VideoWriter('wtx.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (800, 600))  # 写入视频
    last_time = time.time()
    while True:
        # PressKey(W)
        # frame = grab_screen((0, 0, 800, 600))
        frame = np.array(ImageGrab.grab(bbox=(40, 40, 800, 600)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_time = time.time()
        # frame = np.array(ImageGrab.grab(bbox=(40, 40, 1024, 768)))

        img_w = int(frame.shape[1])
        img_h = int(frame.shape[0])
        # print(img_w, img_h)
        cv2.imshow('frame', frame)

        frame, memory, time_backtracking = autopilot(frame, yolo, COLORS, time_backtracking, tracker, memory, cfg, net,
                                                     row_anchor,
                                                     cls_num_per_lane,
                                                     img_h, img_w)
        fps = str(1 // (time.time() - last_time))
        cv2.putText(frame, fps + 'fps', (500, 50), cv2.FONT_ITALIC, 1,
                    (255, 255, 255), 1)
        cv2.imshow('result', frame)
        # outv.write(frame)
        # ReleaseKey(W)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            outv.release()
            cv2.destroyAllWindows()
            break


# def center(img, left_fit, right_fit, lane_center, x_length=700, mind=0):
#     # 计算中心点
#     y_max = img.shape[0]
#     # lane_center = (img.shape[1] / 2 + lane_center) / 2 + 100
#
#     # left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
#     # right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
#     xm_per_pix = 3.7 / x_length
#     cv2.circle(img, (int(mind / 2), y_max - 10), 10, (255, 255, 255), -1)
#     cv2.circle(img, (int(img.shape[1] / 2), y_max - 10), 10, (0, 255, 255), -1)
#
#     # center_depart = ((left_x + right_x) / 2 - lane_center) * xm_per_pix
#     center_depart = (img.shape[1] / 2 - int(mind / 2)) * xm_per_pix
#     print(center_depart, '距离')
#     # 渲染
#     if center_depart > 0.1 * 3.7:
#         cv2.putText(img, 'Vehicle is {}m right of center'.format(center_depart), (20, 100), cv2.FONT_ITALIC, 1,
#                     (255, 255, 255), 5)
#         return 0, center_depart, img
#     elif center_depart < -0.1 * 3.7:
#         cv2.putText(img, 'Vehicle is {}m left of center'.format(-center_depart), (20, 100), cv2.FONT_ITALIC, 1,
#                     (255, 255, 255), 5)
#         return 1, center_depart, img
#
#     else:
#         cv2.putText(img, 'Vehicle is in the center', (20, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 5)
#         return -1, center_depart, img


0
if __name__ == '__main__':
    # grab_screen((40, 40, 1280, 720))
    print('start')
    autopolot_agent()
