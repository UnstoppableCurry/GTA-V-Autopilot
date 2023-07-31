import torch, os, cv2
from fastlanedetection.model import parsingNet
import torch
import scipy.special
import numpy as np
import argparse
import matplotlib.pyplot as plt
import glob
from MutilCarDection_by_pytorch_yolo2 import *

# "参数设置" 车道线检测先眼眶
tusimple_row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                       116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                       168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                       220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                       272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

# "参数设置" 相机矫正棋盘格
nx = 7
ny = 7
file_paths = glob.glob("./camera_cal/*.jpg")


# 根据参数矩阵完成透视变换
def img_perspect_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)


# 图像预处理
def process_data(img):
    # 通道转换 BGR to RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    # 类型转换 uint8 to float32
    img = np.ascontiguousarray(img, dtype=np.float32)
    # 归一化 0 - 255 to 0.0 - 1.0
    img /= 255.0
    return img


def img_undistort(img, mtx, dist):
    dis = cv2.undistort(img, mtx, dist, None, mtx)
    return dis


# 相机校正：外参，内参，畸变系数
def cal_calibrate_params(file_paths):
    # 存储角点数据的坐标
    object_points = []  # 角点在三维空间的对未知
    image_points = []  # 角点在图像空间中的位置
    # 生成角点在真实世界中的位置
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # 角点检测
    for file_path in file_paths:
        img = cv2.imread(file_path)
        # 灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 角点检测
        rect, coners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if rect == True:
            object_points.append(objp)
            image_points.append(coners)
    # 相机较真
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


# 透视变换
# 获取透视变换的参数矩阵
def cal_perspective_params(img, points=[[601, 448], [683, 448], [230, 717], [1097, 717]]):
    offset_x = 330
    offset_y = 0
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(points)
    # 设置俯视图中的对应的四个点
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y], [img_size[0] - offset_x, img_size[1] - offset_y]])
    # 原图像转换到俯视图
    M = cv2.getPerspectiveTransform(src, dst)
    # 俯视图到原图像
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    return M, M_inverse


def cal_line_param(binary_warped):
    # 3.用曲线拟合检测出的点,二次多项式拟合，返回的结果是系数
    lane_param_cil = {}
    for i in binary_warped.keys():
        left_fit = np.polyfit(np.concatenate(np.array(binary_warped[i])[:, 1:]),
                              np.concatenate(np.array(binary_warped[i])[:, :1]), 2)
        lane_param_cil[i] = left_fit
    return lane_param_cil


# 填充车道线之间的多边形
def fill_lane_poly(img, lane_param_cil, lane_dict):
    # 获取图像的行数
    # y_max = img.shape[0]
    # 设置输出图像的大小，并将白色位置设为255
    out_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # out_img = np.dstack((img, img, img))
    collor = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    left_fit = 0
    points = {}
    min_y = 0
    max_y = 0
    left_fit_y_max = []
    left_fit_y_min = []
    for i in lane_param_cil.keys():
        if isinstance(left_fit, int):
            left_fit_y_max = list(np.array(lane_dict[i])[:, 1:]).pop(0)[0]
            left_fit_y_min = list(np.array(lane_dict[i])[:, 1:]).pop()[0]
            left_fit = lane_param_cil[i]
            left_points = [[left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2], y] for y in
                           [i for i in range(left_fit_y_min, left_fit_y_max, 1)]]
            points[i] = left_points
        else:
            right_fit_y_max = list(np.array(lane_dict[i])[:, 1:]).pop(0)[0]
            right_fit_y_min = list(np.array(lane_dict[i])[:, 1:]).pop()[0]
            # min_y = np.max([left_fit_y_min, right_fit_y_min])
            # max_y = np.min([left_fit_y_max, right_fit_y_max])
            min_y = np.min([left_fit_y_min, right_fit_y_min])
            max_y = np.max([left_fit_y_max, right_fit_y_max])
            right_fit = lane_param_cil[i]
            # minst_y = np.min([left_fit_y_min, right_fit_y_min])
            # 在拟合曲线中获取左右车道线的像素位置
            # print('img.shape[0]', img.shape[0])
            left_points = [[left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2], y] for y in
                           [i for i in range(min_y, max_y, 1)]]
            right_points = [[right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2], y] for y in
                            [i for i in range(max_y - 1, min_y, -1)]]
            points[i] = right_points
            # 将左右车道的像素点进行合并
            line_points = np.vstack((left_points, right_points))
            # if line_points.shape[1] == 0:
            #     return img, points
            print(line_points.shape)
            # 根据左右车道线的像素位置绘制多边形
            cv2.fillPoly(out_img, [np.int_([line_points])], collor[i])
            left_fit_y_max = right_fit_y_max
            left_fit_y_min = right_fit_y_min
            left_fit = right_fit
            # cv2.imshow('2   ', out_img)
        # img = img * 0.5 + out_img * 0.1
        # result = cv2.addWeighted(img, 0.9, out_img, 0.8, 0)
        # cv2.imshow('1   ', result)
        # cv2.waitKey(0)
    return cv2.addWeighted(img, 0.9, out_img, 0.8, 0), points


# 计算车道线曲率
def cal_radius(img, lane_data, M):
    # 比例
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    # 得到车道线上的每个点
    left_fit = np.polyfit(np.concatenate(np.array(lane_data)[:, 1:]),
                          np.concatenate(np.array(lane_data)[:, :1]), 2)
    left_y_axis = np.concatenate(np.array(lane_data)[:, 1:])
    left_x_axis = np.concatenate(np.array(lane_data)[:, :1])
    tf_img = np.zeros((img.shape[1], img.shape[0], 1), np.uint8)
    tf_img[np.int_(np.dstack((left_x_axis, left_y_axis)))[0]] = 1
    trasform_img = img_perspect_transform(tf_img, M)
    tf_lane_data = trasform_img.nonzero()
    nonzeroy = np.array(tf_lane_data[0])
    nonzerox = np.array(tf_lane_data[1])
    # 把曲线中的点映射真实世界，在计算曲率
    if len(nonzeroy) == 0:
        print('曲率计算->pass')
        return -1
    left_fit_cr = np.polyfit(nonzeroy * ym_per_pix, nonzerox * xm_per_pix, 2)

    # 计算曲率
    left_curverad = ((1 + (2 * left_fit_cr[0] * nonzeroy * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    # 将曲率半径渲染在图像上
    cv2.putText(img, 'Radius of Curvature = {}(m)'.format(np.mean(left_curverad)), (20, 50), cv2.FONT_ITALIC, 1,
                (255, 255, 255), 1)
    return 1


def get_left_right_fit(dicts, w, h):
    left_fit = None
    lists = []
    lists2 = []
    lists3 = {}
    min_y = 0
    for i in dicts.keys():
        if min_y == 0:
            min_y = np.array(dicts[i])[np.array(dicts[i])[:, 1:].argmax()][1]
        else:
            if min_y > np.array(dicts[i])[np.array(dicts[i])[:, 1:].argmax()][1]:
                min_y = np.array(dicts[i])[np.array(dicts[i])[:, 1:].argmax()][1]
    for i in dicts.keys():
        lists2.append(i)
        # lists.append(np.abs(w / 2 - np.array(dicts[i])[:1, :1][0][0]))#np.array(dicts[i])[:, :1].mean()
        min_x = np.array(dicts[i])[:, :1][np.array(dicts[i])[:, 1:] == min_y][0]
        lists.append(np.abs(
            w / 2 - min_x))  # np.array(dicts[i])[:, :1].mean()
        lists3[i] = min_x
    index = np.argsort(np.array(lists))

    result_index = []
    print('正确道路选择--》', lists, lists2, lists3, index)
    # if dicts[lists2[index[0]]][:1][0][0] > dicts[lists2[index[1]]][:1][0][0]:#lists3[lists2[index[0]]]
    if lists3[lists2[index[0]]] > lists3[lists2[index[1]]]:
        result_index = [lists2[index[1]], lists2[index[0]]]
        left_fit = dicts[lists2[index[1]]]
        right_fit = dicts[lists2[index[0]]]
        mind = lists3[lists2[index[0]]] - lists3[lists2[index[1]]]
    else:
        result_index = [lists2[index[0]], lists2[index[1]]]
        left_fit = dicts[lists2[index[0]]]
        right_fit = dicts[lists2[index[1]]]
        mind = lists3[lists2[index[1]]] - lists3[lists2[index[0]]]

    length = np.linalg.norm(np.array(dicts[lists2[index[0]]][:1][0]) - np.array(dicts[lists2[index[1]]][:1][0]))  # 欧氏距离
    print(result_index)
    # return left_fit, right_fit, result_index, length, dicts[lists2[index[0]]][:1][0][0] - \
    #        dicts[lists2[index[1]]][:1][0][0]
    return np.int_(left_fit), np.int_(right_fit), result_index, length, mind


def get_left_right_fit2(dicts, w, h):
    '''
    前面的 方法效果太差了，废弃写个新方法重构了
    :param dicts:
    :param w:
    :param h:
    :return:
    '''
    sort_dicts = sorted(dicts.items(), key=lambda x: x[1][0][0], reverse=False)  # 按字典集合中，每一个元组的第二个元素排列。
    dicts = {}
    for i in sort_dicts:
        dicts[i[0]] = i[1]
    lanes = dicts.values()
    step = 0
    mind_variable = None
    result = {}
    for i in lanes:
        if step == 0:
            mind_variable = np.array(i)
            step += 1
        else:
            i = np.array(i)
            for j in i:
                x = j[0]
                y = j[1]
                x_minds = mind_variable[:, :1][mind_variable[:, 1:] == y]
                if x_minds:
                    x_minds = x_minds[0]
                    if step in result.keys():
                        result[step].append([(x_minds + x) / 2, y])
                    else:
                        result[step] = [[(x_minds + x) / 2, y]]
            mind_variable = np.array(i)
            step += 1
    # print(result)
    loss_min_k = 0
    loss_min_v = 0
    for k in result.keys():
        v = result[k]
        x_mean = np.concatenate(np.array(v)[:, :1]).mean()
        if loss_min_v == 0:
            loss_min_v = np.abs(w / 2 - x_mean)
            loss_min_k = k
        else:
            if loss_min_v > np.abs(w / 2 - x_mean):
                loss_min_v = np.abs(w / 2 - x_mean)
                loss_min_k = k
    label_index = [x for x in dicts.keys()]
    right_result_index = [label_index[loss_min_k], label_index[loss_min_k - 1]]  # 默认最少有两条识别出来的线
    if right_result_index[0] > right_result_index[1]:
        return right_result_index, result[loss_min_k]
    else:
        return [right_result_index[1], right_result_index[0]], result[loss_min_k]


# 计算车道线中心
def cal_line_center(y_max, img, M, left_fit, right_fit):
    trasform_img = img_perspect_transform(img, M)
    # trasform_img2 = img_perspect_transform(img2, M)
    data = np.nonzero(trasform_img)
    # data2 = np.nonzero(trasform_img)
    left_fit = np.polyfit(data[0], data[1], 2)
    # right_fit = np.polyfit(data2[0], data2[1], 2)
    left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    # right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    # return (left_x + right_x) / 2
    return left_x


def cal_center_departure(img, left_fit, right_fit, lane_center, x_length=700, mind=0):
    # 计算中心点
    y_max = img.shape[0]
    # lane_center = (img.shape[1] / 2 + lane_center) / 2 + 100

    # left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    # right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    xm_per_pix = 3.7 / x_length
    # print('车道线中点', mind)
    cv2.circle(img, (int(mind / 2), y_max - 10), 10, (255, 255, 255), -1)
    cv2.circle(img, (int(img.shape[1] / 2), y_max - 10), 10, (0, 255, 255), -1)

    # center_depart = ((left_x + right_x) / 2 - lane_center) * xm_per_pix
    center_depart = (img.shape[1] / 2 - int(mind / 2)) * xm_per_pix
    print(center_depart, '距离')
    # 渲染
    if center_depart > 0.1 * 3.7:
        cv2.putText(img, 'Vehicle is {}m right of center'.format(center_depart), (20, 100), cv2.FONT_ITALIC, 1,
                    (255, 255, 255), 1)
    elif center_depart < -0.1 * 3.7:
        cv2.putText(img, 'Vehicle is {}m left of center'.format(-center_depart), (20, 100), cv2.FONT_ITALIC, 1,
                    (255, 255, 255), 1)
    else:
        cv2.putText(img, 'Vehicle is in the center', (20, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 1)
    return img


def get_fast_lane_detection_parmers():
    # config start
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--dataset', default='CULane', type=str)
    parser.add_argument('--backbone', default='18', type=str)
    parser.add_argument('--griding_num', default=200, type=int)
    parser.add_argument('--test_model', default='./model/culane_18.pth', type=str)
    parser.add_argument('--auto_backup', action='store_true', help='automatically backup current code in the log path')
    # config end
    torch.backends.cudnn.benchmark = True
    cfg = parser
    cfg.backbone = parser.get_default('backbone')
    cfg.griding_num = parser.get_default('griding_num')
    cfg.test_model = parser.get_default('test_model')
    cfg.backbone = parser.get_default('backbone')
    cfg.dataset = parser.get_default('dataset')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError
    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cuda')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    row_anchor = culane_row_anchor

    return cfg, net, row_anchor, cls_num_per_lane


def lane_predict(frame, image, net, row_anchor, cfg, cls_num_per_lane, img_h, img_w):
    '''
    车道线预测函数
    :param image: 图像
    :param net: 网络
    :param row_anchor: 车道线先眼眶
    :param cfg: 配置
    :param cls_num_per_lane: 间隔
    :param img_h: 图像高
    :param img_w: 图像宽
    :return:车道线预测值，宽度，精确拟合预测值字典
    '''
    with torch.no_grad():
        out = net(image.unsqueeze(0))
    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)  # 以griding_num为间隔划分cls区间
    col_sample_w = col_sample[1] - col_sample[0]  # 固定间隔像素宽度
    out_j = out[0].data.cpu().numpy()  # 降维转numpy
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(cfg.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == cfg.griding_num] = 0
    out_j = loc
    lane_dict = {}
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:

                    ppp = [int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1]
                    if i not in lane_dict.keys():
                        lane_dict[i] = [ppp]
                    else:
                        lane_dict[i].append(ppp)
                    # cv2.circle(frame, ppp, 5, (0, 255, 0), -1)
    # print('\n lane_dict', lane_dict)
    return out_j, col_sample_w, lane_dict


def center(img, lists, x_length=700):
    # 计算中心点
    mind_index = np.concatenate(np.array(lists)[:, 1:]).argmax()
    mind = np.array(lists)[:, :1][mind_index]
    y_max = img.shape[0]
    xm_per_pix = 3.7 / x_length
    cv2.circle(img, (int(mind), y_max - 10), 10, (255, 255, 255), -1)
    cv2.circle(img, (int(img.shape[1] / 2), y_max - 10), 10, (0, 255, 255), -1)

    # center_depart = ((left_x + right_x) / 2 - lane_center) * xm_per_pix
    center_depart = (img.shape[1] / 2 - int(mind)) * xm_per_pix
    # print(center_depart, '距离')
    # 渲染
    if center_depart > 0.1 * 3.7:
        cv2.putText(img, 'Vehicle is {}m right of center'.format(center_depart), (20, 100), cv2.FONT_ITALIC, 1,
                    (255, 255, 255), 1)
        return 0, center_depart, img
    elif center_depart < -0.1 * 3.7:
        cv2.putText(img, 'Vehicle is {}m left of center'.format(-center_depart), (20, 100), cv2.FONT_ITALIC, 1,
                    (255, 255, 255), 1)
        return 1, center_depart, img

    else:
        cv2.putText(img, 'Vehicle is in the center', (20, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 1)
        return -1, center_depart, img


def lane_expand_draw(lane_dict, frame, img_w, img_h, out_img, M):
    index, loss = None, None
    # try:
    #     lane_param_cil = cal_line_param(lane_dict)  # 计算斜率
    #     frame, points = fill_lane_poly(frame, lane_param_cil, lane_dict)  # 多边形填充
    #     cal_radius(frame, lane_param_cil[1])
    #     # left_fit, right_fit, list_index, x_length, mind = get_left_right_fit(lane_dict, points, img_w)
    #     get_left_right_fit2(points, img_w, img_h)
    #     left_fit, right_fit, list_index, x_length, mind = get_left_right_fit(points, img_w, img_h)
    #     out_img[np.vstack((left_fit, right_fit))] = 1
    #     # out_img[np.array(left_fit)] = 1
    #     # out_img2[np.array(right_fit)] = 1
    #     lane_center = cal_line_center(img_h, out_img, M, lane_param_cil[list_index[0]],
    #                                   lane_param_cil[list_index[1]])
    #     # lane_center = 0
    #     # frame = cal_center_departure(frame, left_fit, right_fit,
    #     #                              lane_center, x_length, mind)
    #     index, loss, frame = center(frame, left_fit, right_fit, lane_center, 700, mind)
    # except:
    #     return frame, index, loss
    lane_param_cil = cal_line_param(lane_dict)  # 计算斜率
    frame, points = fill_lane_poly(frame, lane_param_cil, lane_dict)  # 多边形填充
    try:
        if points:
            # left_fit, right_fit, list_index, x_length, mind = get_left_right_fit(points, img_w, img_h) #方法废弃
            # print(points)
            right_result_index, result = get_left_right_fit2(lane_dict, img_w, img_h)
            k = cal_radius(frame, result, M)
            for i in result:
                cv2.circle(frame, [int(i[0]), int(i[1])], 5, (255, 255, 255), -1)
            index, loss, frame = center(frame, result, x_length=700)
            return frame, index, loss
        else:
            print('points errors')
    except:
        print('other errors')
        # print(lane_dict)
        # exit()
        return frame, None, None
    return frame, None, None


def predict_lane():
    yolo, COLORS, time_backtracking, tracker, memory, category_index = get_yolo_parameter()
    cfg, net, row_anchor, cls_num_per_lane = get_fast_lane_detection_parmers()
    yolo.eval()
    cap = cv2.VideoCapture('./video/project_video.mp4')
    # 获取属性
    img_w = int(cap.get(3))
    img_h = int(cap.get(4))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
    outv = cv2.VideoWriter('wtx.mp4', cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                           (img_w, img_h))  # 写入视频

    while cap.isOpened():
        ret, frame = cap.read()  # 获取每一帧图像
        if ret:
            yolo_predict_result = yolo_predict(frame, yolo, img_size=608)  # 目标检测-yolo算法
            out_img = np.zeros((frame.shape[1], frame.shape[0], 1), np.uint8)
            M, M_inverse = cal_perspective_params(out_img)  # 投射变换矩阵
            imgs = torch.tensor(process_data(cv2.resize(frame, (800, 288)))).cuda()
            out_j, col_sample_w, lane_dict = lane_predict(frame, imgs, net, row_anchor, cfg, cls_num_per_lane, img_h,
                                                          img_w)  # 车道线检测
            # print(lane_dict)
            frame, index, loss = lane_expand_draw(lane_dict, frame, img_w, img_h, out_img, M)  # 车道线扩展功能与绘画
            if yolo_predict_result:
                frame, memory, time_backtracking = sort(yolo_predict_result[0], yolo_predict_result[1], tracker, memory,
                                                        yolo_predict_result[2], frame, time_backtracking,
                                                        COLORS, yolo_predict_result[3])  # sort-卡尔曼滤波 多传感器融合
            cv2.imshow('demo', frame)
            outv.write(frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def predict_lanedemo():
    cfg, net, row_anchor, cls_num_per_lane = get_fast_lane_detection_parmers()
    frame = cv2.imread('demo3.jpg')
    # 获取属性
    img_w = int(frame.shape[1])
    img_h = int(frame.shape[0])

    out_img = np.zeros((frame.shape[1], frame.shape[0], 1), np.uint8)
    M, M_inverse = cal_perspective_params(out_img)  # 投射变换矩阵
    imgs = torch.tensor(process_data(cv2.resize(frame, (800, 288)))).cuda()
    out_j, col_sample_w, lane_dict = lane_predict(frame, imgs, net, row_anchor, cfg, cls_num_per_lane, img_h,
                                                  img_w)  # 车道线检测
    print(lane_dict)
    frame, index, loss = lane_expand_draw(lane_dict, frame, img_w, img_h, out_img, M)  # 车道线扩展功能与绘画
    cv2.imshow('demo', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # predict_lanedemo()
    predict_lane()
