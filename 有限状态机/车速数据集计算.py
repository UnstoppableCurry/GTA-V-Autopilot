import cv2
import numpy as np
from PIL import ImageGrab
import time
import win32api
import win32con
import win32gui
import win32ui
from pynput import keyboard
import time
import csv
import codecs
from threading import Thread  # 创建线程的模块


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow([data[0], data[1][0], data[1][1], data[1][2]])
    print("保存文件成功，处理结束")


def data_write_csv_4_imgdata(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow([data[0], data[1], data[2]])
    print("保存文件成功，处理结束")


def on_press(key):
    global last_time
    last_time = time.time()
    data.append([last_time])
    print('start time ->', last_time)
    return False


def on_release(key):
    know = time.time()
    time_step = know - last_time
    print('release', key, know)
    if not hasattr(key, 'vk'):
        key.vk = 'space'
    if len(data) == 1:
        data[-1].append([know, time_step, key])
    else:
        data[-1].append([know - data[0][0], time_step, key])
    return False


def keboard_save(time1):
    global data
    data = []
    while 1:
        print(data)
        with keyboard.Listener(
                on_press=on_press) as listener:
            listener.join()
        with keyboard.Listener(
                on_release=on_release) as listener:
            listener.join()
        print('时间差->', time1 - data[0][0])
        # print(data[-1][1][-1].vk)

        if len(data) > 1 and data[-1][1][-1].vk == 106:  # data[-1][1][-1].vk == 81 or
            data_write_csv('./test.csv', data)
            break


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


if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # fourcc= cv2.VideoWriter_fourcc(*'MP4V')
    outv = cv2.VideoWriter('wtx.mp4', fourcc, 24, (150, 350))  # 写入视频
    last_time1 = time.time()
    x1, y1 = 196 - 50, 513 - 350
    x2, y2 = x1 + 150, y1 + 350
    f = 0
    p = Thread(target=keboard_save, args=(last_time1,))
    p.start()
    img_data = []
    while True:
        # print(f)
        f += 1
        # PressKey(W)
        frame = grab_screen((0, 0, 800, 600))
        # frame = np.array(ImageGrab.grab(bbox=(40, 40, 800, 600)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('demo', frame)
        frame2 = frame[y1:y2, x1:x2, :]
        # print(frame2.shape)
        cv2.putText(frame2, str(f), (0, 25), cv2.FONT_ITALIC, 1,
                    (255, 255, 255), 3)
        cv2.imshow('frame2', frame2)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            data_write_csv_4_imgdata('img_data.csv', img_data)
            break
        know = time.time()
        outv.write(frame2)
        if f < 2:
            img_data.append([last_time1, know, f])
        else:
            img_data.append([last_time1, know - last_time1, f])
        # print(know)

    cv2.destroyAllWindows()
