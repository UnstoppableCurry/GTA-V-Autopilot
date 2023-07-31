import time
import cv2
import numpy as np
import csv
import codecs
import pandas as pd
import matplotlib.pyplot as plt
import torch


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow([data[0], data[1]])
    print("保存文件成功，处理结束")


def read(file_name):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'r', 'utf-8')  # 追加
    writer = csv.reader(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    for i in writer:
        print(i)


def filtter_img(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS).astype(np.float)
    hls = hls[250:330, :]
    h_chanel = hls[:, :, 0]
    l_chanel = hls[:, :, 1]
    s_chanel = hls[:, :, 2]
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s_binary = np.zeros_like(s_chanel, np.uint8)
    # s_binary[(s_chanel >= s_thresh[0]) & (s_chanel <= s_thresh[1])] = 1 * 255
    # s_binary[h_chanel < 30] = 0  # 色调
    # s_binary[h_chanel > 120] = 0
    # s_binary[l_chanel < 70] = 0  # 亮度
    s_binary[(h_chanel > 30) & (h_chanel < 120)] = 1 * 255
    s_binary[l_chanel < 70] = 0  # 亮度

    # speed = s_binary[250:, :]
    speed = s_binary
    index = image_np[0:35, :100]
    # cv2.imshow('s_binary', s_binary)
    # print(image_np.shape)
    # 统计直方图
    return speed, index


def histogram(data, axis, show=False):
    histogram_y = np.sum(data[:, :], axis=axis)
    # print(histogram)
    if show:
        plt.plot(histogram_y)
        plt.show()
    correlation = 0  # 相关性
    extremum = {}  # 极值点
    dert_x, dert_y = 0, 0
    for i in range(len(histogram_y) - 1):
        y = np.int(histogram_y[i])
        now_correlation = y - dert_y
        if correlation == 0:
            if now_correlation > 0:
                correlation = 1
            else:
                correlation = -1
        elif correlation == 1:
            if now_correlation < 0:
                correlation = -1  # 极大值
                extremum[i] = y
            elif now_correlation > 0:
                correlation = 1  # 正相关不变
        elif correlation == -1:
            if now_correlation > 0:
                correlation = 1  # 极小值
                extremum[i] = y
            elif now_correlation:
                correlation = -1  # 负相关不变
        dert_x = i
        dert_y = y
    return extremum


def my_kalman_filter(speed_img):
    extremum_x = histogram(speed_img, 0)
    x_cut = [x for x in extremum_x.keys()]
    if len(x_cut) == 0:
        return None
    # x轴滤波
    if x_cut[-1] - x_cut[0] > 100:  # 如果极值点之间的距离过大，那么认为是噪音 直接进行滤波
        for i in range(len(x_cut) - 1):
            if i != 0 and x_cut[i] - x_cut[0] < 100:
                x_cut = x_cut[:i]
                break
    extremum_y = histogram(speed_img[:, x_cut[0]:x_cut[-1]], 1)
    y_cut = [x for x in extremum_y.keys()]
    if len(y_cut) == 0:
        return None
    speed_img = speed_img[:, x_cut[0]:x_cut[-1]][y_cut[0]:y_cut[-2], :]
    # print('-->cut', extremum_x)
    if speed_img.shape[0] < 2 or speed_img.shape[1] < 2:
        return None
    # cv2.imshow('draw', speed)
    # contours, hierarchy = cv2.findContours(speed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(speed, contours, -1, (0, 255, 0), 3)
    # demo = np.ones((500, 500, 1), np.uint8)
    # demo = demo * 255
    # cv2.drawContours(demo, contours, -1, (0, 0, 0), 1)
    # cv2.imshow('demo', demo)
    return speed_img


def index_histogram(data, axis, show=False):
    histogram_y = np.sum(data[:, :], axis=axis)
    # print(histogram)
    if show:
        plt.plot(histogram_y)
        plt.show()
    correlation = 0  # 相关性
    extremum = {}  # 极值点
    dert_x, dert_y = 0, 0
    for i in range(len(histogram_y) - 1):
        y = np.int(histogram_y[i])
        now_correlation = y - dert_y
        if correlation == 0:
            if now_correlation > 0:
                correlation = 1
            else:
                correlation = -1
        elif correlation == 1:
            if now_correlation < 0:
                if y > 2000:
                    correlation = -1  # 极大值
                    extremum[i] = y
            elif now_correlation > 0:
                correlation = 1  # 正相关不变
        elif correlation == -1:
            if now_correlation > 0:
                if y < 1000:
                    correlation = 1  # 极小值
                    extremum[i] = y
            elif now_correlation:
                correlation = -1  # 负相关不变
        dert_x = i
        dert_y = y
    return extremum


def index_img_split(index):
    index_x = index_histogram(index, 0, True)  # index x轴滤波
    index_x_cut = [x for x in index_x.keys()]
    # print(index_x_cut)
    # if len(index_x_cut) == 0:
    #     continue
    index = index[:, 0:index_x_cut[-2] + 20]
    data_index_list = []
    for i in range(len(index_x_cut)):
        if len(index_x_cut) < 4:
            data_index_list = [index]
            continue
        last_cut = None
        if i % 2 == 0:
            cut = index_x_cut[i - 1]
            if None is last_cut:
                data_index_list.append(index[:, :cut])
            else:
                data_index_list.append(index[:, last_cut:cut])
                last_cut = cut
    # for i in data_index_list:
    #     imgdata = i
    #     imgdata = cv2.resize(imgdata, (28, 28))
    #     imgdata[imgdata != 0] = 1.0 * 255
    #     # robert_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    #     # imgdata = cv2.convertScaleAbs(cv2.filter2D(imgdata, cv2.CV_16S, robert_x))
    #     # imgdata = cv2.copyMakeBorder(imgdata1, 0, 28 - imgdata1.shape[1], 28 - imgdata1.shape[0],
    #     #                              28 - imgdata1.shape[1],
    #     #                              cv2.BORDER_CONSTANT, value=[0, 255, 0])
    #
    #     cv2.imshow('imgdata', imgdata)
    #     result = my_model.query(imgdata.reshape((28 * 28, 1)) / 255.0)
    #     predict_index = list(result.reshape(10)).index(max(result.reshape(10)))
    #     result2 = model(torch.tensor(imgdata.reshape((28, 28))).unsqueeze(0).unsqueeze(0) / 255.0)
    #     predict_index2 = result2.argmax()
    #
    #     print(predict_index, predict_index2)
    # imgdata = cv2.resize(speed, (28, 28))
    # cv2.imshow('imgdata', imgdata)
    # imgdata = np.array(imgdata, np.int)
    # result = my_model.query(imgdata.reshape((28 * 28, 1)) / 255.0)
    # result2 = model(torch.tensor(imgdata.reshape((1, 1, 28, 28))) / 255.0)
    # predict_index = list(result.reshape(10)).index(max(result.reshape(10)))
    # predict_index2 = result2.argmax()
    # print(predict_index)
    # print(predict_index2)


def laplaceOperator(data, x=5):
    '''
    实现图像锐化，x 图像锐化等级
    :param x: 锐化力度
    :param data:图像
    :return:
    '''
    sharpen_op = np.array([[0, -1, 0],
                           [-1, x, -1],
                           [0, -1, 0]], dtype=np.float32)
    sharpen_image = cv2.filter2D(data, cv2.CV_32F, sharpen_op)
    sharpen_image = cv2.convertScaleAbs(sharpen_image)  # 可实现图像增强等相关操作的快速运算
    return sharpen_image


def img_append(speed):
    '''
    图像填充操作
    :param speed:
    :return:
    '''
    if speed is None:
        return None
    if speed.shape[1] > speed.shape[0]:  # 高度大，宽度小
        w = (speed.shape[1] - speed.shape[0]) // 2
        h = 0
    elif speed.shape[1] < speed.shape[0]:
        w = 0
        h = (speed.shape[0] - speed.shape[1]) // 2
    else:
        w, h = 0, 0
    speed = cv2.copyMakeBorder(speed, w + 20, w + 20, h + 20, h + 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # print(speed.shape)
    speed_img = cv2.resize(speed, (28, 28))
    # print(speed.shape)
    return speed_img


def number_predict(img, my_model):
    speed_input = img_append(img)  # 图片填充
    # robert_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    # speed_img = cv2.convertScaleAbs(cv2.filter2D(speed_img, cv2.CV_16S, robert_x))#边缘提取
    # speed_input[speed_input != 0] = 255
    speed_img = speed_input
    # speed_img = laplaceOperator(speed_input)
    # cv2.imshow('resize', speed_img)
    result = my_model.query(speed_img.reshape((28 * 28, 1)) / 255.0)
    predict_index = list(result.reshape(10)).index(max(result.reshape(10)))
    # with torch.no_grad():
    #     result2 = model(torch.tensor(speed_img.reshape((28, 28))).unsqueeze(0).unsqueeze(0) / 255.0)
    # predict_index2 = result2.argmax()
    print('numpy_model_predict->', predict_index, result[predict_index][0] * 100, '%')
    # print('pytorch_model_predict->', predict_index2.data, result2[:, 9][0] * 100, '%')
    if None is speed_input:
        return
    # cv2.imshow('speed2', speed_input)
    # cv2.waitKey(0)
    # plt.imshow(speed_input)
    # plt.show()
    return predict_index


def create_my_datasets():
    my_model = load_model('my_dataset_model.npz')
    model = torch.load('minist.pth')
    model.to('cpu')
    model.eval()
    # f = csv.reader(open('img_data.csv', 'r'))
    # for i in f:
    #     print(i)
    a = pd.read_csv(open('img_data.csv', 'r'), sep='\t')
    print(a.head())
    cap = cv2.VideoCapture('wtx.mp4')
    s_thresh = (23, 170)  # 饱和度
    data = []
    ff = 0
    i = 0
    last_img = None
    while cap.isOpened():
        ret, frame = cap.read()  # 获取每一帧图像
        if ret:
            if i < 526:
                i += 1
                continue
            speed, index = filtter_img(frame)
            speed = cv2.resize(speed, (500, 500))
            cv2.imshow('speed', speed)
            speed = my_kalman_filter(speed)
            if speed is None:
                continue
            if speed.shape[1] > 55:  # 阈值小于这个 说明是一位数
                speed = speed[:, (speed.shape[1] // 2) + 5:]
            number_predict(speed, my_model)
            if last_img is None:
                last_img = speed
                continue
            cv2.imshow('speed', last_img)
            label = input()  # 制作自己的数据集
            print('index', ff, 'input_label-->', label)
            data.append([ff, label])
            filename = 'data/myminists/' + str(ff + 1) + '.jpg'
            cv2.imwrite(filename, last_img)
            # cv2.waitKey(0)
            last_img = speed
            ff += 1
            if cv2.waitKey(25) & 0xFF == ord("q"):
                data_write_csv('data/myminists/my_number_minist_datasets.csv', data)
                break


def predict():
    my_model = load_model('my_dataset_model.npz')
    model = torch.load('minist.pth')
    model.to('cpu')
    model.eval()
    # f = csv.reader(open('img_data.csv', 'r'))
    # for i in f:
    #     print(i)
    a = pd.read_csv(open('img_data.csv', 'r'), sep='\t')
    print(a.head())
    cap = cv2.VideoCapture('wtx.mp4')
    s_thresh = (23, 170)  # 饱和度
    f = 0
    data = []
    while cap.isOpened():
        ret, frame = cap.read()  # 获取每一帧图像
        if ret:
            speed, index = filtter_img(frame)
            speed = cv2.resize(speed, (500, 500))
            cv2.imshow('speed', speed)
            # cv2.imshow('index', index)
            speed = my_kalman_filter(speed)
            if speed is None:
                continue
            if speed.shape[1] > 55:  # 阈值小于这个 说明是一位数
                speed1 = speed[:, :(speed.shape[1] // 2)]
                speed2 = speed[:, (speed.shape[1] // 2) + 5:]
                label = number_predict(speed1, my_model) * 10 + number_predict(speed2, my_model)
            else:
                label = number_predict(speed, my_model)
            f += 1
            print(label)

            # label = input() 制作自己的数据集
            # print('index', f, 'input_label-->', label)
            # data.append([f, label])
            # filename = 'data/myminists/' + str(f) + '.jpg'
            # cv2.imwrite(filename, speed)
            cv2.waitKey(0)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                # data_write_csv('data/myminists/my_number_minist_datasets.csv', data)
                break
            if f > 380:
                cv2.waitKey(0)


if __name__ == '__main__':
    import csv
    from NeatureNetWork import *
    from minist import Net

    # create_my_datasets()
    predict()
