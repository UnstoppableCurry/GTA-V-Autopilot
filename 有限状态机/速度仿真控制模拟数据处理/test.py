import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
# sheet = pd.read_csv(open('./data/myminists/my_number_minist_datasets.csv', 'r'), sep='\t')
# print(a.head())
# files = os.listdir('data/myminists/')  # 得到文件夹下的所有文件名称
# for i in files:
#     print(i)
# for row in sheet.index.values:
# print(sheet.iloc[row, :])
from NeatureNetWork import *


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


def train(model):
    jishu = np.zeros((10), np.int)
    with open('./data/myminists/my_number_minist_datasets.csv', 'r') as f:
        f = [x for x in f]
        for index in range(len(f) - 1):
            i = f[index]
            # while 1:
            #     if jishu[0] > 20 and i.split(' ')[0] == 0:
            #         index += 1
            #         i = f[index]
            #     else:
            #         break

            # print(i)
            img_path = './data/myminists/' + i.split(' ')[0] + '.jpg'
            img = cv2.imread(img_path)
            img = img_append(img)
            label = np.zeros((10), np.int)
            label[int(i.split(' ')[1])] = 1
            jishu[int(i.split(' ')[1])] += 1
            # if int(i.split(' ')[1]) != 0:
            #     print(' ')
            print(index)
            model.fit(img[:, :, 1].reshape(784, 1), label)
            cv2.imshow('a', img[:, :, 1])
            # cv2.waitKey(0)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    print(jishu)
    plt.plot(model.loss_list)
    plt.show()
    return model


if __name__ == '__main__':
    model = neuralNetwork(784, 100, 10, 0.01)
    model = load_model('model.npz')
    model.lr = 0.05
    epoch = 10
    for i in range(epoch):
        print('epoch', i)
        model = train(model)
    model.save_model('my_dataset_model.npz')
