import matplotlib.pyplot as plt
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist

from 交通.Autopilot.acc定速循环.有限状态机.速度仿真控制模拟数据处理 import NeatureNetWork as nnw


def train():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
    plt.rcParams['figure.figsize'] = (7, 7)  # Make the figures a bit bigger
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_train[i], cmap='gray', interpolation='none')  # 灰度一通道,不插值操作
        plt.title("数字{}".format(y_train[i]))
    plt.show()
    # 数据归一化
    x_train = x_train.reshape(60000, 784, 1)
    x_test = x_test.reshape(10000, 784, 1)
    # 格式转换
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # 归一化
    x_train /= 255
    x_test /= 255
    # 目标值热编码
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)
    # 神经网络训练
    model = nnw.neuralNetwork(784, 100, 10, 0.1)
    # print(x_test[1].shape)
    num = 60000
    for i in range(num):
        model.fit(x_train[i], y_train[i])
        print((i / num) * 100, '%')
    print('训练完成')
    # 验证准确率
    x = 0
    num2 = 10000
    for i in range(num2):
        result = model.query(x_test[i])
        predict_index = list(result.reshape(10)).index(max(result.reshape(10)))
        # print(y_test.dtype, y_test)
        label_index = list(y_test[i]).index(max(y_test[i].reshape(10)))
        print(predict_index, label_index)
        plt.imshow(x_test[i].reshape((28, 28)), cmap='gray', interpolation='none')  # 灰度一通道,不插值操作
        plt.title("数字{}".format(label_index))
        plt.show()
        if predict_index == label_index:
            x += 1
        # print('*' * 11)
    print(f'准确率为{x * 100 / num2}%')
    # model.save_model('model.npz')


def data_show():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
    plt.rcParams['figure.figsize'] = (7, 7)  # Make the figures a bit bigger
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_train[i], cmap='gray', interpolation='none')  # 灰度一通道,不插值操作
        plt.title("数字{}".format(y_train[i]))
    plt.show()


if __name__ == '__main__':
    data_show()
    train()
