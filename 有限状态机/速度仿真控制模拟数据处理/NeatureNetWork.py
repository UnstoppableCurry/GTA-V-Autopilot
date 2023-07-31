class neuralNetwork:
    # init net
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, weight=1):
        import numpy as np
        import scipy.special
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inodes = inputnodes  # 输入层
        self.hnodes = hiddennodes  # 隐藏层
        self.onodes = outputnodes  # 输出层
        self.lr = learningrate
        self.weight = weight  # 选择随机初始化权重方式
        self.loss_list = []  # 用于可视化误差
        # 构建权重矩阵
        import numpy as np
        if self.weight == 1:
            # 下一层节点数开根随机初始化权重，因为如果用上一层的神经元数量太麻烦了# pow(x,y) x的y次方
            self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))  # 隐藏层*输入层 矩阵
            self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))  # 输出层*隐藏层 矩阵
        elif self.weight == 0:
            self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
            self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
            # train net

    def fit(self, inputs_list, target_list):
        import numpy as np
        import scipy.special
        # 输入数据处理
        inputs = np.array(inputs_list, ndmin=2).T  # ndmin指定数组所满足的最小维度是什么,int整型
        # print(target_list.shape)
        targets = np.array(target_list, ndmin=2).T
        # print(targets.shape)
        # 传递
        hidden_inputs = np.dot(self.wih, inputs_list)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 隐层 到 输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # 计算误差
        output_errors = targets - final_outputs
        self.loss_list.append(np.abs(np.average(output_errors)))
        print('误差为', np.average(output_errors))
        # 误差反向传播
        # print(self.who.T.shape)(100, 10)
        # output_errors = output_errors[0]  # 产生了自动广播机制,只取第一列就行
        hidden_error = np.dot(self.who.T, output_errors)
        # 更新权重
        # print('hidden_errors', hidden_error.shape)
        # print('output_errors', output_errors.shape)
        # print('final_outputs', final_outputs.shape)
        # print((1 - final_outputs).shape)
        # print(np.transpose(hidden_outputs).shape)
        # print(inputs)
        # print(final_outputs)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))
        # 更新输入层与隐藏层权重
        self.wih += self.lr * np.dot((hidden_error * hidden_outputs * (1 - hidden_outputs)),
                                     np.transpose(inputs.T))

    # 给定输入，从输出节点给出答案

    def query(self, inputs_list):
        import numpy as np
        import scipy.special
        # 输入数据处理
        inputs = np.array(inputs_list, ndmin=2).T  # ndmin指定数组所满足的最小维度是什么,int整型
        # targets = np.array(target_list, ndmin=2).T
        # 传递
        hidden_inputs = np.dot(self.wih, inputs_list)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 隐层 到 输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def save_model(self, path):
        import numpy as np
        print(self.who.shape, self.wih.shape)
        arr = np.array([self.inodes, self.hnodes, self.onodes], np.int32)
        # np.save(path, arr)
        np.savez(path, who=self.who, wih=self.wih, arr=arr, lr=self.lr)


def load_model(path):
    import numpy as np
    data = np.load(path)
    print(data['arr'].shape)
    model = neuralNetwork(inputnodes=data['arr'][0], hiddennodes=data['arr'][1], outputnodes=data['arr'][2],
                          learningrate=data['lr'])
    model.who = data['who']
    model.wih = data['wih']
    return model


# if __name__ == '__main__':
#     import numpy as np
#
#     wih = np.random.normal(0.0, pow(28, -1), (28 * 28, 1))  # 隐藏层*输入层 矩阵
#     # print(wih.shape)
#     model = neuralNetwork(28 * 28, 100, 10, 0.01)
#     label = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#     # print(len(label))
#     model.fit(wih, np.array(label))
if __name__ == '__main__':
    model = load_model('model.npz')
    print(model.inodes)
    print(model)
