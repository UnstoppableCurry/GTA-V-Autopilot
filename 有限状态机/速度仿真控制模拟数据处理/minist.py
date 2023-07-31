import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
# batch_size = 64 * 4
batch_size = 1

# MNIST Dataset# MNIST数据集已经集成在pytorch datasets中，可以直接调用

train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)

        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(40, 10)  # （in_features, out_features）

    def forward(self, x):
        # in_size = 64
        in_size = x.size(
            0)  # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        # x: 64*1*28*28
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*10*12*12  feature map =[(28-4)/2]^2=12*12
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv3(x)))

        x = x.view(in_size, -1)  # flatten the tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        x = self.fc(x)
        # x:64*10
        # print(x.size())
        # return F.log_softmax(x)  # 64*10
        return F.softmax(x, dim=1)  # 64*10


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):  # batch_idx是enumerate（）函数自带的索引，从0开始
        # data.size():[64, 1, 28, 28]
        # target.size():[64]
        data = data.to('cuda:0')
        output = model(data)
        # output:64*10

        loss = F.nll_loss(output, target.to('cuda:0'))
        loss = loss.mean()

        if batch_idx % 200 == 0:
            print(epoch, batch_idx * 100 // len(train_loader.dataset), len(train_loader.dataset),
                  loss.item())

        optimizer.zero_grad()  # 所有参数的梯度清零
        loss.backward()  # 即反向传播求梯度
        optimizer.step()  # 调用optimizer进行梯度下降更新参数


def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target = target.to('cuda:0')
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data.to('cuda:0'))
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).mean().item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        # print(pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # model = Net().to('cuda:0')
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # for epoch in range(1, 50):
    #     torch.save(model, './minist.pth')
    # train(epoch)
    # test()
    model = torch.load('minist.pth')
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target = target.to('cuda:0')
        data, target = Variable(data, volatile=True), Variable(target)
        with torch.no_grad():
            plt.imshow(data[0][0])
            plt.show()
            output = model(data.to('cuda:0'))
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).mean().item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            # print(pred)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
