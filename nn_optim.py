from torch import nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import MaxPool2d, Flatten, Linear, Sequential
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1, drop_last=True)


# 使用sequential
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01, )
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss  # running_loss相当于扫一遍全部数据的loss总和
    print(running_loss)

vgg16_false = torchvision.models.vgg16(pretrained=False) #pretrained=False仅加载网络模型，无参数

vgg16_true = torchvision.models.vgg16(pretrained=True) #pretrained=True加载网络模型，并从网络中下载在数据集上训练好的参数
print(vgg16_true)
import torchvision
import torch.nn as nn
vgg16_true = torchvision.models.vgg16(pretrained=True) #pretrained=True加载网络模型，并从网络中下载在数据集上训练好的参数
dataset = torchvision.datasets.CIFAR10("./dataset",train= False, transform =torchvision.transforms.ToTensor(), download=True)
vgg16_true.add_module('add_linear', nn.Linear(1000,10))
print(vgg16_true)