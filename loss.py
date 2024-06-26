from torch import nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import MaxPool2d,Flatten,Linear, Sequential
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset",train= False, transform =torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1, drop_last=True)

#使用sequential
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
#     print(outputs)
#     print(targets)
    result_loss = loss(outputs, targets)
    print(result_loss)