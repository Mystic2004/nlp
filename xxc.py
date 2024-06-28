import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch import flatten
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


# 搭建网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


tudui = Tudui()
print(tudui)

for data in dataloader:
    imgs, targets = data
    print("原尺寸", imgs.shape)  # 【64,3,32,32】
    output = flatten(imgs)
    print("flatten后尺寸", output.shape)
    output = tudui(output)
    print("全连接层输出尺寸", output.shape)
