import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sigmoid

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 搭建网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


writer = SummaryWriter("logs")  # 日志文件存储位置
tudui = Tudui()
print(tudui)

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    writer.add_images("before_activate", imgs, step)
    writer.add_images("after_activate", output, step)
    step = step + 1
writer.close()