import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs") # 日志文件存储位置
dataset_transform = transforms.Compose([
    transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True,transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False,transform=dataset_transform, download=True)
print(test_set[0])
img, target = test_set[0]  # target对应类的编号 对应cat
print(img)
print(target)
print(test_set.classes[target])

for i in range(10):
    img, target = test_set[i]
    writer.add_image("torchvision",img,i)