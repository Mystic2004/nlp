import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
dataset_transform = transforms.Compose([
    transforms.ToTensor()
])
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False,transform=dataset_transform, download=True)
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
img, target = test_set[0]
print("单个img:",img.shape)
print("单个target:",target)

for data in test_loader:
    imgs,targets = data
    print(imgs.shape)
    print(targets)
    import torchvision
    from torchvision import transforms
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import DataLoader

    dataset_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
    test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    img, target = test_set[0]
    print("单个img:", img.shape)
    print("单个target:", target)

    writer = SummaryWriter("logs")  # 日志文件存储位置
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("test_data", imgs, step)
        step = step + 1

    writer.close()