from PIL.Image import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#python的用法 ->tensor数据类型
#通过transforms.ToTensor去解决两个问题

#2.为什么需要Tensor数据类型

#绝对路径：C:\Users\刘桦桦\Desktop\Learn_Pytorch\Pytorchlearn\dataset\train\ants\0013035.jpg
#相对路径：dataset/train/ants/0013035.jpg
img_path = "C:\\Users\\刘桦桦\\Desktop\\Learn_Pytorch\\Pytorchlearn\\dataset\\train\\ants\\0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs/logs")


#1.Transforms该如何使用(python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_Image",tensor_img)
writer.close()