from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs/logs')
img = Image.open("images/pytorch.png")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

#Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2],[9 , 3, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalization", img_norm,2)

#Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
#img PIL