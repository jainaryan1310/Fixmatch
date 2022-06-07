from dataset.imagenet import Idata
from models.resnet import ResNet18
import torchvision.transforms as transforms

model = ResNet18()
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32,
                          padding=int(32*0.125),
                          padding_mode='reflect'),
    transforms.ToTensor()
])
dataset = Idata(transform=train_transform)
print(len(dataset[:-10000]))
