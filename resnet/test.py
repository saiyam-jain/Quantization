import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize
])

cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

i=0
for (x,y) in cifar_trainset:
    print(x.size())
    print(y.size())
    if i==2:
        break