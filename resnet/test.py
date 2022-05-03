import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize])

cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

arr = []
for i, (x, y) in enumerate(cifar_trainset):
    arr.append(x.numpy())
    if i==3:
        break

array = np.array(arr)
print(array.shape)
