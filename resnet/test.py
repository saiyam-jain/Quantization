import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize])

cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

train_images = []
train_labels = []

for i, (x, y) in enumerate(cifar_trainset):
    train_images.append(x.numpy())
    train_labels.append(y)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

n_train = train_images.shape[0]
num_classes = 10

train_images = train_images.reshape((n_train, 32, 32, 3))

train_images = tf.convert_to_tensor(train_images)
train_labels = tf.convert_to_tensor(train_labels)

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)

print(train_images.shape)
print(train_images.dtype)
print(train_images.min)
print(train_images.max)
print(train_labels.shape)
print(train_labels.dtype)
