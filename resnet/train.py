import resnet
import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

# print(model_names)

batch_size = 64
lr = 0.1
momentum = 0.9
EPOCHS = 200
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize
])

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
tf.keras.backend.clear_session()

cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    normalize
]))

# test_images = []
# test_labels = []
#
# for i, (x, y) in enumerate(cifar_testset):
#     test_images.append(x.numpy())
#     test_labels.append(y)
#
# test_images = np.array(test_images)
# test_labels = np.array(test_labels)
#
# n_test = test_images.shape[0]
#
# test_images = test_images.reshape((n_test, 32, 32, 3))
#
# test_images = tf.convert_to_tensor(test_images)
# test_labels = tf.convert_to_tensor(test_labels)
#
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)
#
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(n_train).batch(batch_size)
# test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

model = resnet.resnet20()


# loss_object = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
#
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
# # test_loss = tf.keras.metrics.Mean(name='test_loss')
# # test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
#
# model.compile(optimizer, loss_object, train_accuracy)

model.summary()

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

model.compile(optimizer, loss_object, train_accuracy)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


# @tf.function
# def test_step(images, labels):
#     predictions = model(images, training=False)
#     t_loss = loss_object(labels, predictions)
#     test_loss(t_loss)
#     test_accuracy(labels, predictions)


for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    # test_loss.reset_states()
    # test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    # for test_images, test_labels in test_ds:
    #     test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        # f'Test Loss: {test_loss.result()}, '
        # f'Test Accuracy: {test_accuracy.result() * 100}'
    )