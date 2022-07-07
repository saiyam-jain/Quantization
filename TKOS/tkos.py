import numpy as np
import tensorflow as tf


def load_data():
    images = []
    labels = []
    shape = (80, 401, 301)
    n_files = 1
    train_split = 0.8
    test_split = 0.2

    for i in range(n_files):
        try:
            image = np.fromfile('data/Alpenmilch/Alpenmilch/{:03}_A.bin'.format(i+1), dtype='float32', sep="")
            image = image[3:]
            image = image.reshape(shape)
            images.append(image)
            labels.append(0)
        except:
            continue

    for i in range(n_files):
        try:
            image = np.fromfile('data/Ganze_Haselnuss/Ganze_Haselnuss/{:03}_A.bin'.format(i+1), dtype='float32', sep="")
            image = image[3:]
            image = image.reshape(shape)
            images.append(image)
            labels.append(0)
        except:
            continue

    for i in range(n_files):
        try:
            image = np.fromfile('data/Haselnuss/Haselnuss/{:03}_A.bin'.format(i+1), dtype='float32', sep="")
            image = image[3:]
            image = image.reshape(shape)
            images.append(image)
            labels.append(0)
        except:
            continue

    image = np.fromfile('data/Alpenmilch/Alpenmilch/100_A_impurities.bin', dtype='float32', sep="")
    image = image[3:]
    image = image.reshape(shape)
    images.append(image)
    labels.append(1)

    image = np.fromfile('data/Ganze_Haselnuss/Ganze_Haselnuss/100_A_impurities.bin', dtype='float32', sep="")
    image = image[3:]
    image = image.reshape(shape)
    images.append(image)
    labels.append(1)

    image = np.fromfile('data/Haselnuss/Haselnuss/100_A_impurities.bin', dtype='float32', sep="")
    image = image[3:]
    image = image.reshape(shape)
    images.append(image)
    labels.append(1)

    images = np.array(images)
    n_images = images.shape[0]
    np.random.shuffle(images)
    labels = np.array(labels)
    images = tf.data.Dataset(images)
    labels = tf.data.Dataset(labels)
    labels = tf.keras.utils.to_categorical(labels, 2)
    n_train = int(train_split * n_images)
    train_images = images.take(n_train)
    test_images = images.skip(n_train)
    train_labels = labels.take(n_train)
    test_labels = labels.skip(n_train)
    print(train_images.shape)
    print(test_images.shape)
    print(train_labels.shape)
    print(test_labels.shape)
    #
    # train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(n_train).batch(batch_size)
    # test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

load_data()