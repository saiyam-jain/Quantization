import numpy as np
import tensorflow as tf


shape = (301, 401, 80)
EPOCHS = 10


def load_data():
    images = []
    labels = []
    n_files = 100
    train_split = 0.8

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
    labels = np.array(labels)
    n_images = images.shape[0]
    n_train = int(train_split * n_images)
    train_images = tf.convert_to_tensor(images[0:n_train, :])
    test_images = tf.convert_to_tensor(images[n_train:, :])
    train_labels = tf.convert_to_tensor(labels[0:n_train])
    test_labels = tf.convert_to_tensor(labels[n_train:])
    train_labels = tf.keras.utils.to_categorical(train_labels, 2)
    test_labels = tf.keras.utils.to_categorical(test_labels, 2)

    train_images = train_images.reshape((n_train, 301, 401, 80))
    test_images = test_images.reshape((n_images - n_train, 301, 401, 80))

    print(train_images.shape)
    print(test_images.shape)
    print(train_labels.shape)
    print(test_labels.shape)

    train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(n_train)
    test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    return train, test


def create_model():
    tf_model = tf.keras.Sequential()

    tf_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=shape))
    tf_model.add(tf.keras.layers.MaxPool2D(2, 2))
    tf_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    tf_model.add(tf.keras.layers.MaxPool2D(2, 2))
    tf_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    tf_model.add(tf.keras.layers.MaxPool2D(2, 2))
    tf_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    tf_model.add(tf.keras.layers.MaxPool2D(2, 2))
    tf_model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    tf_model.add(tf.keras.layers.MaxPool2D(2, 2))
    tf_model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    tf_model.add(tf.keras.layers.MaxPool2D(2, 2))
    tf_model.add(tf.keras.layers.Flatten())
    tf_model.add(tf.keras.layers.Dense(512, activation='relu'))
    tf_model.add(tf.keras.layers.Dense(256, activation='relu'))
    tf_model.add(tf.keras.layers.Dense(2, activation='softmax'))

    tf_model.summary()
    return tf_model


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


model = create_model()
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

model.compile(optimizer, loss_object, train_accuracy)
train_ds, test_ds = load_data()

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
