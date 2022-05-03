import tensorflow as tf
import resnet

model = resnet.resnet20()

batch_size = 64
EPOCHS = 150
lr = 0.01
decay = 0.0001

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
n_train = train_images.shape[0]
n_test = test_images.shape[0]
num_classes = 10

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)

train_images = train_images.reshape((n_train, 32, 32, 3)).astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(n_train).batch(batch_size)

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, decay=decay)

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


for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
    )
