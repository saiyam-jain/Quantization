import tensorflow as tf
from qkeras import *
import wandb

wandb.init(project="quantization", entity="saiyam-jain")

batch_size = 64
EPOCHS = 50
lr = 0.01
decay = 0.0001

wandb.config = {
    "epochs": EPOCHS,
    "batch_size": batch_size,
    "lr": lr
}

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
n_train = train_images.shape[0]
n_test = test_images.shape[0]
num_classes = 10

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

train_images = train_images/255.0
test_images = test_images/255.0

train_images = train_images.reshape((n_train, 28, 28, 1))
test_images = test_images.reshape((n_test, 28, 28, 1))

train_images = tf.keras.layers.ZeroPadding2D(padding=2)(train_images)
test_images = tf.keras.layers.ZeroPadding2D(padding=2)(test_images)

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(n_train).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)


def createmodel(first, second, third, fourth, fifth):
    model = tf.keras.models.Sequential()
    model.add(QConv2D(filters=6, kernel_size=(3, 3),
                      kernel_quantizer=quantized_bits(first, 0, 1),
                      bias_quantizer=quantized_bits(first, 0, 1),
                      activation='relu',
                      input_shape=(32, 32, 1)))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(QConv2D(filters=16, kernel_size=(3, 3),
                      kernel_quantizer=quantized_bits(second, 0, 1),
                      bias_quantizer=quantized_bits(second, 0, 1),
                      activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(QDense(units=120, activation='relu',
                     kernel_quantizer=quantized_bits(third, 0, 1),
                     bias_quantizer=quantized_bits(third, 0, 1)))
    model.add(QDense(units=84, activation='relu',
                     kernel_quantizer=quantized_bits(fourth, 0, 1),
                     bias_quantizer=quantized_bits(fourth, 0, 1)))
    model.add(QDense(units=10, activation='softmax',
                     kernel_quantizer=quantized_bits(fifth, 0, 1),
                     bias_quantizer=quantized_bits(fifth, 0, 1)))
    return model

# model.summary()


loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, decay=decay)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels, model):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels, model):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


for fifth in [8, 6, 4, 2]:
    for fourth in [8, 6, 4, 2]:
        for third in [8, 6, 4, 2]:
            for second in [8, 6, 4, 2]:
                for first in [8, 6, 4, 2]:
                    model = createmodel(first, second, third, fourth, fifth)
                    model.compile(optimizer, loss_object, train_accuracy)
                    wandb.log({
                        "first layer": first,
                        "second layer": second,
                        "third layer": third,
                        "fourth layer": fourth,
                        "fifth layer": fifth
                    })
                    for epoch in range(EPOCHS):
                        # Reset the metrics at the start of the next epoch
                        train_loss.reset_states()
                        train_accuracy.reset_states()
                        test_loss.reset_states()
                        test_accuracy.reset_states()

                        for images, labels in train_ds:
                            train_step(images, labels, model)

                        for test_images, test_labels in test_ds:
                            test_step(test_images, test_labels, model)

                        print(
                            f'Epoch {epoch + 1}, '
                            f'Loss: {train_loss.result()}, '
                            f'Accuracy: {train_accuracy.result() * 100}, '
                            f'Test Loss: {test_loss.result()}, '
                            f'Test Accuracy: {test_accuracy.result() * 100}'
                        )

                        wandb.log({
                            "Epoch": epoch + 1,
                            "Train Loss": train_loss.result().numpy(),
                            "Train Accuracy": train_accuracy.result().numpy(),
                            "Test Loss": test_loss.result().numpy(),
                            "Test Accuracy": test_accuracy.result().numpy()
                        })
