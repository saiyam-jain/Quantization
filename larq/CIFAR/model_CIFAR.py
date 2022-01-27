import tensorflow as tf
import larq as lq
import wandb

wandb.init(project="larq", entity="saiyam-jain", group="CIFAR", job_type="train")

batch_size = 64
EPOCHS = 100
lr = 0.01
decay = 0.0001

wandb.config = {
    "epochs": EPOCHS,
    "batch_size": batch_size,
    "lr": lr
}

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
n_train = train_images.shape[0]
n_test = test_images.shape[0]
num_classes = 10

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

train_images = train_images.reshape((n_train, 32, 32, 3)).astype("float32")
test_images = test_images.reshape((n_test, 32, 32, 3)).astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(n_train).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

kwargs = dict(input_quantizer="SteHeaviside",
              kernel_quantizer="SteHeaviside",
              kernel_constraint="weight_clip",
              use_bias=False)

model = tf.keras.models.Sequential()

# In the first layer we only quantize the weights and not the input
model.add(lq.layers.QuantConv2D(128, 3,
                                kernel_quantizer="SteHeaviside",
                                kernel_constraint="weight_clip",
                                use_bias=False,
                                input_shape=(32, 32, 3), name='first_conv'))
model.add(tf.keras.layers.BatchNormalization(momentum=0.999, scale=False))

model.add(lq.layers.QuantConv2D(128, 3, padding="same", name='second_conv', **kwargs))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.BatchNormalization(momentum=0.999, scale=False))

model.add(lq.layers.QuantConv2D(256, 3, padding="same", name='third_conv', **kwargs))
model.add(tf.keras.layers.BatchNormalization(momentum=0.999, scale=False))

model.add(lq.layers.QuantConv2D(256, 3, padding="same", name='fourth_conv', **kwargs))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.BatchNormalization(momentum=0.999, scale=False))

model.add(lq.layers.QuantConv2D(512, 3, padding="same", name='fifth_conv', **kwargs))
model.add(tf.keras.layers.BatchNormalization(momentum=0.999, scale=False))

model.add(lq.layers.QuantConv2D(512, 3, padding="same", name='sixth_conv', **kwargs))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.BatchNormalization(momentum=0.999, scale=False))

model.add(tf.keras.layers.Flatten())

model.add(lq.layers.QuantDense(1024, name='first_dense', **kwargs))
model.add(tf.keras.layers.BatchNormalization(momentum=0.999, scale=False))

model.add(lq.layers.QuantDense(1024, name='second_dense', **kwargs))
model.add(tf.keras.layers.BatchNormalization(momentum=0.999,scale=False))

model.add(lq.layers.QuantDense(10, name='third_dense', **kwargs))
model.add(tf.keras.layers.BatchNormalization(momentum=0.999,scale=False))

model.add(tf.keras.layers.Activation("softmax"))

lq.models.summary(model)

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, decay=decay)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

model.compile(optimizer, loss_object, train_accuracy)

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


for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
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

    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss.result().numpy(),
        "Train Accuracy": train_accuracy.result().numpy(),
        "Test Loss": test_loss.result().numpy(),
        "Test Accuracy": test_accuracy.result().numpy()
    })


model.save("full_precision_model_CIFAR.h5")

fp_weights = model.get_weights()

with lq.context.quantized_scope(True):
    model.save("binary_model_CIFAR.h5")
    weights = model.get_weights()


test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy before error injection {test_acc * 100:.2f} %")
