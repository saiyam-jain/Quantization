import tensorflow as tf
import wandb
from tensorflow.keras import layers
import numpy
from tensorflow.keras.models import Model
import qkeras
from tensorflow.keras.layers import Input

wandb.init(project='quantization', entity='saiyam-jain')

EPOCHS = 200
BATCH_SIZE = 32
LR = 0.0001

wandb.config = {
       "learning_rate": LR,
       "epochs": EPOCHS,
       "batch_size": BATCH_SIZE
}

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

trainX = x_train.astype('float32') / 255.0
testX = x_test.astype('float32') / 255.0

# trainY = tf.one_hot(y_train, 10)
# testY = tf.one_hot(y_test, 10)

trainY = tf.squeeze(y_train, 1)
testY = tf.squeeze(y_test, 1)

train_ds = tf.data.Dataset.from_tensor_slices((trainX, trainY)).shuffle(10000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((testX, testY)).batch(BATCH_SIZE)

x = x_in = Input(shape=(32, 32, 3))
x = QConv2D(
    32, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = QConv2D(
    32, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = QConv2D(
    64, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = QConv2D(
    64, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = QConv2D(
    128, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = QConv2D(
    128, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = QConv2D(
    128, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = QConv2D(
    256, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = QConv2D(
    256, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = QConv2D(
    256, (3, 3), padding="same",
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = layers.Flatten()(x)
x = QDense(
    units=256, kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = QDense(
    units=128, kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x = QActivation("quantized_relu(4,0)")(x)
x = QDense(
    units=10, kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1))(x)
x_out = x = layers.Activation("softmax")(x)

model = Model(inputs=[x_in], outputs=[x_out])
model.summary()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
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

    print(" Epoch: ", epoch + 1,
          " Train Loss: ", train_loss.result(),
          " Train Accuracy: ", train_accuracy.result() * 100,
          " Validation Loss: ", test_loss.result(),
          " Validation Accuracy: ", test_accuracy.result() * 100)

    wandb.log({"epoch": epoch+1,
               "train_loss": train_loss.result().numpy(),
               "train_accuracy": train_accuracy.result().numpy(),
               "val_loss": test_loss.result().numpy(),
               "val_accuracy": test_accuracy.result().numpy()})
