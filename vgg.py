import tensorflow as tf
import wandb
from tensorflow.keras import layers

wandb.init(project='quantization', entity='saiyam-jain')

EPOCHS = 100
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

model = tf.keras.Sequential(
        [
            layers.Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2)),
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(units=256, activation="relu"),
            layers.Dense(units=128, activation="relu"),
            layers.Dense(units=10, activation="softmax")
        ]
)

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
               "train_loss": train_loss.numpy(),
               "train_accuracy": train_accuracy.numpy(),
               "val_loss": test_loss.numpy(),
               "val_accuracy": test_accuracy.numpy()
               })
