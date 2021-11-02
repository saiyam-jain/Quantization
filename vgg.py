import tensorflow as tf
# import wandb
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# wandb.init(project='quantization', entity='saiyam-jain')

EPOCHS = 100
BATCH_SIZE = 64
LR = 0.0001

# wandb.config = {
#        "learning_rate": LR,
#        "epochs": EPOCHS,
#        "batch_size": BATCH_SIZE
# }

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

trainX = x_train.astype('float32') / 255.0
testX = x_test.astype('float32') / 255.0

trainY = tf.one_hot(y_train, 10)
testY = tf.one_hot(y_test, 10)

trainY = tf.squeeze(trainY, 1)
testY = tf.squeeze(testY, 1)

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

model.compile(optimizer=Adam(learning_rate = LR), loss="categorical_crossentropy", metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=1, validation_data=(testX, testY), verbose=2)
print(history.history)
# wandb.tensorflow.log(tf.summary.merge_all())

