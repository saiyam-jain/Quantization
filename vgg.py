import tensorflow as tf
#import wandb
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

#wandb.init(project='quantization', entity='saiyam-jain') 

EPOCHS = 20
BATCH_SIZE = 64
LR = 0.0001

#wandb.config = {
 #       "learning_rate": LR,
  #      "epochs": EPOCHS,
   #     "batch_size": BATCH_SIZE
#}

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

y_test = tf.squeeze(y_test, 1)
y_train = tf.squeeze(y_train, 1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = tf.keras.Sequential(
        [
            layers.Conv2D(input_shape=(32,32,3), filters=64, kernel_size=(3,3), padding="same", activation="relu"),
            layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
            layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
            layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
            layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
            layers.Flatten(),
            layers.Dense(units=512,activation="relu"),
            layers.Dense(units=256,activation="relu"),
            layers.Dense(units=10, activation="softmax")
        ]
)

model.summary()

model.compile(optimizer=Adam(learning_rate = LR), loss="categorical_crossentropy", metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=1, validation_data=(x_test, y_test), verbose=2)
print(history.history)
#wandb.tensorflow.log(tf.summary.merge_all())

