import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import numpy as np
import time
import tensorflow as tf

print("Hello :)")
print("GPU available(T/F): ", tf.config.list_physical_devices('GPU'))
print("GPU:", tf.test.is_gpu_available())

np.random.seed(42)

NB_EPOCH = 1
BATCH_SIZE = 64
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.0001, decay=0.000025)
VALIDATION_SPLIT = 0.1

train = 1

(x_train, y_train), (x_test, y_test) = mnist.load_data()
RESHAPED = 784

x_test_orig = x_test

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

x_train /= 256.0
x_test /= 256.0

y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

def CreateModel(shape, nb_classes):
    x = x_in = Input(shape)
    x = Conv2D(32, (2, 2), strides=(2,2), name="conv2d_0_m")(x)
    x = Activation("relu", name="act0_m")(x)
    x = Conv2D(64, (3, 3), strides=(2,2), name="conv2d_1_m")(x)
    x = Activation("relu", name="act1_m")(x)
    x = Conv2D(64, (2, 2), strides=(2,2), name="conv2d_2_m")(x)
    x = Activation("relu", name="act2_m")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(nb_classes, name="dense")(x)
    x = Activation("softmax", name="softmax")(x)
    
    model = Model(inputs=x_in, outputs=x)

    return model

#model = CreateModel(x_train.shape[1:], y_train.shape[-1])
#model.summary()
#model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

#tic = time.time()
#model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, initial_epoch=1, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
#toc = time.time()
#print("Unquantized model trained on HPC (MNIST data). Time taken:", toc - tic)

