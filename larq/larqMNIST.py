import tensorflow as tf
import larq as lq

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between -1 and 1
# train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

# All quantized layers except the first will use the same options
kwargs = dict(input_quantizer="SteHeaviside",
              kernel_quantizer="SteHeaviside",
              kernel_constraint="weight_clip")

model = tf.keras.models.Sequential()

# In the first layer we only quantize the weights and not the input
model.add(lq.layers.QuantConv2D(16, (3, 3),
                                kernel_quantizer="SteHeaviside",
                                kernel_constraint="weight_clip",
                                input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(lq.layers.QuantConv2D(32, (3, 3), **kwargs))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(lq.layers.QuantConv2D(64, (3, 3), **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Flatten())

model.add(lq.layers.QuantDense(16, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(lq.layers.QuantDense(10, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Activation("softmax"))

lq.models.summary(model)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=64, epochs=10)

#print(model.layers[1].weights)
#print(model.layers[1].bias.numpy())
#print(model.layers[1].bias_initializer)
#print(model.trainable_variables)

model.save("full_precision_model.h5")

fp_weights = model.get_weights()

with lq.context.quantized_scope(True):
    model.save("binary_model.h5")
    weights = model.get_weights()

print(fp_weights)
print(weights)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy {test_acc * 100:.2f} %")

