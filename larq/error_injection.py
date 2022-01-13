import tensorflow as tf
import numpy as np
import larq as lq

model = tf.keras.models.load_model('binary_model_MNIST.h5')

(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

test_images = test_images.reshape((10000, 28, 28, 1))


def error_injection(weight, n_weights, s, p):
    w_conv = weight
    w_conv = np.reshape(w_conv, newshape=[n_weights, 1])
    rng = np.random.default_rng()
    indices = rng.choice(a=n_weights, size=int(p * n_weights / 100), replace=False)
    for i in indices:
        w_conv[i] = np.where(w_conv[i] > 0, -1, 1)
    w_conv = np.reshape(w_conv, newshape=s)
    return w_conv


test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy before error injection {test_acc * 100:.2f} %")

layer = model.get_layer(name='first_conv')
# layer = model.get_layer(name='second_conv')
# layer = model.get_layer(name='third_conv')
# layer = model.get_layer(name='first_dense')
# layer = model.get_layer(name='second_dense')

weights = layer.get_weights()[0]
bias = layer.get_weights()[1]

new_weights = error_injection(weights, weights.size, weights.shape, p=20)
# weights[5] = error_injection(weights[5], 3*3*32*64, weights[5].shape)
# weights[10] = error_injection(weights[10], 3*3*64*64, weights[10].shape)
# weights[15] = error_injection(weights[15], 576*64, weights[15].shape)
# weights[20] = error_injection(weights[20], 64*10, weights[20].shape)

layer.set_weights([new_weights, bias])

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy after 2% error injection {test_acc * 100:.2f} %")
