import tensorflow as tf
import numpy as np
import larq as lq


def flip_weights(weight, n_weights, s, p):
    w_conv = weight
    w_conv = np.reshape(w_conv, newshape=[n_weights, 1])
    rng = np.random.default_rng()
    indices = rng.choice(a=n_weights, size=int(p * n_weights / 100), replace=False)
    for i in indices:
        w_conv[i] = np.where(w_conv[i] > 0, -1, 1)
    w_conv = np.reshape(w_conv, newshape=s)
    return w_conv


def error_injection(layer_name, percent):
    model = tf.keras.models.load_model('binary_model_CIFAR.h5')
    (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    n_test = test_images.shape[0]
    test_images = test_images.reshape((n_test, 32, 32, 3))
    # test_images = test_images.reshape((10000, 28, 28, 1))
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    layer = model.get_layer(name=layer_name)

    weights = layer.get_weights()[0]
    # bias = layer.get_weights()[1]

    new_weights = flip_weights(weights, weights.size, weights.shape, p=percent)

    layer.set_weights([new_weights])

    _, test_acc = model.evaluate(test_images, test_labels)

    # print(f"Test accuracy after 2% error injection {test_acc * 100:.2f} %")
    return test_acc

