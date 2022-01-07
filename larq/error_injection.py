import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('binary_model.h5')

(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

test_images = test_images.reshape((10000, 28, 28, 1))


def error_injection(weight, n_weights, s, p=2):
    w_conv = weight
    w_conv = np.reshape(w_conv, newshape=[n_weights, 1])
    rng = np.random.default_rng()
    indices = rng.choice(a=n_weights, size=int(p * n_weights / 100), replace=False)
    for i in indices:
        w_conv[i] = np.where(w_conv[i] > 0, -1, 1)
    w_conv = np.reshape(w_conv, newshape=s)
    return w_conv


weights = model.get_weights()

weights[0] = error_injection(weights[0], 3*3*32, weights[0].shape)
# weights[5] = error_injection(weights[5], 3*3*32*64, weights[5].shape)
# weights[10] = error_injection(weights[10], 3*3*64*64, weights[10].shape)
# weights[15] = error_injection(weights[15], 576*64, weights[15].shape)
# weights[20] = error_injection(weights[20], 64*10, weights[20].shape)

model.set_weights(weights)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy after 2% error injection {test_acc * 100:.2f} %")
