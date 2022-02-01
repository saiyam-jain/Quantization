import tensorflow as tf
import numpy as np
import larq as lq
import wandb

wandb.init(project="larq", entity="saiyam-jain", group="CIFAR_SEI", job_type="error_injection")


def flip_weights(weight, n_weights, s, p, digit, print_p):
    w_conv = weight
    unique, counts = np.unique(w_conv, return_counts=True)
    zeros = counts[0]
    ones = counts[1]
    w_conv = np.reshape(w_conv, newshape=[n_weights, ])
    sort = np.argsort(w_conv)
    rng = np.random.default_rng()

    if print_p == 1:
        wandb.log({'Percent of -1 weights': zeros/counts.sum()*100, 'Percent of +1 weights': ones/counts.sum()*100})

    if digit == 0:
        indices = rng.choice(a=zeros, size=int(p * zeros / 100), replace=False)
        for i in indices:
            w_conv[sort[i]] = np.where(w_conv[i] < 0, 1, -1)
    elif digit == 1:
        indices = rng.choice(a=ones, size=int(p * ones / 100), replace=False)
        for i in indices:
            w_conv[sort[i+zeros-1]] = np.where(w_conv[i] > 0, -1, 1)

    w_conv = np.reshape(w_conv, newshape=s)
    return w_conv


def error_injection(layer_name, percent, print_p):
    model = tf.keras.models.load_model('binary_model_CIFAR.h5')
    (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    n_test = test_images.shape[0]
    test_images = test_images.reshape((n_test, 32, 32, 3))
    # test_images = test_images.reshape((10000, 28, 28, 1))
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    layer = model.get_layer(name=layer_name)

    weights = layer.get_weights()[0]
    # bias = layer.get_weights()[1]

    new_weights = flip_weights(weights, weights.size, weights.shape, p=percent, digit=0, print_p=print_p)

    layer.set_weights([new_weights])

    _, test_acc = model.evaluate(test_images, test_labels)

    # print(f"Test accuracy after 2% error injection {test_acc * 100:.2f} %")
    return test_acc

