import tensorflow as tf
import numpy as np
import larq as lq
import wandb

wandb.init(project="larq", entity="saiyam-jain", group="MNIST_1", job_type="mixed_layer_2%EI")


def flip_weights(weight, n_weights, s, p):
    w_conv = weight
    w_conv = np.reshape(w_conv, newshape=[n_weights, 1])
    rng = np.random.default_rng()
    indices = rng.choice(a=n_weights, size=int(p * n_weights / 100), replace=False)
    for i in indices:
        w_conv[i] = np.where(w_conv[i] > 0, -1, 1)
    w_conv = np.reshape(w_conv, newshape=s)
    return w_conv


def error_injection():
    model = tf.keras.models.load_model('binary_model_MNIST.h5')
    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    n_test = test_images.shape[0]
    test_images = test_images.reshape((n_test, 28, 28, 1))
    # test_images = test_images.reshape((10000, 28, 28, 1))
    # test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    params = [['first_conv', 2], ['second_conv', 5], ['third_conv', 2], ['first_dense', 5], ['second_dense', 2]]

    # layers = ['first_conv', 'second_conv', 'third_conv', 'first_dense', 'second_dense']

    for param in params:
        layer = param[0]
        percent = param[1]
        layer = model.get_layer(name=layer)
        weights = layer.get_weights()[0]
        # bias = layer.get_weights()[1]
        new_weights = flip_weights(weights, weights.size, weights.shape, p=percent)
        layer.set_weights([new_weights])

    _, test_accc = model.evaluate(test_images, test_labels)

    # print(f"Test accuracy after 2% error injection {test_acc * 100:.2f} %")
    return test_accc


runs = 100

# for p in [2, 3, 5, 10, 15, 20]:

accuracies = []
for run in range(runs):
    test_acc = error_injection()
    accuracies.append(test_acc)
    wandb.log({'Test acc': test_acc})
average = sum(accuracies)/len(accuracies)
print(f"Average test accuracy on {runs} runs of mixed error injection is: {average * 100:.2f} %")
# wandb.log({'Avg test acc': average})
