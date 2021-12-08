import matplotlib.pyplot as plt
import nengo
import numpy as np

from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask


def one_hot(labels, c=None):
    assert labels.ndim == 1
    n = labels.shape[0]
    c = len(np.unique(labels)) if c is None else c
    y = np.zeros((n, c))
    y[np.arange(n), labels] = 1
    return y


img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = 2 * X_train - 1  # normalize to -1 to 1
X_test = 2 * X_test - 1  # normalize to -1 to 1

train_targets = one_hot(y_train, 10)
test_targets = one_hot(y_test, 10)

rng = np.random.RandomState(9)

# --- set up network parameters
# Want to encode and decode the image
n_vis = X_train.shape[1]
n_out = X_train.shape[1]
# number of neurons/dimensions of semantic pointer
n_hid = 1000  # Try with more neurons for more accuracy

# Want the encoding/decoding done on the training images
ens_params = dict(
    eval_points=X_train,
    neuron_type=nengo.LIF(),  # Why not use LIF? originally used LIFRate()
    intercepts=nengo.dists.Choice([-0.5]),
    max_rates=nengo.dists.Choice([100]),
)

# Least-squares solver with L2 regularization.
solver = nengo.solvers.LstsqL2(reg=0.01)
# solver = nengo.solvers.LstsqL2(reg=0.0001)
solver2 = nengo.solvers.LstsqL2(reg=0.01)

# network that generates the weight matrices between neuron activity and images and the labels
with nengo.Network(seed=3) as model:
    a = nengo.Ensemble(n_hid, n_vis, seed=3, **ens_params)
    v = nengo.Node(size_in=n_out)
    conn = nengo.Connection(
        a, v, synapse=None,
        eval_points=X_train, function=X_train,  # want the same thing out (identity)
        solver=solver)

    v2 = nengo.Node(size_in=train_targets.shape[1])
    conn2 = nengo.Connection(
        a, v2, synapse=None,
        eval_points=X_train, function=train_targets,  # Want to get the labels out
        solver=solver2)

# linear filter used for edge detection as encoders, more plausible for human visual system
encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)
# Set the ensembles encoders to this
a.encoders = encoders

# Check the encoders were correctly made
plt.imshow(encoders[0].reshape(28, 28), vmin=encoders[0].min(), vmax=encoders[0].max(), cmap='gray')
plt.show()
