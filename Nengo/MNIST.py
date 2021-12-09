import nengo
import numpy as np
import matplotlib.pyplot as plt
from nengo_extras.data import one_hot_from_labels
import matplotlib.cm as cm
import tensorflow as tf

rng = np.random.RandomState(1)
# --- load the data of training and testing
img_rows, img_cols = 28, 28
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train * 0.5  # ------Normalize to 0 to 128
X_test = X_test * 0.5  # ------Normalize to 0 to 128
T_train = one_hot_from_labels(y_train, classes=10)

def Crossbar_NEF(n_hid):
    # ------Label numbers are from 0 to 9 (10 numbers in total)
    # ------Encode categorical integer features using a one-hot aka one-of-K scheme.
    # for i in range(6):
    #     print("%d -> %s" % (y_train[i], T_train[i]))

    # --- set up network parameters
    n_vis = X_train.shape[1]
    n_out = T_train.shape[1]
    # --- number of hidden units
    # --- More means better performance but longer training time.
    intercepts = np.linspace(-0.0001, + 0.0001, n_hid)  # Introduce normally distr

    ens_params = dict(
        eval_points=X_train,
        neuron_type=nengo.RectifiedLinear(),  # nengo.RectifiedLinear(), # --- Rate-based Intergrate fire neuron
        intercepts=nengo.dists.Choice([0.01]),
        max_rates=nengo.dists.Choice([100]),
        # encoders=nengo.dists.Choice([[1.0]]),
        # max_rates = Uniform(800, 1000),
        # max_rates=nengo.dists.Choice([100]),
    )

    # Least-squares solver with L2 regularization.
    solver = nengo.solvers.LstsqL2(reg=0.01)

    with nengo.Network(seed=3) as model:
        a = nengo.Ensemble(n_hid, n_vis, **ens_params)
        v = nengo.Node(size_in=n_out)
        conn = nengo.Connection(
            a, v, synapse=None,
            eval_points=X_train, function=T_train, solver=solver)

    def get_outs(sim, images):
        # encode the images to get the ensemble activations
        _, acts = nengo.utils.ensemble.tuning_curves(a, sim, inputs=images)

        # decode the ensemble activities using the connection's decoders
        return np.dot(acts, (sim.data[conn].weights.T))

    def get_error(sim, images, labels):
        # the classification for each example is index of
        # the output dimension with the highest value
        return np.argmax(get_outs(sim, images), axis=1) != labels

    def print_error(sim):
        train_error = 100 * get_error(sim, X_train, y_train).mean()
        test_error = 100 * get_error(sim, X_test, y_test).mean()
        return test_error

    encoders = np.round(rng.normal(-0.45, +0.45, size=(n_hid, 28 * 28)))
    a.encoders = encoders

    # X = encoders
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.xlabel('784 pixels in one pattern')
    # plt.ylabel('1000 neurons in hidden layer')
    # plt.title('Encoder weight')
    # ax.imshow(X, cmap=cm.Reds, interpolation='nearest',aspect='auto')
    # numrows, numcols = X.shape
    #
    # def format_coord(x, y):
    #     col = int(x+0.5)
    #     row = int(y+0.5)
    #     if col>=0 and col<numcols and row>=0 and row<numrows:
    #         z = X[row,col]
    #         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    #     else:
    #         return 'x=%1.4f, y=%1.4f'%(x, y)
    # ax.format_coord = format_coord
    # plt.show()
    #
    # with nengo.Simulator(model) as sim:
    #     Y = sim.data[conn].weights.T
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.xlabel('10 output neurons')
    # plt.ylabel('1000 neurons in hidden layer')
    # plt.title('Decoder weight')
    # ax.imshow(Y, cmap=plt.cm.Reds, interpolation='nearest',aspect='auto')
    #
    # numrows, numcols = Y.shape
    with nengo.Simulator(model) as sim:
        print_error(sim)
    return print_error(sim)


n = 2
N = np.zeros(n)

for i in range(1, n):
    N[i] = Crossbar_NEF(i)

plt.plot(N)
plt.show()
