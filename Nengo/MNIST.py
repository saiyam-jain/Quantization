import nengo
import time
import numpy as np
import multiprocessing             # multiprocessing: 
import matplotlib.pyplot as plt
from nengo_extras.data import load_mnist, one_hot_from_labels
from nengo_extras.vision import Gabor, Mask
from nengo_extras.matplotlib import tile
from nengo.dists import Uniform
import matplotlib.cm as cm
from  multiprocessing import Pool

rng = np.random.RandomState(1)
# --- load the data of training and testing
img_rows, img_cols = 28, 28
(X_train, y_train), (X_test, y_test) = load_mnist()
X_train = (256 * X_train  - 128)  #------Normalize to 0 to 128
X_test = (256 * X_test - 128)    #------Normalize to 0 to 128
T_train = one_hot_from_labels(y_train, classes=10)

def Crossbar_NEF (n_hid):
  
