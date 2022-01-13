import tensorflow as tf
import numpy as np
import larq as lq
from EI import error_injection

for i in [2, 5, 10, 15, 20]:
    test_acc = error_injection(layer_name='first_conv', percent=i)
    print(f"Test accuracy after {i}% error injection {test_acc * 100:.2f} %")

