import tensorflow as tf
import numpy as np
import larq as lq
from EI import error_injection

accuracies = []
runs = 100
layer = 'first_conv'

for p in [2, 5, 10, 15, 20]:
    for run in range(runs):
        test_acc = error_injection(layer_name=layer, percent=p)
        accuracies.append(test_acc)
    average = sum(accuracies)/len(accuracies)
    print(f"Average test accuracy on {runs} runs of {p}% error injection on {layer} layer is: {test_acc * 100:.2f} %")
