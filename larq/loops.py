import tensorflow as tf
import numpy as np
import larq as lq
from EI import error_injection
import wandb

wandb.init(project="larq", entity="saiyam-jain", group="MNIST", job_type="error_injection")

runs = 10
layers = ['first_conv', 'second_conv', 'third_conv', 'first_dense', 'second_dense']

for layer in layers:
    for p in [2, 5, 10, 15, 20]:
        accuracies = []
        for run in range(runs):
            test_acc = error_injection(layer_name=layer, percent=p)
            accuracies.append(test_acc)
            wandb.log({'Test acc': test_acc, 'error percent': p, 'layer': layer})
        average = sum(accuracies)/len(accuracies)
        print(f"Average test accuracy on {runs} runs of {p}% error injection on {layer} layer is: {test_acc * 100:.2f} %")
        wandb.log({'Avg test acc': average})
