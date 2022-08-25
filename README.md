# Quantization
### LeNet
 - mixed layer quantization on LeNet using Qkeras
 - the test and train accuracies of different combinaiton of weights are mentioned in quantized_weights.csv
 
### TKOS
  - train an unquantized CNN on TKOS chocolate data.
  - it is clearly evident from training the model that we need mode data for impurities, otherwise the model is overfitting.
  
### Larq
 - train LeNet model for CIFAR and MNIST using Larq
 - perform Error Injections for CIFAR and MNIST data on CNN models using larq quantization
 
### Resnet
 - Train Resnet for CIFAR dataset
