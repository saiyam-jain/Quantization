# Quantization
### LeNet
 - mixed layer quantization on LeNet using Qkeras
 - the test and train accuracies of different combinaiton of weights are mentioned in quantized_weights.csv
 - All results for LeNet training are available at https://wandb.ai/saiyam-jain/quantization?workspace=user-saiyam-jain
 
### TKOS
  - train an unquantized CNN on TKOS chocolate data.
  - it is clearly evident from training the model that we need mode data for impurities, otherwise the model is overfitting.
  
### Larq
 - train LeNet model for CIFAR and MNIST using Larq
 - perform Error Injections for CIFAR and MNIST data on CNN models using larq quantization
 - All results for larq training are available at https://wandb.ai/saiyam-jain/larq?workspace=user-saiyam-jain
 
### Resnet
 - Train Resnet for CIFAR dataset
 - All results for resnet training are available at https://wandb.ai/saiyam-jain/resnet?workspace=user-saiyam-jain

### virtual environment on HPC:
source /daten/qkeras/bin/activate

### saiyam's folder on HPC:
cd /home/IPMS.FRAUNHOFER.DE/sai69199

