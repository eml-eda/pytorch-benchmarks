# MLPerf Tiny image classification reference model

This is the MLPerf Tiny image classification reference model.

A ResNet8 model is trained on the CIFAR10 dataset available at:
https://www.cs.toronto.edu/~kriz/cifar.html

- Model: ResNet8
- Dataset: Cifar10

## Quick start

To run the code and replicate the results it is suggested to create a new python virtual environment and install the required libraries:
``` Bash

# Create a new python virtual environment 
python3 -m venv <environment-name>

# Activate the virtual environment 
source <environment-name>/bin/activate

# Upgrade pip to the latest version
python3 -m pip install --upgrade pip

# Install the required libraries
pip install -r requirements.txt

# Uninstall all the libraries not present in the requirements file
pip freeze | grep -v -f requirements.txt - | xargs pip uninstall -y
```

Run the following commands to go through the whole training and validation process

``` Bash

# Download training, train model, test the model
./cifar10_resnet.sh
```
