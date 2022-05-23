# MLPerf Tiny keyword spotting reference model

This is the MLPerf Tiny keyword spotting reference model.

- Model: Depthwise Separable CNN
- Dataset: Speech Commands

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

# Train and test the model
./keyword_spotting_dscnn.sh
```
