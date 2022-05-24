# MLPerf Tiny visual wake words reference model

This is the MLPerf Tiny visual wake words reference model.

- Model: MobileNet
- Dataset: COCO

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

# If the dataset has not been already download, run the following script
./download_dataset.sh
```
``` Bash

# Train and test the model
./coco_mobilenet.sh
```
