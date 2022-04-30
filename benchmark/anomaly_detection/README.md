# MLPerf Tiny anomaly detection reference model

This is the MLPerf Tiny anomaly detection reference model.

An autoencoder model is trained on the ToyCar dataset.

- Model: autoencoder
- Dataset: ToyCar

## Quick start

Run the following commands to go through the whole training and validation process

``` Bash

# If the dataset has not been already download, run the following script
./download_dataset.sh
```
``` Bash

# Train and test the model
./toycar_autoencoder.sh
```
