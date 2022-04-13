# MLPerf Tiny visual wake words reference model

This is the MLPerf Tiny visual wake words reference model.

A MobileNet model is trained on the COCO dataset.

## Quick start

Run the following commands to go through the whole training and validation process

``` Bash

# If the dataset has not been already download, run the following script
./download_dataset.sh
```
``` Bash

# Train and test the model
./coco_mobilenet.sh
```
