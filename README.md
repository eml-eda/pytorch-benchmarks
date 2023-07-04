# Pytorch Benchmarks
`pytorch-benchmarks` is the benchmark library of the **eml** branch of the **EDA** group within Politecnico di Torino.

The library is entirely written in pytorch and addresses the training of DNNs models on edge-relevant use-cases.

In its latest release, the library currently includes the following benchmarks:
1. [Image Classification](./pytorch_benchmarks/image_classification/) on the CIFAR10 dataset.
2. [Keyword Spotting](./pytorch_benchmarks/keyword_spotting/) on the Google Speech Commands v2 dataset.
3. [Visual Wake Words](./pytorch_benchmarks/visual_wake_words/) on the MSCOCO dataset.
4. [Anomaly Detection](./pytorch_benchmarks/anomaly_detection/) on the ToyADMOS dataset.
5. [Heart Rate Detection](./pytorch_benchmarks/hr_detection/) on the PPG-DALIA dataset.
6. [TinyImageNet](./pytorch_benchmarks/tiny_imagenet/) on the omonimous dataset.
7. [Gesture Recognition](./pytorch_benchmarks/gesture_recognition/) on the NinaProDB6 dataset.
8. [Image Classification - ViT](./pytorch_benchmarks/transformers/image_classification/) for Vision Transformers on the CIFAR10 and Tiny-ImageNet datasets.
9. [InfraRed Person Counting](./pytorch_benchmarks/LINAIGE_Kaggle/) on the LINAIGE dataset.

N.B., tasks from 1. to 4. represent our *in-house* implementation of the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark suite (originally implemented in the `tf-keras` framework).

## Installation
To install the latest release:

```
$ git clone https://github.com/eml-eda/pytorch-benchmarks
$ cd pytorch-benchmarks
$ python setup.py install
```

## API Details
Each benchmark is a stand-alone python module based on three python files, namely:
1. [`data.py`](#datapy)
2. [`model.py`](#modelpy)
3. [`train.py`](#trainpy)
4. [`__init__.py`](#__init__py)

#### **`data.py`**
This module **must** implement all the functions needed to gather the data, eventually pre-process them and finally ship them to the user both in the form of [Pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

The two mandatory and standard functions that need to be implemented are:
- `get_data`, which returns a tuple of [Pytorch Datasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). Depending on the task, the number of returned datasets may vary from 2 (train and test) to 3 (train, validation and test). Conversely, the function arguments depends on the specific task.
- `build_dataloaders`, which returns a tuple of [Pytorch Dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). In general, takes as inputs the dataset returned by `get_data` and constants such as the *batch-size* and the *number of workers*. The number of elements of the returned tuple will depends on the number of provided datasets.

#### **`model.py`**
This module **must** implement at least one model for the specific benchmark.

The mandatory and standard function that needs to be implemented is:
- `get_reference_model`, the function always take as first argument the *model_name* which is a string associated to a specific pytorch model. Optionally, the function can take as argument *model_config* i.e., a python dictionary of additional configurations for the model. It returns the requested pytorch model.

If the provided *model_name* is not supported an error is raised.

#### **`train.py`**
This module **must** implement the minimum set of information required to implement a training loop.

In particular, the mandatory and standard functions that needs to be implemented are:
- `get_default_optimizer`, it takes as input the pytorch model returned by `get_reference_model` and returns the default optimizer for the task. 
- `get_default_criterion`, it takes no inputs and returns the default loss function for the task.
- `train_one_epoch`, implements one epoch of training and validation for the benchmark. For the validation part it directly calls the `evaluate` function. It takes as input an integer specifying the current *epoch*, the *model* to be trained, the *criterion*, the *optimizer*, the *train* and *val* dataloaders and finally the *device* to be used for the training. It returns a dictionary of tracked metrics.
- `evaluate`, implement an evaluation step of the model. This step can be both of validation or test depending on the specific dataloader provided as input. It takes as input the *model*, the *criterion*, the *dataloader* and the *device*. It returns a dictionary of tracked metrics.

Optionally, the benchmark may defines and implements the `get_default_scheduler` function which takes as input the optimizer and returns a specified learning-rate scheduler.

#### **`__init__.py`**
The body of this file **must** import all the standard functions described in `data.py`, `model.py` and `train.py`.
This file is mandatory to identify the parent directory as a python package and to expose to the user the developed functions.

To gain more insights about how this file is structurated and about how the user can develop one on its own, please consult one of the different `__init__.py` files already included in the library. E.g., [`image_classification/__init__.py`](./pytorch_benchmarks/image_classification/__init__.py).

### Example Scripts
Finally, for each benchmark an end-to-end example script is provided in the `examples` directory:
1. [Image Classification Example](examples/image_classification_example.py)
2. [Keyword Spotting Example](examples/keyword_spotting_example.py)
3. [Visual Wake Words Example](examples/visual_wake_words_example.py)
4. [Anomaly Detection Example](examples/anomaly_detection_example.py)
5. [Heart Rate Detection Example](examples/hr_detection_example.py)
6. [Tiny ImageNet Example](examples/tiny_imagenet_example.py)
7. [Gesture Recognition Example](examples/gesture_recognition_example.py)
8. [Image Classification ViT CIFAR10 Example](examples/vit_cifar10_example.py) and [Image Classification ViT Tiny-ImageNet Example](examples/vit_tinyimaget_example.py)

Each example shows how to use the different functions in order to build a neat and simple DNN training.

## Contribution guidelines
If you want to contribute to `pytorch-benchmarks` with your code, please follow this steps:
1. Create a new directory within [`./pytorch_benchmarks`](./pytorch_benchmarks/) giving a meaningful name to the task.
2. Follow the format described in [API Details](#api-details).
3. Include an end-to-end [example script](#example-scripts).
4. Update [this README](README.md) with the relevant pointers to your new task.
5. If you are not a maintainer of the repository, please **create a pull-request**.
