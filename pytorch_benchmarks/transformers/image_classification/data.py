# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Leonardo Tredese <s302294@studenti.polito.it>                     *
# *----------------------------------------------------------------------------*
from typing import Literal
import os
import warnings
import torch
import torchvision.datasets.utils as ds_utils
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

def get_data(dataset: Literal['cifar10', 'tiny-imagenet', 'imagenet'],
             validation_split: float=0.2,
             data_dir:str = None,
             rand_augment: bool = True,
             download: bool = True,
             image_size: tuple = (384, 384)):
    # impose default data directory
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), f'./{dataset}_data')
    
    # imagenet mean and std
    mean = 0.485, 0.456, 0.406
    std = 0.229, 0.224, 0.225

    normalize = transforms.Normalize(mean, std)

    train_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation = InterpolationMode.BICUBIC),
        transforms.RandAugment(num_ops=2, magnitude=9 if rand_augment else 0),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation = InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    
    # load dataset
    dataset = dataset.lower()
    if dataset == 'cifar10':
        return get_cifar10(data_dir, download, validation_split, train_transform, test_transform)
    elif dataset == 'tiny-imagenet':
        return get_tiny_imagenet(data_dir, download, validation_split, train_transform, test_transform)
    elif dataset == 'imagenet':
        if download:
            warnings.warn("""Unfortunately, ImageNet cannot be downloaded automatically,
                             please download it manually and set download=False, and 
                             specify the path to the dataset""", RuntimeWarning)
        return get_imagenet(data_dir, validation_split, train_transform, test_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}, please choose between: cifar10, tiny-imagenet, imagenet")

def get_train_val_datasets(train_set: torch.utils.data.Dataset, val_set: torch.utils.data.Dataset, validation_split: float = 0.2):
    train_size = len(train_set)
    val_size = int(train_size * validation_split)
    indices = torch.randperm(train_size)
    train_set = torch.utils.data.Subset(train_set, indices[:-val_size])
    val_set = torch.utils.data.Subset(val_set, indices[-val_size:])
    return train_set, val_set

def get_cifar10(data_dir: str, download: bool, validation_split: float, train_transform, test_transform): 
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=test_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=test_transform)
    return *get_train_val_datasets(train_dataset, val_dataset, validation_split), test_dataset

def get_tiny_imagenet(data_dir: str, download: bool, validation_split: float, train_transform, test_transform):
    URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    # check if dataset is already downloaded
    is_downloaded = os.path.exists(os.path.join(data_dir, 'tiny-imagenet-200.zip')) and os.path.exists(os.path.join(data_dir, 'tiny-imagenet-200'))
    assert download or is_downloaded, "Dataset is not downloaded or not decompressed, please set download=True or decompress the dataset manually"
    if download and not is_downloaded:
        ds_utils.download_and_extract_archive(URL, data_dir)
        assert ds_utils.check_integrity(os.path.join(data_dir, 'tiny-imagenet-200.zip'), '90528d7ca1a48142e341f4ef8d21d0de'), "Downloaded dataset does not match the expected checksum"
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'tiny-imagenet-200', 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'tiny-imagenet-200', 'train'), transform=test_transform)
    test_dir = structure_tiny_imagenet_val(os.path.join(data_dir, 'tiny-imagenet-200'))
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    return *get_train_val_datasets(train_dataset, val_dataset, validation_split), test_dataset

def structure_tiny_imagenet_val(root_dir: str, out_dir: str = 'structured_val'):
    image_class = dict()
    val_data_dir = os.path.join(root_dir, 'val')
    out_dir = os.path.join(root_dir, out_dir)
    # read couples (image_name, class_name) from val_annotations.txt
    with open(os.path.join(val_data_dir, 'val_annotations.txt'), 'r') as f:
        for line in f.readlines():
            file_name, class_name = line.split('\t')[:2]
            image_class[file_name] = class_name
    # create a directory for each class
    for class_name in set(image_class.values()):
        os.makedirs(os.path.join(out_dir, class_name), exist_ok=True)
    # move each image to its corresponding class directory
    for file_name, class_name in image_class.items():
        src = os.path.join(val_data_dir, 'images', file_name)
        if os.path.exists(src):
            dst = os.path.join(out_dir, class_name, file_name)
            os.replace(src, dst)
    return out_dir
    


def get_imagenet(data_dir: str, validation_split: float, train_transform, test_transform):
    train_dataset = datasets.ImageNet(root, split='train', transform=train_transform)
    val_dataset = datasets.ImageNet(root, split='val', transform=test_transform)
    test_dataset = datasets.ImageNet(root, split='val', transform=test_transform)
    return *get_train_val_datasets(train_dataset, val_dataset, validation_split), test_dataset

def build_dataloaders(datasets: tuple, batch_size: int = 32, num_workers: int = 2):
    train_dataset, val_dataset, test_dataset = datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
