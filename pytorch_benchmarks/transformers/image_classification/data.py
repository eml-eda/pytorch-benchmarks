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

import os
import torch
from torchvision import datasets, transforms
from transformers import ViTImageProcessor

def get_data(dataset: str,
             validation_split: float=0.2,
             preprocessor_name: str = '',
             data_dir:str = None,
             download: bool = True):
    # impose default data directory
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), f'./{dataset}_data')

    # load preprocessing pipeline
    if preprocessor_name != '':
        processor = ViTImageProcessor.from_pretrained(preprocessor_name)
    else:
        processor = ViTImageProcessor()
    
    size = processor.size
    width, height = size['width'], size['height']
    mean, std = processor.image_mean, processor.image_std
    
    normalize = transforms.Normalize(mean, std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.CenterCrop((height, width)),
        transforms.ToTensor(),
        normalize,
    ])
    
    # load dataset
    dataset = dataset.lower()
    if dataset == 'cifar10':
        train_dataset, test_dataset = get_cifar10(data_dir, download, train_transform, test_transform)
    else:
        raise ValueError(f'Unknown dataset: {dataset}, please choose between: cifar10')

    # split train and validation
    train_size = len(train_dataset)
    val_size = int(train_size * validation_split)
    train_size -= val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    return train_dataset, val_dataset, test_dataset


def get_cifar10(data_dir: str, download: bool, train_transform, test_transform):
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=test_transform)
    return train_dataset, test_dataset

def build_dataloaders(datasets: tuple, batch_size: int = 32, num_workers: int = 2):
    train_dataset, val_dataset, test_dataset = datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
