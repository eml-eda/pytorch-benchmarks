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
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

# Code insipired by: https://tinyurl.com/tiny-imagenet
from pathlib import Path
import shutil
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms.functional import InterpolationMode

URL_DATA = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


def get_data(data_dir=None,
             val_split=0.1,
             inp_res=64,
             ) -> Tuple[Dataset, ...]:
    if data_dir is None:
        data_dir = Path.cwd() / 'data'
    else:
        data_dir = Path(data_dir)
    if not data_dir.exists():  # Check existence
        data_dir.mkdir(parents=True)
    # TODO: Use checksum to verify if we need to download data
    if next(data_dir.iterdir(), None) is None:  # Check if empty
        # Download data
        download_and_extract_archive(URL_DATA, str(data_dir))
        (data_dir / 'tiny-imagenet-200.zip').unlink()

    train_data_dir = str(data_dir / 'tiny-imagenet-200' / 'train')
    # Train data are already subdivided in (image, label) format
    if inp_res == 64:
        transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    elif inp_res == 224:
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError('Use either 64 or 224 as inp_res.')
    ds_train_val = ImageFolder(train_data_dir, transform=transform_train)

    # Split train_val data in training and validation
    if val_split != 0.0:
        val_len = int(val_split * len(ds_train_val))
        train_len = len(ds_train_val) - val_len
        ds_train, ds_val = random_split(ds_train_val, [train_len, val_len])
    else:
        ds_train, ds_val = ds_train_val, None

    # Validation data are here used as Test data
    test_data_dir = data_dir / 'tiny-imagenet-200' / 'val'
    if (test_data_dir / 'val_annotations.txt').exists():
        # Validation data folder need to be organized in labels
        # First two columns of file 'tiny-imagenet-200/val/val_annotations.txt'
        # contains img filename and label
        with open(test_data_dir / 'val_annotations.txt', 'r') as f:
            test_image_dict = dict()
            for line in f:
                words = line.split('\t')
                test_image_dict[words[0]] = words[1]
        for image, folder in test_image_dict.items():
            newpath = test_data_dir / folder
            newpath.mkdir(exist_ok=True)
            shutil.move(
                str(test_data_dir / 'images' / image),
                newpath)
        shutil.rmtree(test_data_dir / 'images')
        (test_data_dir / 'val_annotations.txt').unlink()

    if inp_res == 64:
        transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
    elif inp_res == 224:
        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError('Use either 64 or 224 as inp_res.')
    ds_test = ImageFolder(str(test_data_dir), transform=transform_test)

    return ds_train, ds_val, ds_test


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size=100,
                      num_workers=4
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = datasets
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
    else:
        val_loader = None
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
