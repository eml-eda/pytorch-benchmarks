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

import random
from typing import Tuple
import os
import glob
import numpy
import torch
import torchvision.transforms as transforms
import requests
import tarfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

DATA_URL = 'https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/'


class Augment(Dataset):
    def __init__(self, base_data, augment=True):
        super().__init__()
        self.base_data = base_data

        # Defining the same data augmentation steps of the TinyML paper for training
        if augment:
            self.augmentation = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=96, scale=(0.9, 1.1)),
                transforms.ToTensor()
            ])
        else:  # Used for test, simply convert to tensor
            self.augmentation = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        img, label = self.base_data[index]
        img = self.augmentation(img)
        return img, label

    def __len__(self):
        return len(self.base_data)


class Coco(Dataset):
    def __init__(self, image_path_train, image_path_val):
        super().__init__()
        # self.augment = augment

        # images with person and train in the filename
        self.person_list_train = glob.glob(image_path_train[0])
        self.person_label_train = list(torch.ones(len(self.person_list_train),
                                                  dtype=torch.long))

        # images with person and val in the filename
        self.person_list_val = glob.glob(image_path_val[0])
        self.person_label_val = list(torch.ones(len(self.person_list_val),
                                                dtype=torch.long))

        # images with no person and train in the file name
        self.non_person_list_train = glob.glob(image_path_train[1])
        self.non_person_label_train = list(torch.zeros(len(self.non_person_list_train),
                                                       dtype=torch.long))

        # images with no person and val in the file name
        self.non_person_list_val = glob.glob(image_path_val[1])
        self.non_person_label_val = list(torch.zeros(len(self.non_person_list_val),
                                                     dtype=torch.long))

        self.set = (self.person_list_train + self.non_person_list_train +
                    self.person_list_val + self.non_person_list_val)
        self.label = (self.person_label_train + self.non_person_label_train +
                      self.person_label_val + self.non_person_label_val)

    def __getitem__(self, index):
        # TODO: is it possible to pre-open all the images?
        img = Image.open(self.set[index])
        flag = self.label[index]
        # img = transforms.ToTensor()(img)
        return img, flag

    def __len__(self):
        return len(self.set)


def get_data(data_dir=None,
             url=DATA_URL,
             ds_name='vw_coco2014_96.tar.gz',
             val_split=0.2,
             test_split=0.1,
             seed=None
             ) -> Tuple[Dataset, ...]:
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), 'vww_data')
    filename = data_dir + '/' + ds_name
    if not os.path.exists(filename):
        ds_coco = requests.get(url + ds_name)
        os.makedirs(data_dir)
        with open(filename, 'wb') as f:
            f.write(ds_coco.content)

    ds_person_name = '/vw_coco2014_96/person/'
    ds_non_person_name = '/vw_coco2014_96/non_person/'

    if not os.path.exists(data_dir + ds_person_name) or \
       not os.path.exists(data_dir + ds_non_person_name):
        file = tarfile.open(filename)
        file.extractall(data_dir)
        file.close()

    # Merge together train and val samples in a single dataset
    # MLPerf Tiny doesn't use the original validation split but it merges all data
    # and then it takes random 10% as test set.
    ds_train_val = Coco([data_dir + ds_person_name + '*train*',
                         data_dir + ds_non_person_name + '*train*'],
                        [data_dir + ds_person_name + '*val*',
                         data_dir + ds_non_person_name + '*val*'])

    # Maybe fix seed of RNG
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    train_val_len = len(ds_train_val) - int(test_split * len(ds_train_val))
    test_len = int(len(ds_train_val) * test_split)
    ds_train_val, ds_test = random_split(ds_train_val, [train_val_len, test_len],
                                         generator=generator)

    # Add augmentation for train-val split
    ds_train_val = Augment(ds_train_val)
    ds_test = Augment(ds_test, augment=False)

    # We take another 20% as validation split
    train_len = len(ds_train_val) - int(val_split * len(ds_train_val))
    val_len = int(val_split * len(ds_train_val))
    ds_train, ds_val = random_split(ds_train_val, [train_len, val_len],
                                    generator=generator)

    return ds_train, ds_val, ds_test


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size=32,
                      num_workers=2,
                      seed=None
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = datasets

    # Maybe fix seed of RNG
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    # Maybe define worker init fn
    if seed is not None:
        def worker_init_fn(worker_id):
            numpy.random.seed(seed)
            random.seed(seed)
    else:
        worker_init_fn = None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return train_loader, val_loader, test_loader
