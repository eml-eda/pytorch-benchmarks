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

from typing import Tuple
import os
import glob
import torch
import torchvision.transforms as transforms
import requests
import tarfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split


class Coco(Dataset):
    def __init__(self, image_path):
        super().__init__()

        # images with person and train in the filename
        self.person_list = glob.glob(image_path[0])
        self.person_label = list(torch.ones(len(self.person_list), dtype=torch.long))

        # images with no person and train in the file name
        self.non_person_list = glob.glob(image_path[1])
        self.non_person_label = list(torch.zeros(len(self.non_person_list), dtype=torch.long))

        self.set = self.person_list + self.non_person_list
        self.label = self.person_label + self.non_person_label

        # Defining the same data augmentation steps of the TinyML paper for training
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96, scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # TODO: is it possible to pre-open all the images?
        img = Image.open(self.set[index])
        flag = self.label[index]
        img = self.augmentation(img)
        return img, flag

    def __len__(self):
        return len(self.set)


def get_data(data_dir=None,
             url='https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/',
             ds_name='vw_coco2014_96.tar.gz',
             val_split=0.2,
             test_split=0.1
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

    ds_train_orig = Coco([data_dir + ds_person_name + '*train*',
                          data_dir + ds_non_person_name + '*train*'])
    ds_val_orig = Coco([data_dir + ds_person_name + '*val*',
                        data_dir + ds_non_person_name + '*val*'])

    # TinyML doesnt' use the original validation split but it merges all data
    # and then it takes random 10% as test set.
    ds_train_val = ConcatDataset([ds_train_orig, ds_val_orig])

    train_val_len = len(ds_train_val) - int(test_split * len(ds_train_val))
    test_len = int(len(ds_train_val) * test_split)
    ds_train_val, ds_test = random_split(ds_train_val, [train_val_len, test_len])

    # We take another 20% as validation split
    train_len = len(ds_train_val) - int(val_split * len(ds_train_val))
    val_len = int(val_split * len(ds_train_val))
    ds_train, ds_val = random_split(ds_train_val, [train_len, val_len])

    return ds_train, ds_val, ds_test


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size=32,
                      num_workers=2
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = datasets
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
