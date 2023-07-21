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
import requests
import zipfile
import librosa
import numpy as np
import sys
import glob
from tqdm import tqdm
import torch
import itertools
import re
from torch.utils.data import Dataset, DataLoader, random_split

URL_TRAIN = 'https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1'
URL_TEST = 'https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1'


def _get_machine_id_list_for_test(target_dir=os.path.join(os.getcwd(), 'amd_data')):
    """
    target_dir : str
        base directory path of "dev_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir + '/ToyCar',
                                                                 dir_name='test', ext='wav'))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def _file_to_vector_array(file_name,
                          n_mels,
                          frames,
                          n_fft,
                          hop_length,
                          power,
                          ):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram
    y, sr = librosa.load(file_name, sr=None, mono=False)

    # 02a generate melspectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 3b take central part only
    log_mel_spectrogram = log_mel_spectrogram[:, 50:250]

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:
                                                                            t + vector_array_size].T
    return vector_array


def _list_to_vector_array(file_list,
                          n_mels,
                          frames,
                          n_fft,
                          hop_length,
                          power
                          ):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames
    for idx in tqdm(range(len(file_list)), desc="preprocessing"):
        vector_array = _file_to_vector_array(file_list[idx],
                                             n_mels=n_mels,
                                             frames=frames,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             power=power)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    return dataset


def _test_file_list_generator(target_dir,
                              dir_name,
                              id_name,
                              prefix_normal="normal",
                              prefix_anomaly="anomaly",
                              ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
    """
    normal_files = sorted(
        glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}"
                  .format(dir=target_dir,
                          dir_name=dir_name,
                          prefix_normal=prefix_normal,
                          id_name=id_name,
                          ext=ext)))
    normal_labels = np.zeros(len(normal_files))
    anomaly_files = sorted(
        glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}"
                  .format(dir=target_dir,
                          dir_name=dir_name,
                          prefix_anomaly=prefix_anomaly,
                          id_name=id_name,
                          ext=ext)))
    anomaly_labels = np.ones(len(anomaly_files))
    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return files, labels


class ToyCar(Dataset):
    def __init__(self,
                 target_dir,
                 n_mels=128,
                 frames=5,
                 n_fft=1024,
                 hop_length=512,
                 power=2.0
                 ):
        super().__init__()
        # generate training list
        training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}"
                                             .format(dir=target_dir + '/ToyCar',
                                                     dir_name='train', ext='wav'))
        self.set = sorted(glob.glob(training_list_path))
        self.wav = _list_to_vector_array(self.set,
                                         n_mels=n_mels,
                                         frames=frames,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         power=power)
        self.wav = self.wav.astype('float32')
        self.wav = torch.from_numpy(self.wav)

    def __getitem__(self, index):
        wav = self.wav[index]
        return wav

    def __len__(self):
        return len(self.wav)


class ToyCarTest(Dataset):
    def __init__(self,
                 target_dir,
                 id
                 ):
        super().__init__()
        # generate test list
        self.id = id
        self.set, self.y_true = _test_file_list_generator(target_dir=target_dir + '/ToyCar',
                                                          dir_name='test',
                                                          id_name=id)

    def __getitem__(self, index):
        wav = self.set[index]
        label = self.y_true[index]
        return wav, label, self.id

    def __len__(self):
        return len(self.set)


def get_data(data_dir=None,
             val_split=0.1,
             seed=None,
             ) -> Tuple[Dataset, Dataset, list[Dataset]]:
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), 'amd_data')
    dev_zip = os.path.join(data_dir, 'dev_data_ToyCar.zip')
    eval_zip = os.path.join(data_dir, 'eval_data_train_ToyCar.zip')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        ds_dev = requests.get(URL_TRAIN)
        with open(dev_zip, 'wb') as f:
            f.write(ds_dev.content)
        ds_eval = requests.get(URL_TEST)
        with open(eval_zip, 'wb') as f:
            f.write(ds_eval.content)
        with zipfile.ZipFile(dev_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        with zipfile.ZipFile(eval_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # Maybe fix seed of RNG
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    ds_train_val = ToyCar(data_dir)
    val_len = int(val_split * len(ds_train_val))
    train_len = len(ds_train_val) - val_len
    ds_train, ds_val = random_split(ds_train_val, [train_len, val_len],
                                    generator=generator)

    machine_id_list = _get_machine_id_list_for_test(target_dir=data_dir)
    ds_test = []
    for id in machine_id_list:
        ds_test.append(ToyCarTest(data_dir, id))
    return ds_train, ds_val, ds_test


def build_dataloaders(datasets: Tuple[Dataset, Dataset, list[Dataset]],
                      batch_size=512,
                      num_workers=2,
                      seed=None
                      ) -> Tuple[DataLoader, DataLoader, list[DataLoader]]:
    train_set, val_set, test_set = datasets

    # Maybe fix seed of RNG
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    # Maybe define worker init fn
    if seed is not None:
        def worker_init_fn(worker_id):
            np.random.seed(seed)
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
    test_loader = []
    for dataset in test_set:
        test_loader.append(DataLoader(
            dataset,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=generator,
            ))
    return train_loader, val_loader, test_loader
