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

import copy
from pathlib import Path
# import pickle
import random
from typing import Tuple, Optional, Literal, Generator, Union

import numpy as np
import opendatasets as od
import pandas as pd
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# TODO: move this in the README.md
'''
Read carefully the steps of the link below to download the datasets.
https://github.com/JovianHQ/opendatasets/blob/master/README.md#kaggle-credentials
Briefly, follow these steps to find your API credentials:

1. Go to https://kaggle.com/me/account (sign in if required).

2. Scroll down to the "API" section and click "Create New API Token".
   This will download a file kaggle.json with the following contents:
   {"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}

3. When you run opendatsets.download,
   you will be asked to enter your username & Kaggle API,
   which you can get from the file downloaded in step 2.

Note that you need to download the kaggle.json file only once.
You can also place the kaggle.json file in the same directory as the Jupyter notebook,
and the credentials will be read automatically.
'''

LINAIGE_URL = ('https://www.kaggle.com/datasets/francescodaghero/'
               'linaige/download?datasetVersionNumber=3')


class Linaige(Dataset):
    def __init__(self, samples, targets):
        super(Linaige).__init__()
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.targets = targets

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.samples[idx]
        target = self.targets[idx]
        return sample, target

    def __len__(self):
        return len(self.samples)


def get_data(data_dir: Optional[str] = None,
             win_size: int = 1,
             confindence: Literal['easy', 'all'] = 'easy',
             remove_frame: bool = True,
             classification: bool = True,
             session_number: Optional[int] = None,
             test_split: Optional[float] = None,
             majority_win: Optional[int] = None,
             seed: Optional[int] = None,
             ) -> Union[Generator, Tuple[Dataset, Optional[Dataset], torch.Tensor]]:
    """The function that download, preprocess and build dataset.

    :param data_dir: path where to download data. If is None the cwd will be used.
    (default: None)
    :type data_dir: Optional[str]
    :param win_size: the number of windows to be considered.
    It can be from 1 (no windowing) to any integer odd number (default: 1)
    :type win_size: int
    :param confidence: control which samples will be considered.
    If 'easy' samples with high labeling confindence are considered (default: 'easy')
    :type confidence: Literal['easy', 'all']
    :param remove_frame: remove first 8 frames of test set.
    Needed for making comparison fair (default: True)
    :type remove_frame: bool
    :param classification: if true classification will be performed.
    Conversely, regression will be perfomed. (default: True)
    :type classification: bool
    :param session_number: affects what the function returns. If it is None
    a generator will be returned. Conversely, if it is an int the specific
    session data will be returned with corresponding class weights (default: None)
    :type session_number: Optional[int]
    :param test_split: the percentage of data of a specific session
    to be used as test (default: None)
    :type test_split: Optional[float]
    :param majority_win: the window size to be considered for majority ensembling.
    The test set will contain data properly shaped to support ensembling.
    If it is None no ensembling will be performed
    and a standard test set will be returned (default: None)
    :type majority_win: Optional[int]
    :param seed: an optional seed for reproducibility (default: None)
    :type seed: Optional[int]

    :return: see `session_number`
    :rtype: Union[Generator, Tuple[Dataset, Optional[Dataset], torch.Tensor]]
    """
    # Maybe download data
    if data_dir is None:
        data_dir = Path('.').absolute() / 'linaige_data'
    if not data_dir.exists():
        print('Downloading...')
        data_dir.mkdir()
        od.download(LINAIGE_URL, data_dir)

    # Read data
    data = _read_files(data_dir)

    # Return a specific session denoted by `session_number`
    if session_number is not None:
        data_and_labels = _get_session(data, win_size, confindence,
                                       classification, session_number,
                                       test_split, seed)
        x_train, y_train, x_test, y_test, class_weights = data_and_labels

        train_set = Linaige(x_train, y_train)

        if x_test is not None:
            test_set = Linaige(x_test, y_test)
        else:
            test_set = None
        return train_set, test_set, class_weights
    else:
        dataset_cv = _cross_validation(data, win_size, confindence,
                                       remove_frame, classification, majority_win)
        return dataset_cv


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size: int = 128,
                      num_workers: int = 2,
                      seed: Optional[int] = None,
                      ) -> Tuple[DataLoader, DataLoader]:
    # Extracting datasets from get_data function
    train_set, test_set, _ = datasets

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

    # Build dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=len(test_set),
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
    else:
        test_loader = None

    # TODO: wtf is this...
    '''
    if len(x_test_majority) == 0:
        majority_loader = 0
    else:
        majority_loader = DataLoader(
            majority_set,
            batch_size=len(x_test_majority),
            shuffle=False,
            num_workers=num_workers,
        )
    '''

    return train_loader, test_loader


def _create_majority(samples, label, confidence,
                     majority_win, remove_frame, remove_frame_index):
    X_win = []

    for i in range(len(samples) - majority_win + 1):
        window = samples[i:i+majority_win]
        X_win.append(window)
    X_win = np.array(X_win)

    # label
    y_win = np.array(label[(majority_win - 1):])
    conf_win = np.array(confidence[(majority_win - 1):])

    # if window size is more than 8, we can not remove any thing
    # since we need the eighth element for our first comparison
    if remove_frame and majority_win <= remove_frame_index:
        # By this index, we can reach the first element for comparison and remove previous ones
        cutting_index = remove_frame_index - majority_win + 1
        X_win = X_win[cutting_index:]
        y_win = y_win[cutting_index:]
        conf_win = conf_win[cutting_index:]
    # remove windows with hard confidence
    e_idx = np.where(conf_win == 'e')[0]
    X_win_e = X_win[e_idx]
    y_win_e = y_win[e_idx]
    return X_win_e, y_win_e


# TODO: create ad-hoc function for the majority case...
def _cross_validation(data: pd.DataFrame,
                      num_w: int,
                      confindence: Literal['easy', 'all'],
                      remove_frame: bool,
                      classification: bool,
                      majority_win: Optional[int],
                      ):
    remove_frame_index = 8
    data_majority = copy.deepcopy(data)  # TODO: Check, I added deepcopy

    # Maybe remove records with hard confidence
    if confindence == 'easy':
        data = data[data["confidence"] == 'e']

    # Maybe change the target to a binary classification problem
    if not classification:
        data.loc[:, "target"] = (data["people_number"] > 1).astype(int)
        data_majority.loc[:, "target"] = (data_majority["people_number"] > 1).astype(int)
    else:
        data.loc[:, "target"] = (data["people_number"]).astype(int)
        data_majority.loc[:, "target"] = (data_majority["people_number"]).astype(int)

    # Finding how many distinct sessions exist
    session_number = set(data.loc[:, 'session'])

    # Reshaping vector(64,) of pixels to (1, 8, 8)
    # Storing tuple of (images, labels) of each session in sessions_image_label
    # Seperating data for each specific session
    sessions_image_label = list()
    for sn in session_number:
        pixel_df = data[data["session"] == sn]
        labels = pixel_df.values[:, -1]
        images = pixel_df.values[:, 2:66].reshape(
            (pixel_df.shape[0], 1, 8, 8)).astype(np.float32)
        sessions_image_label.append((images, labels))
    sessions_image_label_majority = list()
    for sn in session_number:
        pixel_df_majority = data_majority[data_majority["session"] == sn]
        labels_majority = pixel_df_majority.values[:, -1]
        images_majority = pixel_df_majority.values[:, 2:66].reshape(
            (pixel_df_majority.shape[0], 1, 8, 8)).astype(np.float32)
        conf_majority = pixel_df_majority.values[:, -3]
        sessions_image_label_majority.append((images_majority,
                                              labels_majority,
                                              conf_majority))

    if num_w < 2:  # single frame implementation
        # Starting "cross_validation" part and removing first frames
        # 'remove_frame' can be either true or false to remove first 8 frames of test session
        # Set last index to remove first and be as test set
        # Determining the first session to be as test set (which is the last session here)
        remove_session = len(sessions_image_label) - 1
        # Loop for creating train and test sets, session 1 is always in the train set
        while remove_session >= 1:
            x_train = np.array([])
            y_train = np.array([])
            # Preparing train sets
            # Putting images and labels of all sessions together,respectively.
            # sessions_image_label[i][0] stores image session of (1,8,8) and
            # sessions_image_label[i][1] stores corresponding labels
            # At last, x_train includes all images of (1,8,8)
            # from e.g., sessions 1 to 4, and y_train includes corresponding labels
            for i in range(len(sessions_image_label)):
                if i != remove_session:
                    # To vstack an empty array with another array (n_D)
                    # we need to check the array exist and
                    # then align the dimension for concat
                    if x_train.size:
                        x_train = np.vstack([x_train, sessions_image_label[i][0]])
                    else:
                        x_train = sessions_image_label[i][0]
                    # We can't use vstack for 1_D array so 'append'
                    # is used here to avoid misalignment
                    y_train = np.append(y_train, sessions_image_label[i][1])
            if classification:
                class_weights = _get_class_weight(y_train)
            else:
                class_weights = _get_class_weight(y_train)[1]
            # Preparing test sets
            # Remove first frames if it is needed for comparison
            if remove_frame:
                x_test = sessions_image_label[remove_session][0][remove_frame_index:]
                y_test = sessions_image_label[remove_session][1][remove_frame_index:]
            else:
                x_test = sessions_image_label[remove_session][0]
                y_test = sessions_image_label[remove_session][1]

            # Majority windowing part
            if majority_win is not None:
                majority_samples = sessions_image_label_majority[remove_session][0]
                majority_labels = sessions_image_label_majority[remove_session][1]
                majority_conf = sessions_image_label_majority[remove_session][2]

                data = _create_majority(majority_samples, majority_labels,
                                        majority_conf, majority_win,
                                        remove_frame, remove_frame_index)
                x_test_majority, y_test_majority = data

                test_indices_majority = np.arange(x_test_majority.shape[0])
                # np.random.shuffle(test_indices_majority)

                x_test_majority = x_test_majority[test_indices_majority]
                y_test_majority = y_test_majority[test_indices_majority]
            else:
                x_test_majority, y_test_majority = ([], [])

            # test_indices = np.arange(x_test.shape[0])
            # np.random.shuffle(test_indices)
            # x_test = x_test[test_indices]
            # y_test = y_test[test_indices]

            # train_indices = np.arange(x_train.shape[0])
            # np.random.shuffle(train_indices)
            # x_train = x_train[train_indices]
            # y_train = y_train[train_indices]

            # pickle_dict = {'removed_session': remove_session,
            #                'x_train': x_train, 'y_train': y_train,
            #                'x_test': x_test, 'y_test': y_test}
            # with open('./pickle_file_single', 'wb') as f:
            #     pickle.dump(pickle_dict, f, pickle.HIGHEST_PROTOCOL)

            # Build dataset
            train_set = Linaige(x_train, y_train)
            if majority_win is None:
                test_set = Linaige(x_test, y_test)
            else:
                test_set = Linaige(x_test_majority, y_test_majority)

            yield train_set, test_set, class_weights
            remove_session -= 1
    else:  # windowing implementation
        # Determining the last session to be the test set
        remove_session = len(sessions_image_label) - 1
        # Loop for creating train and test sets,
        # session 1 is always in the train set by this condition
        while remove_session >= 1:
            x_train = np.array([])
            y_train = np.array([])
            # Preparing train sets
            # Putting images and lebals of all sessions together,respectively.
            # sessions_image_label[i][0] stores image session of (8, 8, 1)
            # and sessions_image_label[i][1] stores corresponding labels
            # sessions_image_label[i][2] stores corresponding conf
            # At last, x_train includes all images of (1,8,8)
            # from e.g., sessions 1 to 4, y_train
            # includes corresponding labels and conf_train includes corresponding conf
            for i in range(len(sessions_image_label)):
                if i != remove_session:
                    # To vstack an empty array with another array (n_D) we need
                    # to check the array exists and then align the dimension for concat
                    if x_train.size:
                        x_train = np.vstack([x_train, sessions_image_label[i][0]])
                    else:
                        x_train = sessions_image_label[i][0]
                    # We can't use vstack for 1_D array so append is used here to avoid misalignment
                    y_train = np.append(y_train, sessions_image_label[i][1])

            if classification:
                class_weights = _get_class_weight(y_train)
            else:
                class_weights = _get_class_weight(y_train)[1]

            x_train_w = np.array([])
            y_train_w_list = list()
            for i in range(len(x_train) - num_w + 1):
                x_combined = x_train[i, :]
                for j in range(num_w - 1):
                    # Concatenating images (1, 8, 8) to get (num_w, 8, 8)
                    x_combined = np.concatenate((x_combined, x_train[i + j + 1, :]), axis=0)

                # To convert (1, 8, 8) to (1, 1, 8, 8)
                x_combined_reshape = x_combined[np.newaxis, :]
                # To vstack an empty array with another array (n_D) we need
                # to check the array exists and
                # then align the deminasion for concat to have (x_train_w.shape[0], num_w, 8, 8)
                if x_train_w.size:
                    x_train_w = np.vstack([x_train_w, x_combined_reshape])
                else:
                    x_train_w = x_combined_reshape
                # Appending labels with size of 'windowing' as a list,
                # then converting to numpy array
                y_train_w_list.append(y_train[(i + num_w - 1)])

            # To have numpy array of labels with shape (y_train_w.shape[0])
            y_train_w = np.array(y_train_w_list)

            x_test = sessions_image_label[remove_session][0]
            y_test = sessions_image_label[remove_session][1]

            x_test_w = np.array([])
            y_test_w_list = list()

            for i in range(len(x_test) - num_w + 1):
                x_combined = x_test[i, :]
                for j in range(num_w - 1):
                    # Concatenating images (1, 8, 8) to get (num_w, 8, 8)
                    x_combined = np.concatenate((x_combined, x_test[i + j + 1, :]), axis=0)

                # To convert (1, 8, 8) to (1, 1, 8, 8)
                x_combined_reshape = x_combined[np.newaxis, :]
                # To vstack an empty array with another array (n_D) we need
                # to check the array exists and
                # then align the dimension for concat to have (x_train_w.shape[0], num_w, 8, 8)
                if x_test_w.size:
                    x_test_w = np.vstack([x_test_w, x_combined_reshape])
                else:
                    x_test_w = x_combined_reshape
                # Appending labels with size of 'windowing' as a list,
                # then converting to numpy array
                y_test_w_list.append(y_test[(i + num_w - 1)])

            # To have numpy array of labels with shape (y_test_w.shape[0])
            y_test_w = np.array(y_test_w_list)

            # Preparing test sets
            # Remove first frames if it is needed for comparison
            # if window size is more than 8,
            # we can not remove any thing since we need the eighth element for our first comparison
            if remove_frame and num_w <= remove_frame_index:
                # By this index, we can reach the first element for comparison
                # and remove previous ones
                cutting_index = remove_frame_index - num_w + 1
                x_test_w = x_test_w[cutting_index:]
                y_test_w = y_test_w[cutting_index:]

            x_test = x_test_w
            y_test = y_test_w
            x_train = x_train_w
            y_train = y_train_w

            # Build dataset
            train_set = Linaige(x_train, y_train)
            test_set = Linaige(x_test, y_test)

            yield train_set, test_set, class_weights
            remove_session -= 1


def _get_class_weight(y):
    # If ‘balanced’, class weights will be given by n_samples / (n_classes * np.bincount(y))
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights


def _get_session(data: pd.DataFrame,
                 num_w: int,
                 confindence: Literal['easy', 'all'],
                 classification: bool,
                 session_number: int,
                 test_split: Optional[float],
                 seed: Optional[int]):
    # Maybe remove records with hard confidence
    if confindence == 'easy':
        data = data[data["confidence"] == 'e']

    # Maybe change the target to a binary classification problem
    if not classification:
        data.loc[:, "target"] = (data["people_number"] > 1).astype(int)
    else:
        data.loc[:, "target"] = (data["people_number"]).astype(int)

    # Reshaping vector(64,) of pixels to (1,8,8)
    # Storing tuple of (images, labels) of each session in sessions_image_label
    # Seperating data for each specific session
    pixel_df = data[data["session"] == session_number]
    y_train_s = pixel_df.values[:, -1]  # -1 -> "target"
    # 2:66 are the values corresponding to pixels in the df
    x_train_s = pixel_df.values[:, 2:66].reshape(
        (pixel_df.shape[0], 1, 8, 8)).astype(np.float32)

    # Get class weights
    if classification:
        class_weights = _get_class_weight(y_train_s)
    else:
        class_weights = _get_class_weight(y_train_s)[1]

    if num_w < 2:  # single frame implementation
        if test_split is not None:
            # AAA: originally random_state was None...
            train_and_test = train_test_split(x_train_s, y_train_s,
                                              test_size=test_split,
                                              random_state=seed,
                                              shuffle=True,
                                              stratify=None)
            x_train, x_test, y_train, y_test = train_and_test
            return x_train, y_train, x_test, y_test, class_weights
        else:
            x_train = x_train_s
            y_train = y_train_s
            return x_train, y_train, None, None, class_weights
    else:  # windowing implementation
        x_train_w = np.array([])
        y_train_w_list = list()

        for i in range(len(x_train_s) - num_w + 1):
            x_combined = x_train_s[i, :]
            for j in range(num_w - 1):
                # Concatenate images of shape (1, 8, 8) to get shape (num_w, 8, 8)
                x_combined = np.concatenate((x_combined, x_train_s[i + j + 1, :]), axis=0)

            # Convert (num_w, 8, 8) to (1, num_w, 8, 8)
            x_combined_reshape = x_combined[np.newaxis, :]
            # vstack an empty array with another array (n_D:
            # need to check the array existence
            # then align the dimension for concat to have (x_train_w.shape[0], num_w, 8, 8)
            if x_train_w.size:
                x_train_w = np.vstack([x_train_w, x_combined_reshape])
            else:
                x_train_w = x_combined_reshape
            # Appending labels with size of 'num_w' as a list,
            # then converting to numpy array
            y_train_w_list.append(y_train_s[(i + num_w - 1)])

        # Build numpy array of labels with shape (y_train_w.shape[0])
        y_train_w = np.array(y_train_w_list)
        if test_split is not None:
            # NB: originally random_state was None...
            train_and_test = train_test_split(x_train_w, y_train_w,
                                              test_size=test_split,
                                              random_state=seed,
                                              shuffle=True,
                                              stratify=None)
            x_train, x_test, y_train, y_test = train_and_test
            return x_train, y_train, x_test, y_test, class_weights
        else:
            x_train = x_train_w
            y_train = y_train_w
            return x_train, y_train, None, None, class_weights


def _read_files(data_dir):
    data = list()
    # Data are in a subdirectory called 'linaige'
    subdir = list(data_dir.glob('*'))[0]
    for file_path in sorted(subdir.glob('*')):
        session_name = file_path.stem.split("_")[0]
        session_id = int(session_name.replace("Session", ""))
        # Read the file
        file_data = pd.read_csv(file_path)
        file_data["session"] = session_id
        data.append(file_data)
    data = pd.concat(data)
    return data
