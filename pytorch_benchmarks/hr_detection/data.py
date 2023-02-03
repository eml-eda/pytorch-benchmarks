from typing import Tuple
import pandas as pd
from pathlib import Path
import pickle
import random
import requests
import zipfile

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from skimage.util.shape import view_as_windows
import torch
from torch.utils.data import Dataset, DataLoader

DALIA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip"


def _collect_data(data_dir):
    random.seed(42)

    dataset = dict()
    num = list(range(1, 15+1))
    session_list = random.sample(num, len(num))
    for subj in session_list:
        with open(data_dir / 'PPG_FieldStudy' / f'S{str(subj)}' / f'S{str(subj)}.pkl', 'rb') as f:
            subject = pickle.load(f, encoding='latin1')
        ppg = subject['signal']['wrist']['BVP'][::2].astype('float32')
        acc = subject['signal']['wrist']['ACC'].astype('float32')
        target = subject['label'].astype('float32')
        dataset[subj] = {
                'ppg': ppg,
                'acc': acc,
                'target': target
                }
    return dataset


def _preprocess_data(data_dir, dataset):
    """
    Process data with a sliding window of size 'time_window' and overlap 'overlap'
    """
    fs = 32
    time_window = 8
    overlap = 2

    groups = list()
    signals = list()
    targets = list()

    for k in dataset:
        sig = np.concatenate((dataset[k]['ppg'], dataset[k]['acc']), axis=1)
        sig = np.moveaxis(
            view_as_windows(sig, (fs*time_window, 4), fs*overlap)[:, 0, :, :],
            1, 2)
        groups.append(np.full(sig.shape[0], k))
        signals.append(sig)
        targets.append(np.reshape(
            dataset[k]['target'],
            (dataset[k]['target'].shape[0], 1)))

    groups = np.hstack(groups)
    X = np.vstack(signals)
    y = np.reshape(np.vstack(targets), (-1, 1))

    dataset = {'X': X, 'y': y, 'groups': groups}
    with open(data_dir / 'slimmed_dalia.pkl', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    return X, y, groups


def _get_data_gen(samples, targets, groups, cross_val):
    n = 4
    subjects = 15
    indices, _ = _rndgroup_kfold(groups, n)
    kfold_it = 0
    while kfold_it < subjects:
        fold = kfold_it // n
        print(f'kFold-iteration: {kfold_it}')
        train_index, test_val_index = indices[fold]
        # Train Dataset
        train_samples = samples[train_index]
        train_targets = targets[train_index]
        ds_train = Dalia(train_samples, train_targets)
        # Val and Test Dataset
        logo = LeaveOneGroupOut()
        samples_val_test = samples[test_val_index]
        targets_val_test = targets[test_val_index]
        groups_val_test = groups[test_val_index]
        j = 0
        for val_index, test_index in logo.split(samples_val_test,
                                                targets_val_test,
                                                groups_val_test):

            if j == kfold_it % n:
                val_samples = samples_val_test[val_index]
                val_targets = targets_val_test[val_index]
                ds_val = Dalia(val_samples, val_targets)
                test_subj = groups[test_val_index][test_index][0]
                print(f'Test Subject: {test_subj}')
                test_samples = samples_val_test[test_index]
                test_targets = targets_val_test[test_index]
                ds_test = Dalia(test_samples, test_targets, test_subj)
            j += 1

        yield ds_train, ds_val, ds_test
        kfold_it += 1


def _rndgroup_kfold(groups, n, seed=35):
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.
    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    indices = list()
    split_dict = dict()
    i = 0
    for split in np.array_split(unique, n):
        split_dict[i] = split
        i += 1
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        indices.append((train, test))
    return indices, split_dict


class Dalia(Dataset):
    def __init__(self, samples, targets, test_subj=None):
        super(Dalia).__init__()
        self.samples = samples
        self.targets = targets
        self.test_subj = test_subj

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.samples[idx]
        target = self.targets[idx]
        return sample, target

    def __len__(self):
        return len(self.samples)


def get_data(data_dir=None,
             url=DALIA_URL,
             ds_name='ppg_dalia.zip',
             cross_val=True) -> Tuple[Dataset, ...]:
    if data_dir is None:
        data_dir = Path('.').absolute() / 'hrd_data'
    filename = data_dir / ds_name
    # Download if does not exist
    if not filename.exists():
        print('Download in progress... Please wait.')
        ds_dalia = requests.get(url)
        data_dir.mkdir()
        with open(filename, 'wb') as f:
            f.write(ds_dalia.content)
    # Unzip if needed
    if not (data_dir / 'PPG_FieldStudy').exists():
        print('Unzip files... Please wait.')
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(data_dir)

    # This step slims the dataset. This will help to speedup following usage of data
    if not (data_dir / 'slimmed_dalia.pkl').exists():
        dataset = _collect_data(data_dir)
        samples, target, groups = _preprocess_data(data_dir, dataset)
    else:
        with open(data_dir / 'slimmed_dalia.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        samples, target, groups = dataset.values()

    generator = _get_data_gen(samples, target, groups, cross_val)
    return generator


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size=128,
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
