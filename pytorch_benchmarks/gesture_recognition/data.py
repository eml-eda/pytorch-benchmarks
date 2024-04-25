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
# * Author:  Alessio Burrello <alessio.burrello@polito.it>                     *
# *----------------------------------------------------------------------------*

from typing import Tuple
from pathlib import Path
import random
import os
import requests
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from collections.abc import Sequence


class SuperSet(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lens = np.cumsum([0] + list(map(len, self.datasets)))

    def __getitem__(self, idx):
        dataset_idx = int(np.argwhere(self.lens > idx)[0]) - 1
        idx = idx - self.lens[dataset_idx]
        return self.datasets[dataset_idx][idx]

    def __len__(self):
        return self.lens[-1]


def windowing(X_instants, R_instants, Y_instants,
              v_Hz=2000, window_time_s=.150, relative_overlap=.7,
              steady=True, steady_margin_s=1.5):
    """
        steady=True, steady_margin_s=0 -> finestre dove sample tutti
        della stessa label (o solo movimento o solo rest)
        steady=True, steady_margin_s=1.5 -> finestre dove tagli i primi
        e ultimi 1.5s di movimento
        steady=False -> tutte le finestre, anche quelle accavallate
        tra movimento e rest
    """
    # Centro della finestra (numero di campioni)
    r = int((v_Hz * window_time_s) / 2)
    N = 2 * r
    # Campioni fuori finestra da guardare per capire se steady
    margin_samples = round(v_Hz * steady_margin_s)

    overlap_pixels = round(v_Hz * relative_overlap * window_time_s)
    slide = (N - overlap_pixels)
    M_instants, C = X_instants.shape
    # M = Numero di finestre
    M = (M_instants - N) // slide + 1 * int(((M_instants - N) % slide) != 0)

    # La label dovrebbe essere quello indicato nell'ultimo istante
    Y_windows = Y_instants[r:M_instants - r:slide]
    # La repetition è quello che viene indicato a metà della finestra
    R_windows = R_instants[r:M_instants - r:slide]
    X_windows = np.zeros((M, N, C))
    is_steady_windows = np.zeros(M, dtype=bool)
    for m in range(M):
        c = r + m * slide  # c is python-style
        X_windows[m, :, :] = X_instants[c - r:c + r, :]
        if Y_instants[c] == 0:  # rest position is not margined
            is_steady_windows[m] = len(set(Y_instants[c - r: c + r])) == 1
        else:
            is_steady_windows[m] = len(set(
                Y_instants[
                    max(0, c - r - margin_samples):min(c + r + margin_samples, len(Y_instants))
                    ])) == 1
    if steady:
        return (X_windows[is_steady_windows], R_windows[is_steady_windows],
                Y_windows[is_steady_windows])
    return X_windows, R_windows, Y_windows


def read_session(filename):
    annots = loadmat(filename)

    X = annots['emg'][:, np.r_[0:8, 10:16]]
    R = annots['rerepetition'].squeeze()
    y = annots['restimulus'].squeeze()

    # Fix class numbering (id -> index)
    y[y >= 3] -= 1
    y[y >= (6 - 1)] -= 1
    y[y >= (8 - 2)] -= 1
    y[y >= (9 - 3)] -= 1

    return X, R, y


class DB6Session(Dataset):
    def __init__(self, filename):
        self.X, self.R, self.Y = read_session(filename)
        self.X_min, self.X_max = self.X.min(axis=0), self.X.max(axis=0)

    def minmax(self, minmax=None):
        if isinstance(minmax, Sequence) and minmax[0] is not None:
            X_min, X_max = minmax
        else:
            X_min, X_max = self.X_min, self.X_max

        X_std = (self.X - X_min) / (X_max - X_min)
        X_scaled = X_std * 2 - 1
        self.X = X_scaled
        return self

    def windowing(self, steady=True, n_classes='7+1', image_like_shape=False, **kwargs):
        if str(n_classes) not in {'7+1', '7'}:
            raise ValueError('Wrong n_classes')

        X_windows, R_windows, Y_windows = windowing(self.X, self.R, self.Y, steady=steady, **kwargs)

        if n_classes == '7':
            # Filtra via finestre di non movimento
            mask = Y_windows != 0
            X_windows, R_windows, Y_windows = X_windows[mask], R_windows[mask], Y_windows[mask]
            # Rimappa label da 1-7 a 0-6
            Y_windows -= 1

        self.X = torch.tensor(X_windows, dtype=torch.float32).permute(0, 2, 1)
        if image_like_shape:
            self.X = self.X.unsqueeze(dim=2)
        self.Y = torch.tensor(Y_windows, dtype=torch.long)
        self.R = R_windows

        return self

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.Y.shape[0]


class DB6MultiSession(SuperSet):

    def __init__(self, subjects, sessions, folder='.', minmax=False, **kwargs):
        self.sessions = [
            DB6Session(os.path.join(folder, f'S{subject}_D{(i // 2) + 1}_T{(i % 2) + 1}.mat'))
            for i in sessions for subject in subjects]
        self.patients = [subject for i in sessions for subject in subjects]

        if minmax:  # Apply global minmax
            self.X_min = np.vstack([session.X_min for session in self.sessions]).min(axis=0)
            self.X_max = np.vstack([session.X_max for session in self.sessions]).max(axis=0)
            for session in self.sessions:
                session.minmax(minmax=(self.X_min, self.X_max))
        elif minmax is not False:
            self.X_min, self.X_max = minmax
            for session in self.sessions:
                session.minmax(minmax=minmax)

        for session in self.sessions:
            session.windowing(**kwargs)

        # After windowing, each session-dataset length changes, so initialize SuperSet handling here
        super().__init__(*self.sessions)

    def to(self, device):
        for session in self.sessions:
            session.to(device)
        return self


ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36")
headers = {'User-agent': ua}


def download_file(subject, part, download_dir, keep_zip):
    filename = f'DB6_s{subject}_{part}.zip'
    url = f'http://ninapro.hevs.ch/system/files/DB6_Preproc/{filename}'
    download_path = os.path.join(download_dir, filename)

    os.makedirs(download_dir, exist_ok=True)

    # https://stackoverflow.com/a/1094933
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    if not os.path.isfile(download_path):
        with requests.get(url, headers=headers,  stream=True) as r:
            r.raise_for_status()

            total_filesize_GiB = int(r.headers['Content-Length']) / (2 ** 30)
            print(f'File size: {total_filesize_GiB:.1f} GiB', flush=True)

            with open(download_path + '.part', 'wb') as f:
                tot_bytes_downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    tot_bytes_downloaded += len(chunk)
                    print("Downloaded " + sizeof_fmt(tot_bytes_downloaded) +
                          "" * 20 + "\r", end="", flush=True)
        os.rename(download_path + '.part', download_path)
        print("Downloaded in", download_path, flush=True)

    with ZipFile(download_path, 'r') as zipFile:
        sessions = [info for info in zipFile.infolist()
                    if os.path.splitext(info.filename)[1] == '.mat']
        for session in sessions:
            session.filename = os.path.basename(session.filename)
            print("Extracting", session.filename)
            zipFile.extract(session, path=download_dir)

    if keep_zip == 'no':
        os.remove(download_path)


def get_data(data_dir=None,
             subjects=1) -> Tuple[Dataset, ...]:
    if data_dir is None:
        data_dir = Path('.').absolute() / 'db6_data'
    data = data_dir / "S10_D5_T2.mat"
    if not data.exists():
        print('Download in progress... Please wait.')
        for subject in np.arange(1, 11):
            for part in ['a', 'b']:
                download_file(subject, part, download_dir=data_dir, keep_zip='no')
    else:
        print('Dataset already in {} directory'.format(data_dir))
    train_sessions = [0, 1, 2, 3, 4]
    test_sessions = [5, 6, 7, 8, 9]
    ds = DB6MultiSession(folder=data_dir, subjects=subjects,
                         sessions=train_sessions, steady=True,
                         n_classes='7+1', minmax=True, image_like_shape=True)
    test_ds = DB6MultiSession(folder=data_dir, subjects=subjects,
                              sessions=test_sessions, steady=True,
                              n_classes='7+1',  minmax=(ds.X_min, ds.X_max), image_like_shape=True)
    train_ds, val_ds = ds, [ds[0]]
    return train_ds, val_ds, test_ds


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size=64,
                      num_workers=4,
                      seed=None,
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
            np.random.seed(seed)
            random.seed(seed)
    else:
        worker_init_fn = None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1024,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1024,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return train_loader, val_loader, test_loader
