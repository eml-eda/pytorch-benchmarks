#  Majority of this code was taken from:
# https://github.com/KinWaiCheuk/AudioLoader/blob/master/AudioLoader/speech/speechcommands.py
#
import os
from pathlib import Path
import pickle
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.hub import download_url_to_file
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
# from torchaudio.datasets.utils import download_url,
from torchaudio.datasets.utils import extract_archive
from tqdm import tqdm

SAMPLE_RATE = 16000
FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
_CHECKSUMS = {
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz":
    "3cd23799cb2bbdec517f1cc028f8d43c",
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz":
    "6b74f3901214cb2c2934e98196829835",
}

UNKNOWN = [
    'backward',
    'bed',
    'bird',
    'cat',
    'dog',
    'eight',
    'five',
    'follow',
    'forward',
    'four',
    'happy',
    'house',
    'learn',
    'marvin',
    'nine',
    'one',
    'seven',
    'sheila',
    'six',
    'three',
    'tree',
    'two',
    'visual',
    'wow',
    'zero'
]

NAME2IDX = {
    'down': 0,
    'go': 1,
    'left': 2,
    'no': 3,
    'off': 4,
    'on': 5,
    'right': 6,
    'stop': 7,
    'up': 8,
    'yes': 9,
    '_silence_': 10,
    '_unknown_': 11
}


def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


def load_speechcommands_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    # Besides the officially supported split method for datasets defined by "validation_list.txt"
    # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
    # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
    # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
    # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt  # noqa
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)  # type: ignore
    return waveform, sample_rate, label, speaker_id, utterance_number


def _caching_data(_walker, path, subset):
    cache = []
    for filepath in tqdm(_walker, desc=f'Loading {subset} set'):
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
        if label in UNKNOWN:  # if the label is not one of the 10 commands, map them to unknown
            label = '_unknown_'

        speaker, _ = os.path.splitext(filename)
        speaker, _ = os.path.splitext(speaker)

        # When loading test_set, there is a folder for _silence_
        if label == '_silence_':
            speaker_id = speaker.split(HASH_DIVIDER)
            utterance_number = -1
        else:
            speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
            utterance_number = int(utterance_number)

        # Load audio
        audio_samples, rate = torchaudio.load(filepath, normalize=False)  # type: ignore
        # audio_sample (1, len)

        if audio_samples.shape[1] != SAMPLE_RATE:
            pad_length = SAMPLE_RATE - audio_samples.shape[1]
            # pad the end of the audio until 1 second
            audio_samples = F.pad(audio_samples, (0, pad_length))  # (1, 16000)
        cache.append((audio_samples, rate, NAME2IDX[label], speaker_id, utterance_number))

    # include silence
    if subset == 'training':
        silence_clips = [
            'dude_miaowing.wav',
            'white_noise.wav',
            'exercise_bike.wav',
            'doing_the_dishes.wav',
            'pink_noise.wav'
        ]
    elif subset == 'validation':
        silence_clips = [
            'running_tap.wav'
        ]
    else:
        silence_clips = []

    for i in silence_clips:
        audio_samples, rate = torchaudio.load(  # type: ignore
            os.path.join(path, '_background_noise_', i), normalize=False)
        for start in range(0,
                           audio_samples.shape[1] - SAMPLE_RATE,
                           SAMPLE_RATE // 2):
            audio_segment = audio_samples[0, start:start + SAMPLE_RATE]
            cache.append((audio_segment.unsqueeze(0), rate, NAME2IDX['_silence_'], '00000000', -1))

    return cache


def _extract_features(raw_data, one_dim):
    desired_samples = int(SAMPLE_RATE)
    audio_list = list()
    label_list = list()

    with tqdm(total=len(raw_data), unit="samples") as t:
        t.set_description("Feature Extraction: ")
        for audio, _, label, _, _ in raw_data:
            t.update(1)
            label_list.append(label)
            audio = audio.numpy().squeeze()
            audio = (audio * 1.0) / (2 ** 15 * 1.0)
            audio = torch.tensor(audio, dtype=torch.float32)
            length = len(audio)
            audio = np.pad(audio, (0, desired_samples - length), 'constant', constant_values=0)

            time_shift_padding_placeholder_ = (2, 2)
            time_shift_offset_placeholder_ = 2

            # import pdb; pdb.set_trace()
            audio = np.pad(audio, time_shift_padding_placeholder_, 'constant',
                           constant_values=0)
            audio = audio[time_shift_offset_placeholder_:
                          desired_samples + time_shift_offset_placeholder_]
            audio = torch.tensor(audio)
            audio = torch.squeeze(audio)

            n_mfcc = 10

            melkwargs = {
                "n_fft": 512,
                "n_mels": 40,
                "win_length": 480,
                "hop_length": 320,
                "f_min": 20,
                "f_max": 4000,
                "center": False,
                "norm": None,
            }
            mfcc_torch = torchaudio.transforms.MFCC(
                sample_rate=SAMPLE_RATE,
                n_mfcc=n_mfcc,
                dct_type=2,
                norm='ortho',
                log_mels=True,
                melkwargs=melkwargs)
            mfcc_torch_log = mfcc_torch(audio)
            if not one_dim:
                mfcc_torch_log = np.moveaxis(mfcc_torch_log.numpy(), 0, -1)
                mfcc_torch_log = np.expand_dims(mfcc_torch_log, axis=0)
                mfcc_torch_log = torch.from_numpy(mfcc_torch_log)
            audio_list.append(mfcc_torch_log)

    return audio_list, label_list


class SpeechCommands(Dataset):
    def __init__(self,
                 root,
                 one_dim,
                 folder_in_archive='SpeechCommands',
                 download=True,
                 subset='training',
                 ):

        assert subset in ['training', 'validation', 'testing']

        if subset == 'testing':
            url = "speech_commands_test_set_v0.02"
        else:
            url = "speech_commands_v0.02"

        base_url = "http://download.tensorflow.org/data/"
        ext_archive = ".tar.gz"

        url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(url)
        print(f"{basename=}")
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    # download_url(url, root, hash_value=checksum, hash_type="md5")
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive, self._path)

        output_file = root + f'/{subset}.pkl' if not one_dim else root + f'/{subset}_onedim.pkl'
        subset_exist = os.path.isfile(output_file)

        if not subset_exist:
            if subset == "validation":
                self._walker = _load_list(self._path, "validation_list.txt")
                # self._raw_data = _caching_data(self._walker, self._path, subset)
            elif subset == "testing":
                self._walker = list(Path(self._path).glob('*/*.wav'))
                # self._raw_data = _caching_data(self._walker, self._path, subset)
            elif subset == "training":
                excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
                walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
                self._walker = [
                    w for w in walker
                    if (HASH_DIVIDER in w)
                    and (EXCEPT_FOLDER not in w)
                    and (os.path.normpath(w) not in excludes)
                ]
            _raw_data = _caching_data(self._walker, self._path, subset)
            self._data = _extract_features(_raw_data, one_dim)
            with open(output_file, 'wb') as handle:
                pickle.dump(self._data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(f'{output_file} already exist.')
            print('Skipping Preprocessing and Feature Extraction step.', end='\n')
            with open(output_file, 'rb') as handle:
                self._data = pickle.load(handle)

    def __getitem__(self, idx):
        return self._data[0][idx], self._data[1][idx]

    def __len__(self):
        return len(self._data[0])


def get_data(data_dir=None, one_dim=False
             ) -> Tuple[Dataset, ...]:
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), 'kws_data')
    ds_train = SpeechCommands(data_dir, one_dim, subset='training')
    ds_val = SpeechCommands(data_dir, one_dim, subset='validation')
    ds_test = SpeechCommands(data_dir, one_dim, subset='testing')

    return ds_train, ds_val, ds_test


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size=100,
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
