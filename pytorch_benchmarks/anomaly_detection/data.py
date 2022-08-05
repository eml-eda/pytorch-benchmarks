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
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, random_split


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


def _calculate_ae_accuracy(y_pred, y_true):
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        correct = np.sum(y_pred_binary == y_true)
        accuracy_tmp = 100 * correct / len(y_pred_binary)
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp
    return accuracy


def _calculate_ae_pr_accuracy(y_pred, y_true):
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    n_normal = np.sum(y_true == 0)
    precision = np.zeros(len(thresholds))
    recall = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build matrix of TP, TN, FP and FN
        false_positive = np.sum((y_pred_binary[0:n_normal] == 1))
        true_positive = np.sum((y_pred_binary[n_normal:] == 1))
        false_negative = np.sum((y_pred_binary[n_normal:] == 0))
        # Calculate and store precision and recall
        precision[threshold_item] = true_positive / (true_positive + false_positive)
        recall[threshold_item] = true_positive / (true_positive + false_negative)
        # See if the accuracy has improved
        accuracy_tmp = 100 * (precision[threshold_item] + recall[threshold_item]) / 2
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp
    return accuracy


def _calculate_ae_auc(y_pred, y_true):
    """
    Autoencoder ROC AUC calculation
    """
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.01, .01) * (np.amax(y_pred) - np.amin(y_pred))
    roc_auc = 0

    n_normal = np.sum(y_true == 0)
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(1, len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build TP and FP
        tpr[threshold_item] = np.sum((y_pred_binary[n_normal:] == 1)
                                     ) / float(len(y_true) - n_normal)
        fpr[threshold_item] = np.sum((y_pred_binary[0:n_normal] == 1)) / float(n_normal)

    # Force boundary condition
    fpr[0] = 1
    tpr[0] = 1

    # Integrate
    for threshold_item in range(len(thresholds) - 1):
        roc_auc += .5 * (tpr[threshold_item] + tpr[threshold_item + 1]) * (
            fpr[threshold_item] - fpr[threshold_item + 1])
    return roc_auc


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


def test_model(ds_test,
               model
               ):
    test_metrics = {}
    for machine in ds_test:
        y_pred = [0. for k in range(len(machine))]
        y_true = []
        machine_id = ''
        for file_idx, element in tqdm(enumerate(machine), total=len(machine), desc="preprocessing"):
            file_path, label, id = element
            machine_id = id[0]
            y_true.append(label[0].item())
            data = _file_to_vector_array(file_path[0],
                                         n_mels=128,
                                         frames=5,
                                         n_fft=1024,
                                         hop_length=512,
                                         power=2.0
                                         )
            data = data.astype('float32')
            data = torch.from_numpy(data)
            pred = model(data)
            data = data.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            errors = np.mean(np.square(data - pred), axis=1)
            y_pred[file_idx] = np.mean(errors)
        y_true = np.array(y_true, dtype='float64')
        y_pred = np.array(y_pred, dtype='float64')
        acc = _calculate_ae_accuracy(y_pred, y_true)
        pr_acc = _calculate_ae_pr_accuracy(y_pred, y_true)
        auc = _calculate_ae_auc(y_pred, y_true)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
        test_metrics[machine_id] = {
            'acc': acc,
            'pr_acc': pr_acc,
            'auc': auc,
            'p_auc': p_auc
        }
    return test_metrics


def get_data(data_dir=None,
             url1='https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1',
             url2='https://zenodo.org/record/3727685/files/eval_data_train_ToyCar.zip?download=1',
             ds_name1='dev_data_ToyCar.zip',
             ds_name2='eval_data_train_ToyCar.zip',
             val_split=0.1,
             ) -> Tuple[Dataset, Dataset, list[Dataset]]:
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), 'amd_data')
    filename1 = data_dir + '/' + ds_name1
    filename2 = data_dir + '/' + ds_name2
    if not os.path.exists(filename1) or not os.path.exists(filename2):
        os.makedirs(data_dir)
        ds_dev = requests.get(url1 + '/' + ds_name1)
        with open(filename1, 'wb') as f:
            f.write(ds_dev.content)
        ds_eval = requests.get(url2 + '/' + ds_name2)
        with open(filename2, 'wb') as f:
            f.write(ds_eval.content)

    with zipfile.ZipFile(filename1, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    with zipfile.ZipFile(filename2, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    ds_train_val = ToyCar(data_dir)
    val_len = int(val_split * len(ds_train_val))
    train_len = len(ds_train_val) - val_len
    ds_train, ds_val = random_split(ds_train_val, [train_len, val_len])

    machine_id_list = _get_machine_id_list_for_test()
    ds_test = []
    for id in machine_id_list:
        ds_test.append(ToyCarTest(data_dir, id))
    return ds_train, ds_val, ds_test


def build_dataloaders(datasets: Tuple[Dataset, Dataset, list[Dataset]],
                      batch_size=512,
                      num_workers=2
                      ) -> Tuple[DataLoader, DataLoader, list[DataLoader]]:
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
    test_loader = []
    for dataset in test_set:
        test_loader.append(DataLoader(
            dataset,
            shuffle=False,
            num_workers=num_workers))
    return train_loader, val_loader, test_loader
