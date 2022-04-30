########################################################################
# import python-library
########################################################################
# default
import csv
import glob
import argparse
import itertools
import re
import sys
import os

# additional
import numpy
import librosa
import librosa.core
import librosa.feature
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # No pictures displayed
import pylab
import librosa.display

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import numpy as np
import random
import common as com
import eval_functions_eembc
from sklearn import metrics

#torch.manual_seed(46)

########################################################################

def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files
########################################################################

class ToyCar(torch.utils.data.Dataset):
    def __init__(self, target_dir, config):
        super().__init__()
        self.set = file_list_generator(target_dir)
        self.wav = list_to_vector_array(self.set,
                                   msg="generate train_dataset",
                                   n_mels=config["n_mels"],
                                   frames=config["frames"],
                                   n_fft=config["n_fft"],
                                   hop_length=config["hop_length"],
                                   power=config["power"])

    def __getitem__(self, index):
        wav = self.wav[index].astype('float32')
        wav = torch.from_numpy(wav)
        return wav, wav

    def __len__(self):
        return len(self.wav)


# Function to retrieve the ToyCar benchmark
def get_benchmark(target_dir, config):
    train_val_set = ToyCar(target_dir, config)
    return train_val_set

# Function to retrieve the training, validation and test dataloaders
def get_dataloaders(config, train_val_set):

        val_len = int(config['val_split'] * len(train_val_set))
        train_len = len(train_val_set) - val_len
        train_set, val_set = torch.utils.data.random_split(train_val_set, [train_len, val_len])

        train_loader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=config['batch_size'],
                                                  shuffle=True,
                                                  num_workers=config['num_workers'])
        val_loader = torch.utils.data.DataLoader(val_set,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                num_workers=config['num_workers'])

        return train_loader, val_loader

# I define a class ConvBlock to simplify the definition of the network later
class NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.dense(input)
        return self.relu(self.bn(x))

# Definition of MobileNet architecture
class AutoEncoder(torch.nn.Module):
    def __init__(self, inputDims):
        super().__init__()
        self.inputDims = inputDims

        # AutoEncoder layers

        # 1st layer
        self.layer1 = NetBlock(in_channels=self.inputDims, out_channels=128)

        # 2nd layer
        self.layer2 = NetBlock(in_channels=128, out_channels=128)

        # 3d layer
        self.layer3 = NetBlock(in_channels=128, out_channels=128)

        # 4th layer
        self.layer4 = NetBlock(in_channels=128, out_channels=128)

        # 5h layer
        self.layer5 = NetBlock(in_channels=128, out_channels=8)

        # 6th layer
        self.layer6 = NetBlock(in_channels=8, out_channels=128)

        # 7th layer
        self.layer7 = NetBlock(in_channels=128, out_channels=128)

        # 8th layer
        self.layer8 = NetBlock(in_channels=128, out_channels=128)

        # 9th layer
        self.layer9 = NetBlock(in_channels=128, out_channels=128)

        # 10th layer
        self.out = nn.Linear(128, self.inputDims)

    def forward(self, input):

        # Input tensor shape   # [input dims]

        # 1st layer
        x = self.layer1(input) # [128]

        # 2nd layer
        x = self.layer2(x)     # [128]

        # 3rd layer
        x = self.layer3(x)     # [128]

        # 4th layer
        x = self.layer4(x)     # [128]

        # 5th layer
        x = self.layer5(x)     # [8]

        # 6th layer
        x = self.layer6(x)     # [128]

        # 7th layer
        x = self.layer7(x)     # [128]

        # 8th layer
        x = self.layer8(x)     # [128]

        # 9th layer
        x = self.layer9(x)     # [128]

        # 10th layer
        x = self.out(x)        # [input dims]

        return x

class AverageMeter(object):

    """Computes and stores the average and current value"""
    def __init__(self, fmt='f', name='meter'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
         self.val = val
         self.sum += val * n
         self.count += n
         self.avg = self.sum / self.count

    def get(self):
         return float(self.avg)

    def __str__(self):
         fmtstr = '{:' + self.fmt + '}'
         return fmtstr.format(float(self.avg))


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res


class CheckPoint():
    """
    save/load a checkpoint based on a metric
    """
    def __init__(self, dir, net, optimizer, mode='min', fmt='ck_{epoch:03d}.pt'):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported")
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir = dir
        self.mode = mode
        self.format = fmt
        self.chkname = fmt
        self.net = net
        self.optimizer = optimizer
        self.val = None
        self.epoch = None
        self.best_path = None

    def __call__(self, epoch, val):
        val = float(val)
        if self.val == None:
            self.update_and_save(epoch, val)
        elif self.mode == 'min' and val < self.val:
            self.update_and_save(epoch, val)
        elif self.mode == 'max' and val > self.val:
            self.update_and_save(epoch, val)

    def update_and_save(self, epoch, val):
        self.epoch = epoch
        self.val = val
        self.update_best_path()
        self.save()

    def update_best_path(self):
        self.best_path = os.path.join(self.dir, self.format.format(**self.__dict__))

    def save(self, path=None):
        if path is None:
            path = self.best_path
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val': self.val,
        }, path)

    def load_best(self):
        for filename in os.listdir(self.dir):
          f = os.path.join(self.dir, filename)
          accuracy = 0
          if os.path.isfile(f):           
            chkpt = torch.load(f)
            chkname = "ck_{:03d}.pt".format(chkpt['epoch'])
            if chkpt['val'] > accuracy:
              self.best_path = os.path.join(self.dir, chkname)
        if self.best_path is None:
          raise FileNotFoundError("Checkpoint folder is empty")    
        self.load(self.best_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def run_model(model, image, target, criterion, device):
    output = model(image)
    loss = criterion(output, target)
    return output, loss


def evaluate(model, criterion, data_loader, device, neval_batches = None):
  model.eval()
  avgloss = AverageMeter('2.5f')
  step = 0
  with torch.no_grad():
    for input, target in data_loader:
      step += 1
      target = target.type(torch.long)
      input, target = input.to(device), target.to(device)
      output, loss = run_model(model, input, target, criterion, device)
      avgloss.update(loss, input.size(0))
      if neval_batches is not None and step >= neval_batches:
        return avgloss
  return avgloss


def train_one_epoch(epoch, model, criterion, optimizer, train, val, device):
  model.train()
  avgloss = AverageMeter('2.5f')
  step = 0
  with tqdm(total=len(train), unit="batch") as tepoch:
    tepoch.set_description(f"Epoch {epoch+1}")
    for input, target in train:
      step += 1
      tepoch.update(1)
      input, target = input.to(device), target.to(device)
      output, loss = run_model(model, input, target, criterion, device)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avgloss.update(loss, input.size(0))
      if step % 100 == 99:
          tepoch.set_postfix({'loss': avgloss})
    val_loss = evaluate(model, criterion, val, device)
    final_metrics = {
            'loss': avgloss.get(),
            'val_loss': val_loss.get(),
            }
    tepoch.set_postfix(final_metrics)
    tepoch.close()
  return final_metrics