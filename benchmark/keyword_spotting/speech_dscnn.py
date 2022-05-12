#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import sys

def prepare_model_settings(label_count, args):
  """Calculates common settings needed for all models.
  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.
  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(args.sample_rate * args.clip_duration_ms / 1000)
  if args.feature_type == 'td_samples':
    window_size_samples = 1
    spectrogram_length = desired_samples
    dct_coefficient_count = 1
    window_stride_samples = 1
    fingerprint_size = desired_samples
  else:
    dct_coefficient_count = args.dct_coefficient_count
    window_size_samples = int(args.sample_rate * args.window_size_ms / 1000)
    window_stride_samples = int(args.sample_rate * args.window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
      spectrogram_length = 0
    else:
      spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
      fingerprint_size = args.dct_coefficient_count * spectrogram_length
  return {
    'desired_samples': desired_samples,
    'window_size_samples': window_size_samples,
    'window_stride_samples': window_stride_samples,
    'feature_type': args.feature_type, 
    'spectrogram_length': spectrogram_length,
    'dct_coefficient_count': dct_coefficient_count,
    'fingerprint_size': fingerprint_size,
    'label_count': label_count,
    'sample_rate': args.sample_rate,
    'background_frequency': 0.8, # args.background_frequency
    'background_volume_range_': 0.1
  }

# Definition of DSCnn architecture
class DSCnn(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args.model_architecture

        label_count = 12
        model_settings = prepare_model_settings(label_count, args)

        self.input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
        self.num_filters = 64
        self.final_pool_size = (int(self.input_shape[0] / 2), int(self.input_shape[1] / 2))

        # Model layers

        # Input pure conv2d
        self.inputlayer = nn.Conv2d(in_channels=1, out_channels=64,  kernel_size=(10,4), stride=(2,2), padding=(5,1))
        self.bn = nn.BatchNorm2d(64, momentum=0.99)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        # First layer of separable depthwise conv2d
        # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
        self.depthwise1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu11 = nn.ReLU()
        self.conv1= nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn12 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu12 = nn.ReLU()

        # Second layer of separable depthwise conv2d
        self.depthwise2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu21 = nn.ReLU()
        self.conv2= nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn22 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu22 = nn.ReLU()

        # Third layer of separable depthwise conv2d
        self.depthwise3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn31 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu31 = nn.ReLU()
        self.conv3= nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn32 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu32 = nn.ReLU()

        # Fourth layer of separable depthwise conv2d
        self.depthwise4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn41 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu41 = nn.ReLU()
        self.conv4= nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn42 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu42 = nn.ReLU()

        self.dropout2 = nn.Dropout(p=0.4)
        self.avgpool = torch.nn.AvgPool2d((25,5))
        self.out = nn.Linear(64, 12)


    def forward(self, input):

        # Input pure conv2d
        x = self.inputlayer(input)
        x = self.dropout1(self.relu(self.bn(x)))

        # First layer of separable depthwise conv2d
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.relu11(self.bn11(x))
        x = self.conv1(x)
        x = self.relu12(self.bn12(x))

        # Second layer of separable depthwise conv2d
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.relu21(self.bn21(x))
        x = self.conv2(x)
        x = self.relu22(self.bn22(x))

        # Third layer of separable depthwise conv2d
        x = self.depthwise3(x)
        x = self.pointwise3(x)
        x = self.relu31(self.bn31(x))
        x = self.conv3(x)
        x = self.relu32(self.bn32(x))

        # Fourth layer of separable depthwise conv2d
        x = self.depthwise4(x)
        x = self.pointwise4(x)
        x = self.relu41(self.bn41(x))
        x = self.conv4(x)
        x = self.relu42(self.bn42(x))

        x = self.dropout2(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.out(x)

        return x


def lr_schedule(optimizer, epoch):
    if epoch >=0 and epoch <=12:
        for opt in optimizer.param_groups:
            opt['lr'] = 0.0005
            lrate = 0.0005
    elif epoch >12 and epoch <=24:
        for opt in optimizer.param_groups:
            opt['lr'] = 0.0001
            lrate = 0.0001
    else:
        for opt in optimizer.param_groups:
            opt['lr'] = 2e-05
            lrate = 2e-05
    return lrate


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


def evaluate(model, criterion, data_loader, device):
  model.eval()
  avgacc = AverageMeter('6.2f')
  avgloss = AverageMeter('2.5f')
  step = 0
  for audio, target in data_loader:
      step += 1
      audio, target = audio.to(device), target.to(device)
      output, loss = run_model(model, audio, target, criterion, device)
      acc_val = accuracy(output, target, topk=(1,))
      avgacc.update(acc_val[0], audio.size(0))
      avgloss.update(loss, audio.size(0))
  return avgloss, avgacc


def train_one_epoch(epoch, model, criterion, optimizer, train, val, device):
  model.train()
  avgacc = AverageMeter('6.2f')
  avgloss = AverageMeter('2.5f')
  step = 0
  with tqdm(total=len(train), unit="batch") as tepoch:
    tepoch.set_description(f"Epoch {epoch+1}")
    for audio, target in train:
      step += 1
      tepoch.update(1)
      audio, target = audio.to(device), target.to(device)
      output, loss = run_model(model, audio, target, criterion, device)
      lrate = lr_schedule(optimizer, epoch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      acc_val = accuracy(output, target, topk=(1,))
      avgacc.update(acc_val[0], audio.size(0))
      avgloss.update(loss, audio.size(0))
      if step % 100 == 99:
        tepoch.set_postfix({'loss': avgloss, 'acc': avgacc, 'lrate': lrate})
    val_loss, val_acc = evaluate(model, criterion, val, device)
    final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get(),
            'val_loss': val_loss.get(),
            'val_acc': val_acc.get()
            }
    tepoch.set_postfix(final_metrics)
    tepoch.close()
  return final_metrics
