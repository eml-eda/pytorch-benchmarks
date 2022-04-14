import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
import sys
torch.manual_seed(23)

class Coco(torch.utils.data.Dataset):
    def __init__(self, image_path):
        super().__init__()

        # images with person and train in the filename
        self.train_person_list = glob.glob(image_path[0])
        self.train_person_label = list(np.ones(len(self.train_person_list)))

        # images with no person and train in the file name
        self.train_non_person_list = glob.glob(image_path[1])
        self.train_non_person_label = list(np.zeros(len(self.train_non_person_list)))

        # images with person and val in the filename
        self.val_person_list = glob.glob(image_path[2])
        self.val_person_label = list(np.ones(len(self.val_person_list)))

        # images with no person and val in the filename
        self.val_non_person_list = glob.glob(image_path[3])
        self.val_non_person_label = list(np.zeros(len(self.val_non_person_list)))

        self.set = self.train_person_list + self.train_non_person_list + \
                   self.val_person_list + self.val_non_person_list

        self.label = self.train_person_label + self.train_non_person_label + \
                     self.val_person_label + self.val_non_person_label

    def __getitem__(self, index):
        img = Image.open(self.set[index])
        flag = self.label[index]

        # Defining the same data augmentation steps of the TinyML paper for training
        augmentation = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96, scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])

        img = augmentation(img)
        return img, flag

    def __len__(self):
        return len(self.set)


# Function to retrieve the COCO benchmark
def get_benchmark():
    training_set = Coco(['vw_coco2014_96/person/*train*', 'vw_coco2014_96/non_person/*train*',
                         'vw_coco2014_96/person/*val*', 'vw_coco2014_96/non_person/*val*'])

    train_val_len = len(training_set) - int(len(training_set) * 0.1)
    test_len = int(len(training_set) * 0.1)
    labels = ('non person', 'person')
    train_val_set, test_set = torch.utils.data.random_split(training_set, [train_val_len, test_len])

    return train_val_set, test_set, labels

# Function to retrieve the training, validation and test dataloaders
def get_dataloaders(config, train_val_set, test_set):
    if config['val_split'] > 0:
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
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False,
                                                  num_workers=config['batch_size'])
        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(train_val_set,
                                                  batch_size=config['batch_size'],
                                                  shuffle=True,
                                                  num_workers=config['num_workers'])
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False,
                                                  num_workers=config['batch_size'])
        return train_loader, "", test_loader

# I define a class ConvBlock to simplify the definition of the network later
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

# Definition of MobileNet architecture
class MobilenetV1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # MobileNet v1 parameters
        self.input_shape = [3, 96, 96]  # default size for coco dataset
        self.num_classes = 2  # binary classification: person or non person
        self.num_filters = 8

        # MobileNet v1 layers

        # 1st layer
        self.inputblock = ConvBlock(in_channels=3, out_channels=8,  kernel_size=3, stride=2, padding=1)

        # 2nd layer
        self.depthwise2 = ConvBlock(in_channels=8, out_channels=8,  kernel_size=3, stride=1, padding=1, groups=8)
        self.pointwise2 = ConvBlock(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0)

        # 3d layer
        self.depthwise3 = ConvBlock(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, groups=16)
        self.pointwise3 = ConvBlock(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)

        # 4th layer
        self.depthwise4 = ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pointwise4 = ConvBlock(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        # 5h layer
        self.depthwise5 = ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32)
        self.pointwise5 = ConvBlock(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # 6th layer
        self.depthwise6 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64)
        self.pointwise6 = ConvBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # 7th layer
        self.depthwise7 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64)
        self.pointwise7 = ConvBlock(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)

        # 8th layer
        self.depthwise8 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise8 = ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        # 9th layer
        self.depthwise9 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise9 = ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        # 10th layer
        self.depthwise10 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise10 = ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        # 11th layer
        self.depthwise11 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise11 = ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        # 12th layer
        self.depthwise12 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise12 = ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        # 13th layer
        self.depthwise13 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128)
        self.pointwise13 = ConvBlock(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)

        # 14th layer
        self.depthwise14 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256)
        self.pointwise14 = ConvBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.avgpool = torch.nn.AvgPool2d(3)

        self.out = nn.Linear(256, 2)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, input):

        # Input tensor shape       # [96, 96,  3]

        # 1st layer
        x = self.inputblock(input) # [48, 48,  8]

        # 2nd layer
        x = self.depthwise2(x)     # [48, 48,  8]
        x = self.pointwise2(x)     # [48, 48, 16]

        # 3rd layer
        x = self.depthwise3(x)     # [24, 24, 16]
        x = self.pointwise3(x)     # [24, 24, 32]

        # 4th layer
        x = self.depthwise4(x)     # [24, 24, 32]
        x = self.pointwise4(x)     # [24, 24, 32]

        # 5th layer
        x = self.depthwise5(x)     # [12, 12, 32]
        x = self.pointwise5(x)     # [12, 12, 64]

        # 6th layer
        x = self.depthwise6(x)     # [12, 12, 64]
        x = self.pointwise6(x)     # [12, 12, 64]

        # 7th layer
        x = self.depthwise7(x)     # [ 6,  6, 64]
        x = self.pointwise7(x)     # [ 6,  6, 128]

        # 8th layer
        x = self.depthwise8(x)     # [ 6,  6, 128]
        x = self.pointwise8(x)     # [ 6,  6, 128]

        # 9th layer
        x = self.depthwise9(x)     # [ 6,  6, 128]
        x = self.pointwise9(x)     # [ 6,  6, 128]

        # 10th layer
        x = self.depthwise10(x)    # [ 6,  6, 128]
        x = self.pointwise10(x)    # [ 6,  6, 128]

        # 11th layer
        x = self.depthwise11(x)    # [ 6,  6, 128]
        x = self.pointwise11(x)    # [ 6,  6, 128]

        # 12th layer
        x = self.depthwise12(x)    # [ 6,  6, 128]
        x = self.pointwise12(x)    # [ 6,  6, 128]

        # 13th layer
        x = self.depthwise13(x)    # [ 3,  3, 128]
        x = self.pointwise13(x)    # [ 3,  3, 256]

        # 14th layer
        x = self.depthwise14(x)    # [ 3,  3, 256]
        x = self.pointwise14(x)    # [ 3,  3, 256]

        x = self.avgpool(x)        # [ 1,  1, 256]
        x = torch.squeeze(x)       # [256]
        x = self.out(x)            # [2]

        return x

def lr_schedule(optimizer, epoch):
    lrate = 0.001
    if epoch >=0 and epoch <=20:
        for opt in optimizer.param_groups:
            opt['lr'] = lrate
    elif epoch >20 and epoch <=30:
        for opt in optimizer.param_groups:
            opt['lr'] = lrate/2
            lrate = lrate/2
    else:
        for opt in optimizer.param_groups:
            opt['lr'] = lrate / 4
            lrate = lrate / 4
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


def evaluate(model, criterion, data_loader, device, neval_batches = None):
  model.eval()
  avgacc = AverageMeter('6.2f')
  avgloss = AverageMeter('2.5f')
  step = 0
  with torch.no_grad() and tqdm(total=len(data_loader), unit="batch") as tepoch:
    tepoch.set_description(f"Testing phase: ")
    for image, target in data_loader:
      step += 1
      tepoch.update(1)
      target = target.type(torch.long)
      image, target = image.to(device), target.to(device)
      output, loss = run_model(model, image, target, criterion, device)
      acc_val = accuracy(output, target, topk=(1,))
      avgacc.update(acc_val[0], image.size(0))
      avgloss.update(loss, image.size(0))
      tepoch.set_postfix({'loss': avgloss, 'acc': avgacc})
      if neval_batches is not None and step >= neval_batches:
        return avgloss, avgacc
  return avgloss, avgacc


def train_one_epoch(epoch, model, criterion, optimizer, train, val, device):
  model.train()
  avgacc = AverageMeter('6.2f')
  avgloss = AverageMeter('2.5f')
  step = 0
  with tqdm(total=len(train), unit="batch") as tepoch:
    tepoch.set_description(f"Epoch {epoch}")
    for image, target in train:
      step += 1
      tepoch.update(1)
      target = target.type(torch.long)
      image, target = image.to(device), target.to(device)
      output, loss = run_model(model, image, target, criterion, device)
      lrate = lr_schedule(optimizer, epoch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      acc_val = accuracy(output, target, topk=(1,))
      avgacc.update(acc_val[0], image.size(0))
      avgloss.update(loss, image.size(0))
      if step % 100 == 99:
        tepoch.set_postfix({'loss': avgloss, 'acc': avgacc, 'lrate': lrate})
    if len(val) > 0:
        val_loss, val_acc = evaluate(model, criterion, val, device)
        final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get(),
            'val_loss': val_loss.get(),
            'val_acc': val_acc.get()
            }
    else:
        final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get()
            }
    tepoch.set_postfix(final_metrics)
    tepoch.close()
  return final_metrics