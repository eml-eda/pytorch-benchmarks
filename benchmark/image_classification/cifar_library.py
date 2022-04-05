import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import torch.nn.init as init
torch.manual_seed(43)

# Function to retrieve the CIFAR10 benchmark
def get_benchmark():
    # Defining the same data augmentation steps of the TinyML paper for training
    transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    ])

    # Converting to tensor the test images
    test_to_tensor = transforms.Compose([
    transforms.ToTensor()
    ])

    train_val_set = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./',train=False, download=True, transform=test_to_tensor)

    labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_val_set, test_set, labels


# Function to retrieve the training, validation and test dataloaders
def get_dataloaders(config, train_val_set, test_set, PERF_SAMPLE = True):

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

    if PERF_SAMPLE:
        _idxs = np.load('perf_samples_idxs.npy')
        test_set = torch.utils.data.Subset(test_set, _idxs)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=config['batch_size'],
                                                 shuffle=False,
                                                 num_workers=config['batch_size'])
    else:
        test_loader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=config['batch_size'],
                                                 shuffle=False,
                                                 num_workers=config['batch_size'])
  
    return train_loader, val_loader, test_loader


# I define a class ConvBlock to simplify the definition of the network later
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class ResnetV1Eembc(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Resnet v1 parameters
        self.input_shape = [3, 32, 32]  # default size for cifar10
        self.num_classes = 10  # default class number for cifar10
        self.num_filters = 16  # this should be 64 for an official resnet model

        # Resnet v1 layers

        # First stack
        self.inputblock = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.convblock1 = ConvBlock(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Second stack
        self.convblock2 = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2y = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2y.weight)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2x = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv2x.weight)

        # Third stack
        self.convblock3 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3y = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3y.weight)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3x = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv3x.weight)

        self.avgpool = torch.nn.AvgPool2d(8)

        self.out = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.out.weight)


    def forward(self, input):
        # Input layer
        x = self.inputblock(input) # [32, 32, 16]

        # First stack
        y = self.convblock1(x)     # [32, 32, 16]
        y = self.conv1(y)
        y = self.bn1(y)
        x = torch.add(x, y)        # [32, 32, 16]
        x = self.relu(x)

        # Second stack
        y = self.convblock2(x)     # [16, 16, 32]
        y = self.conv2y(y)
        y = self.bn2(y)
        x = self.conv2x(x)         # [16, 16, 32]
        x = torch.add(x, y)        # [16, 16, 32]
        x = self.relu(x)

        # Third stack
        y = self.convblock3(x)     # [8, 8, 64]
        y = self.conv3y(y)
        y = self.bn3(y)
        x = self.conv3x(x)         # [8, 8, 64]
        x = torch.add(x, y)        # [8, 8, 64]
        x = self.relu(x)

        x = self.avgpool(x)        # [1, 1, 64]
        x = torch.squeeze(x)       # [64]
        x = self.out(x)            # [10]

        return x


def lr_schedule(optimizer, epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    for opt in optimizer.param_groups:
        opt['lr'] = lrate


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
  with torch.no_grad():
    for image, target in data_loader:
      step += 1
      image, target = image.to(device), target.to(device)
      output, loss = run_model(model, image, target, criterion, device)
      acc_val = accuracy(output, target, topk=(1,))
      avgacc.update(acc_val[0], image.size(0))
      avgloss.update(loss, image.size(0))     
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
      image, target = image.to(device), target.to(device)
      output, loss = run_model(model, image, target, criterion, device)
      lr_schedule(optimizer, epoch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      acc_val = accuracy(output, target, topk=(1,))
      avgacc.update(acc_val[0], image.size(0))
      avgloss.update(loss, image.size(0))
      if step % 100 == 99:
        tepoch.set_postfix({'loss': avgloss, 'acc': avgacc})
    val_loss, val_acc = evaluate(model, criterion, val, device)
    final_metrics = {
        'loss': avgloss.get(),
        'acc': avgacc.get(),
        'val_loss': val_loss.get(),
        'val_acc': val_acc.get(),
        }
    tepoch.set_postfix(final_metrics)
    tepoch.close()
  return final_metrics