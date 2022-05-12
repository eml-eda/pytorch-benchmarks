
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tensorflow import keras
from speech_dscnn import *
import get_dataset as kws_data
import kws_util
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from pytorch_model_summary import summary

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

num_classes = 12

# Define configuration variables
config = {
  # data
  "batch_size": 100,
  "num_workers": 0,
  "val_split": 0,
  # training
  "n_epochs": 36,
  "lr": 1e-05
}

Flags, unparsed = kws_util.parse_command()
#print(Flags)
#print(unparsed)

print('We will download data to {:}'.format(Flags.data_dir))
print('We will train for {:} epochs'.format(Flags.epochs))

ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)

#print(ds_train, len(ds_train))
#print(ds_val, len(ds_val))
#print(ds_test, len(ds_test))

train_set, val_set, test_set = kws_data.get_benchmark(ds_train, ds_val, ds_test)

#print(train_set, len(train_set))
#print(val_set, len(val_set))
#print(test_set, len(test_set))

# Define training, validation and test dataloader
trainLoader, valLoader, testLoader = kws_data.get_dataloaders(config, train_set, val_set, test_set)

#print(trainLoader, len(trainLoader))
#print(valLoader, len(valLoader))
#print(testLoader, len(testLoader))
print("Done getting data")

# Define the model
net = DSCnn(Flags)
if torch.cuda.is_available():
  net = net.cuda()

print(summary(net, torch.zeros((100, 1, 49, 10)), show_input=True))
print(summary(net, torch.zeros((100, 1, 49, 10)), show_input=False, show_hierarchical=True))

# Define the optimizer, the loss and the number of epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=1e-4)

# Training loop
for epoch in range(config['n_epochs']):
  metrics = train_one_epoch(epoch, net, criterion, optimizer, trainLoader, valLoader, device)
test_loss, test_acc = evaluate(net, criterion, testLoader, device)
print("Test Set Loss:", test_loss.get())
print("Test Set Accuracy:", test_acc.get())