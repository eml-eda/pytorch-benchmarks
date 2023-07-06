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

from typing import Dict, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, \
    f1_score, mean_squared_error, mean_absolute_error

from pytorch_benchmarks.utils import AverageMeter
from collections import Counter


def get_default_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=1e-03)


def get_default_criterion(classification: bool = True,
                          class_weight: Optional[torch.Tensor] = None) -> nn.Module:
    if classification:
        return nn.CrossEntropyLoss(weight=class_weight)
    else:
        return nn.BCELoss(weight=class_weight)


def get_default_scheduler(opt: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
    reduce_lr = lr_scheduler.ReduceLROnPlateau(opt,
                                               mode='min',
                                               factor=0.3,
                                               patience=5,
                                               min_lr=1e-04,
                                               verbose=True)
    return reduce_lr


def train_one_epoch(epoch: int,
                    model: nn.Module,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    train: DataLoader,
                    device: torch.device,
                    classification: bool,
                    class_number: int,
                    ) -> Dict[str, float]:
    model.train()
    # Defining avereging for loss, accuracy and ROC
    avgloss = AverageMeter('2.5f')
    avgacc = AverageMeter('6.2f')
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for sample, target in train:
            step += 1
            tepoch.update(1)
            sample, target = sample.to(device), target.to(device)
            if type(model).__name__ == 'CNN_TCN':
                sample_list = [sample[:, i, :, :].unsqueeze(1) for i in range(sample.shape[1])]
                output, loss = _run_model_tcn(model, sample_list, target, criterion, class_number)
            else:
                output, loss = _run_model(model, sample, target, criterion, class_number)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Compute average loss and average acc
            avgloss.update(loss, sample.size(0))
            if classification:
                pred = output.argmax(dim=1)
                acc = accuracy_score(target, pred)
            else:
                pred = output.detach().numpy()
                acc = accuracy_score(target, pred.round())
            avgacc.update(acc, sample.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss, 'ACC': avgacc})
        final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get(),
        }
        tepoch.set_postfix(final_metrics)
        tepoch.close()
    return final_metrics


def evaluate(model: nn.Module,
             criterion: nn.Module,
             data: DataLoader,
             device: torch.device,
             classification: bool,
             class_number: int,
             majority_win: Optional[int] = None,
             ) -> Dict[str, float]:
    model.eval()
    avgloss = AverageMeter('2.5f')
    avgacc = AverageMeter('6.2f')
    avgbas = AverageMeter('6.2f')
    avgf1 = AverageMeter('6.2f')
    avgmse = AverageMeter('6.2f')
    avgmae = AverageMeter('6.2f')

    roc_pred_stack = np.array([])
    roc_truth_stack = np.array([])

    if majority_win is None:
        step = 0
        with torch.no_grad():
            for sample, target in data:
                step += 1
                sample, target = sample.to(device), target.to(device)
                if type(model).__name__ == 'CNN_TCN':
                    sample_list = [sample[:, i, :, :].unsqueeze(1) for i in range(sample.shape[1])]
                    output, loss = _run_model_tcn(model, sample_list, target,
                                                  criterion, class_number)
                else:
                    output, loss = _run_model(model, sample, target, criterion, class_number)
                # Computing average loss
                avgloss.update(loss, sample.size(0))
                # Compute roc
                target_roc = target.detach().numpy()
                output_roc = F.softmax(output, dim=1).detach().numpy()
                roc_truth_stack = np.append(roc_truth_stack, target_roc)
                # To perform roc_auc_score at the end
                if classification:
                    if roc_pred_stack.size:
                        roc_pred_stack = np.vstack((roc_pred_stack, output_roc))
                    else:
                        roc_pred_stack = output_roc
                    pred = output.argmax(dim=1)
                else:
                    roc_pred_stack = np.append(roc_pred_stack, output_roc)
                    pred = output.detach().numpy().round()
                # Compute accuracy
                acc = accuracy_score(target, pred)
                avgacc.update(acc, sample.size(0))
                # Compute F1 score
                f1 = f1_score(target, pred, average='weighted')
                avgf1.update(f1, sample.size(0))
                # Compute MSE
                mse = mean_squared_error(target, pred)
                avgmse.update(mse, sample.size(0))
                # Compute Balanced Accuracy
                bas = balanced_accuracy_score(target, pred)
                avgbas.update(bas, sample.size(0))
                # Compute MAE
                mae = mean_absolute_error(target, pred)
                avgmae.update(mae, sample.size(0))
            # Finalize roc calculation
            if classification:
                roc = roc_auc_score(roc_truth_stack, roc_pred_stack,
                                    average='macro', multi_class='ovo',
                                    labels=np.array([0, 1, 2, 3]))
            else:
                roc = roc_auc_score(roc_truth_stack, roc_pred_stack,
                                    average='macro', labels=np.array([0, 1]))

            final_metrics = {
                'loss': avgloss.get(),
                'BAS': avgbas.get(),
                'ACC': avgacc.get(),
                'ROC': roc,
                'F1': avgf1.get(),
                'MSE': avgmse.get(),
                'MAE': avgmae.get(),
            }
    '''
    else:
        pred_labels = np.array([])
        truth_labels = np.array([])

        with torch.no_grad():
            for sample, target in data:

                sample, target = sample.to(device), target.to(device)

                for i in range(len(sample)):

                    output = model(sample[i])

                    if classification:
                        pred = output.argmax(dim=1)
                        winner_class = winner_selection(pred.numpy())
                        pred_labels = np.append(pred_labels, winner_class)
                    else:
                        pred = output.detach().numpy().round()
                        winner_class = winner_selection(pred)
                        pred_labels = np.append(pred_labels, winner_class)

            bas = balanced_accuracy_score(target, pred_labels)
            acc = accuracy_score(target, pred_labels)
            f1 = f1_score(target, pred_labels, average='weighted')
            mse = mean_squared_error(target, pred_labels)
            mae = mean_absolute_error(target, pred_labels)

            final_metrics = {
                        'loss':0,
                        'BAS': bas,
                        'ACC': acc,
                        'ROC': 0,
                        'F1': f1,
                        'MSE': mse,
                        'MAE': mae,
                        }
    '''
    return final_metrics


def _run_model(model, sample, target, criterion, class_number):
    output = model(sample)
    if class_number == 2:
        output = output.squeeze()
        target = target.to(torch.float32)
    loss = criterion(output, target)
    return output, loss


def _run_model_tcn(model, sample, target, criterion, class_number):
    output = model(sample[0], sample[1], sample[2])
    if class_number == 2:
        output = output.squeeze()
        target = target.to(torch.float32)
    loss = criterion(output, target)
    return output, loss


# TODO: Check
def _winner_selection(pred_window):
    count_res = Counter(pred_window).most_common()  # return the most common element (class, count)

    # check if the counting number is the only max or there are draws
    if len(count_res) == 1:  # only one class in this window, directly use this class
        pred_winner = count_res[0][0]

    else:  # different class predicitons exist, find the max
        # Initialize the max count by the most common(1)'s counting number
        max_count = count_res[0][1]
        for i in range(1, len(count_res)):
            if max_count == 1:   # meaning each candidate has one vote
                pred_winner = count_res[-1][0]   # directly use the last predictios as winner
                break
            else:
                if count_res[i][1] < max_count:
                    # select the last winner candidate as the winner,
                    # it is the previous res before r[1] < max_count based on
                    # most common order returned->first encounterd first order
                    pred_winner = count_res[i-1][0]
                    break
        else:
            # no results in count_res smaller than max_count->all eual
            pred_winner = count_res[-1][0]  # use the last predictios as winner
            # print("error, cannot find winner!")

    # return the winner for each window
    return pred_winner
