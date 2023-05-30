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
# * Author:  Leonardo Tredese <s302294@studenti.polito.it>                     *
# *----------------------------------------------------------------------------*
import torch
import torch.nn as nn
from pytorch_benchmarks.utils import AverageMeter, accuracy
from tqdm import tqdm

def get_default_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr)

def get_default_criterion() -> nn.Module:
    return nn.CrossEntropyLoss()

def train_one_epoch(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train: torch.utils.data.DataLoader,
        val: torch.utils.data.DataLoader,
        device: torch.device) -> dict:
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    percentile = int(len(train) * 0.1)
    with tqdm(total=len(train)) as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}")
        for step, (data, target) in enumerate(train):
            batch_size = data.size(0)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            acc = accuracy(output, target, topk=(1,))[0]
            avgacc.update(acc, batch_size)
            avgloss.update(loss.item(), batch_size)
            tepoch.update(1)
            if step % percentile == 0:
                tepoch.set_postfix({
                    'train_loss': avgloss.get(),
                    'train_acc': avgacc.get(),
                })
        val_metrics = evaluate(model, criterion, val, device)
        val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
        final_metrics = {
            'train_loss': avgloss.get(),
            'train_acc': avgacc.get(),
        }
        final_metrics.update(val_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.update()
    return final_metrics


def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        val: torch.utils.data.DataLoader,
        device: torch.device) -> dict:
    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for data, target in val:
            batch_size = data.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            acc = accuracy(output, target, topk=(1,))[0]
            avgacc.update(acc, batch_size)
            avgloss.update(loss.item(), batch_size)
            step += 1
    final_metrics = {
        'loss': avgloss.get(),
        'acc': avgacc.get(),
    }
    return final_metrics
