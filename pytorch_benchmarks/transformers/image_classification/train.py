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
from typing import Optional, Dict, Any
from math import ceil
import torch
import torch.nn as nn
from pytorch_benchmarks.utils import AverageMeter
from tqdm import tqdm
from timm.utils import accuracy
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing

def get_default_optimizer(model: nn.Module, lr: float = 1e-3, encoder_lr: float = 1e-4, weight_decay: float = 0.05) -> torch.optim.Optimizer:
    return torch.optim.AdamW([{'params': (p for n, p in model.named_parameters() if 'head' in n), 'lr': lr},
                              {'params': (p for n, p in model.named_parameters() if 'head' not in n), 'lr': encoder_lr}], weight_decay=0.05)

def get_default_criterion() -> nn.Module:
    return SoftTargetCrossEntropy()

def get_default_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: Optional[Dict[str, Any]] = dict()) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    update_freq = scheduler_config.get('update_frequency', 1)
    trainset_len = scheduler_config.get('trainset_len', 50000)
    epochs = scheduler_config.get('epochs', 200)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ceil(trainset_len/update_freq)*epochs)

def train_one_epoch(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        update_freq: int,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
        train: torch.utils.data.DataLoader,
        val: torch.utils.data.DataLoader,
        device: torch.device) -> dict:
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    percentile = int(len(train) * 0.1)

    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=model.num_classes)
    random_erasing = RandomErasing(probability=0.25, mode='pixel')

    scaler = torch.cuda.amp.GradScaler()
    with tqdm(total=len(train)) as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}")
        for step, (data, target) in enumerate(train):
            batch_size = data.size(0)
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            data, target = mixup(data, target)
            data = random_erasing(data)
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            if (step + 1) % update_freq == 0 or step == len(train) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            acc = accuracy(output, torch.argmax(target, dim=1), topk=(1,))[0]
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
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                output = model(data)
            # index to one hot
            target = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
            loss = criterion(output, target)
            acc = accuracy(output, torch.argmax(target, dim=1), topk=(1,))[0]
            avgacc.update(acc, batch_size)
            avgloss.update(loss.item(), batch_size)
            step += 1
    final_metrics = {
        'loss': avgloss.get(),
        'acc': avgacc.get(),
    }
    return final_metrics
