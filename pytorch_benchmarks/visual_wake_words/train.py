from typing import Dict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_benchmarks.utils import AverageMeter, accuracy


def get_default_optimizer(net: nn.Module) -> optim.Optimizer:
    return optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)


def get_default_criterion() -> nn.Module:
    return nn.CrossEntropyLoss()


def get_default_scheduler(opt: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
    return optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 30], gamma=0.5)


def _run_model(model, image, target, criterion, device):
    output = model(image)
    loss = criterion(output, target)
    return output, loss


def train_one_epoch(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train: DataLoader,
        val: DataLoader,
        device: torch.device) -> Dict[str, float]:
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for image, target in train:
            step += 1
            tepoch.update(1)
            image, target = image.to(device), target.to(device)
            output, loss = _run_model(model, image, target, criterion, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], image.size(0))
            avgloss.update(loss, image.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss, 'acc': avgacc})
        val_metrics = evaluate(model, criterion, val, device)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get(),
        }
        final_metrics.update(val_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.close()
    return final_metrics


def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device) -> Dict[str, float]:
    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for image, target in data:
            step += 1
            image, target = image.to(device), target.to(device)
            output, loss = _run_model(model, image, target, criterion, device)
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], image.size(0))
            avgloss.update(loss, image.size(0))
        final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get(),
        }
    return final_metrics
