from typing import Dict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_benchmarks.utils import AverageMeter


class LogCosh(nn.Module):
    def __init__(self):
        super(LogCosh, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = input - target
        return torch.mean(x + nn.Softplus()(-2*x) - torch.log(torch.tensor(2.)))


def get_default_optimizer(net: nn.Module) -> optim.Optimizer:
    return optim.Adam(net.parameters(), lr=0.001)


def get_default_criterion() -> nn.Module:
    return LogCosh()


def _run_model(model, sample, target, criterion, device):
    output = model(sample)
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
    avgmae = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for sample, target in train:
            step += 1
            tepoch.update(1)
            sample, target = sample.to(device), target.to(device)
            output, loss = _run_model(model, sample, target, criterion, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mae_val = F.l1_loss(output, target)
            avgmae.update(mae_val, sample.size(0))
            avgloss.update(loss, sample.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss, 'MAE': avgmae})
        val_metrics = evaluate(model, criterion, val, device)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        final_metrics = {
            'loss': avgloss.get(),
            'MAE': avgmae.get(),
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
    avgmae = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for sample, target in data:
            step += 1
            sample, target = sample.to(device), target.to(device)
            output, loss = _run_model(model, sample, target, criterion, device)
            mae_val = F.l1_loss(output, target)
            avgmae.update(mae_val, sample.size(0))
            avgloss.update(loss, sample.size(0))
        final_metrics = {
            'loss': avgloss.get(),
            'MAE': avgmae.get(),
        }
    return final_metrics
