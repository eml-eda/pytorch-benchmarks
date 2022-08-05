from typing import Dict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_benchmarks.utils import AverageMeter


def get_default_optimizer(net: nn.Module) -> optim.Optimizer:
    return optim.Adam(net.parameters())


def get_default_criterion() -> nn.Module:
    return nn.MSELoss()


def _run_model(model, audio, target, criterion, device):
    output = model(audio)
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
    avgloss = AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for audio in train:
            step += 1
            tepoch.update(1)
            audio = audio.to(device)
            output, loss = _run_model(model, audio, audio, criterion, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avgloss.update(loss, audio.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss})
        val_metrics = evaluate(model, criterion, val, device)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        final_metrics = {
            'loss': avgloss.get(),
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
    avgloss = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for audio in data:
            step += 1
            audio = audio.to(device)
            output, loss = _run_model(model, audio, audio, criterion, device)
            avgloss.update(loss, audio.size(0))
        final_metrics = {
            'loss': avgloss.get()
        }
    return final_metrics
