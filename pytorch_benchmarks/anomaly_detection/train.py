from typing import Dict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
from pytorch_benchmarks.utils import AverageMeter, calculate_ae_accuracy, \
    calculate_ae_pr_accuracy, calculate_ae_auc
from .data import _file_to_vector_array


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


def test(ds_test, model):
    model.eval()
    test_metrics = {}
    for machine in ds_test:
        y_pred = [0. for k in range(len(machine))]
        y_true = []
        machine_id = ''
        for file_idx, element in tqdm(enumerate(machine), total=len(machine), desc="preprocessing"):
            file_path, label, id = element
            machine_id = id[0]
            y_true.append(label[0].item())
            data = _file_to_vector_array(file_path[0],
                                         n_mels=128,
                                         frames=5,
                                         n_fft=1024,
                                         hop_length=512,
                                         power=2.0
                                         )
            data = data.astype('float32')
            data = torch.from_numpy(data)
            pred = model(data)
            data = data.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            errors = np.mean(np.square(data - pred), axis=1)
            y_pred[file_idx] = np.mean(errors)
        y_true = np.array(y_true, dtype='float64')
        y_pred = np.array(y_pred, dtype='float64')
        acc = calculate_ae_accuracy(y_pred, y_true)
        pr_acc = calculate_ae_pr_accuracy(y_pred, y_true)
        auc = calculate_ae_auc(y_pred, y_true)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
        test_metrics[machine_id] = {
            'acc': acc,
            'pr_acc': pr_acc,
            'auc': auc,
            'p_auc': p_auc
        }
    return test_metrics
