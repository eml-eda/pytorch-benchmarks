from typing import Dict
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from pytorch_benchmarks.utils import EarlyStopping, AverageMeter
from torchmetrics import AUROC, Accuracy, F1Score, MeanSquaredError

def get_default_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=1e-03)


def get_default_criterion(classification, class_weight=None) -> nn.Module:
    if classification:
      print(f'Criterion is CrossEntropyLoss with class_weight={class_weight}')
      return nn.CrossEntropyLoss(weight=class_weight)
    else:
      print(f'Criterion is BCELoss with class_weight={class_weight}')
      return nn.BCELoss(weight=class_weight)


def get_default_scheduler(opt: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
    reduce_lr = lr_scheduler.ReduceLROnPlateau(opt,
                                               mode='min',
                                               factor=0.3,
                                               patience=5,
                                               min_lr=1e-04,
                                               verbose=True)
    return reduce_lr


def _run_model(model, sample, target, criterion,  class_number):
    output = model(sample)
    if class_number == 2:
        output = output.squeeze()
        target = target.to(torch.float32)
    # import pdb
    # pdb.set_trace()
    loss = criterion(output, target)
    return output, loss


def train_one_epoch(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train: DataLoader,
        device: torch.device,
        classification,
        class_number) -> Dict[str, float]:
    
    model.train()

    # Defining avereging for loss, accuracy and ROC
    avgloss = AverageMeter('2.5f')
    avgacc = AverageMeter('6.2f')
    avgroc = AverageMeter('6.2f')
    
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for sample, target in train:

            step += 1
            tepoch.update(1)
            sample, target = sample.to(device), target.to(device)
            output, loss = _run_model(model, sample, target, criterion, class_number)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Computing avereging for loss, accuracy and ROC
            avgloss.update(loss, sample.size(0))
            
            if classification:
              pred = output.argmax(dim=1)
            
              # sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
              acc = accuracy_score(target, pred)
              avgacc.update(acc, sample.size(0))
            else:
              pred = output.detach().numpy()
          
              # sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
              acc = accuracy_score(target, pred.round())
              avgacc.update(acc, sample.size(0))
            

            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss, 'ACC': avgacc})
        
        # Performing validation part just by forward path  
        final_metrics = {
            'loss': avgloss.get(),
            'ACC': avgacc.get(),
            # 'ROC': avgroc.get()
        }
        # Updating dict values by the ones of validation step
        # final_metrics.update(val_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.close()
    return final_metrics

def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        classification,
        class_number) -> Dict[str, float]:
    
    model.eval()

    # Defining avereging for loss, balanced_accuracy_score, f1_score, mean_squared_error, mean_absolute_error
    avgloss = AverageMeter('2.5f')
    avgacc = AverageMeter('6.2f')
    avgroc = AverageMeter('6.2f')
    avgbas = AverageMeter('6.2f')
    avgf1 = AverageMeter('6.2f')
    avgmse = AverageMeter('6.2f')
    avgmae = AverageMeter('6.2f')

    roc_pred_stack = np.array([])
    roc_truth_stack = np.array([])


    step = 0
    with torch.no_grad():
        for sample, target in data:
            step += 1
            sample, target = sample.to(device), target.to(device)
            output, loss = _run_model(model, sample, target, criterion, class_number)
            
            nanaz = 'yes' if 1 in target else 'no'
            print(torch.unique(target), nanaz)

            # Computing avereging for metrics
            avgloss.update(loss, sample.size(0))
            
            target_roc = target.detach().numpy()
            output_roc = output.detach().numpy()
            roc_truth_stack = np.append(roc_truth_stack, target_roc) 
            
            if classification:
              roc_pred_stack = np.vstack((roc_pred_stack, output_roc)) if roc_pred_stack.size else output_roc
              pred = output.argmax(dim=1)
            else:
              roc_pred_stack = np.append(roc_pred_stack, output_roc) 
              pred = output.detach().numpy().round()
            
            # sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
            acc = accuracy_score(target, pred)
            avgacc.update(acc, sample.size(0))
                        
            # sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
            f1 = f1_score(target, pred, average='weighted')
            avgf1.update(f1, sample.size(0))

            # sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)
            mse = mean_squared_error(target, pred)
            avgmse.update(mse, sample.size(0))
            
            # sklearn.metrics.balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False)
            bas = balanced_accuracy_score(target, pred)
            avgbas.update(bas, sample.size(0))
            
            # sklearn.metrics.mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
            mae = mean_absolute_error(target, pred)
            avgmae.update(mae, sample.size(0))


        print(set(list(roc_truth_stack)))
        if classification:
          roc = roc_auc_score(roc_truth_stack, roc_pred_stack, average='macro', multi_class='ovo', labels=np.array([0,1,2,3]))   
        else:
          roc = roc_auc_score(roc_truth_stack, roc_pred_stack, average='macro', labels=np.array([0,1]))  

        final_metrics = {
            'loss': avgloss.get(),
            'BAS': avgbas.get(),
            'ACC': avgacc.get(),
            'ROC': roc,
            'F1': avgf1.get(),
            'MSE': avgmse.get(),
            'MAE': avgmae.get(),
        }
    return final_metrics

