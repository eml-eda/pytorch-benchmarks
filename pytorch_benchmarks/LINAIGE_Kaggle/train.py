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
from collections import Counter


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

def _run_model_tcn(model, sample, target, criterion,  class_number):
    output = model(sample[0],sample[1],sample[2])
    if class_number == 2:
        output = output.squeeze()
        target = target.to(torch.float32)
    # import pdb
    # pdb.set_trace()
    loss = criterion(output, target)
    return output, loss

def winner_selection(pred_window):
    # pred_major = []


    count_res = Counter(pred_window).most_common() # return the most common element (class, count)

    # check if the counting number is the only max or there are draws
    if len(count_res) == 1:  # only one class in this window, directly use this class
        pred_winner = count_res[0][0]

    else:  # different class predicitons exist, find the max
        max_count = count_res[0][1]   #initialize the max count by the most common(1)'s counting number
        for i in range(1, len(count_res)):
            if max_count == 1:   # meaning each candidate has one vote
                pred_winner = count_res[-1][0]   # directly use the last predictios as winner
                break
            else:
                if count_res[i][1] < max_count:
                    # select the last winner candidate as the winner, it is the previous res before r[1] < max_count based on most common order returned->first encounterd first order
                    pred_winner = count_res[i-1][0]
                    break
        else:
            # no results in count_res smaller than max_count->all eual
            pred_winner = count_res[-1][0]  # use the last predictios as winner
            # print("error, cannot find winner!")

    # return the winner for each window
        

    return pred_winner


def train_one_epoch(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train: DataLoader,
        device: torch.device,
        classification: bool,
        class_number: int) -> Dict[str, float]:
    
    model.train()

    # Defining avereging for loss, accuracy and ROC
    avgloss = AverageMeter('2.5f')
    avgacc = AverageMeter('6.2f')
    avgroc = AverageMeter('6.2f') 
    
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for sample, target in train:
            
            if type(model).__name__ == 'CNN3_TCN_FC1_pc':
                sample_list = [sample[:, i, :, :].unsqueeze(1) for i in range(sample.shape[1])]
                step += 1
                tepoch.update(1)
                sample, target = sample.to(device), target.to(device)
                output, loss = _run_model_tcn(model, sample_list, target, criterion, class_number)
            else:
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
        print(f'For train_one_epoch: {final_metrics}')
    return final_metrics

def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        classification: bool,
        class_number: int,
        majority_win:int) -> Dict[str, float]:
    
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

    if majority_win == None:
        step = 0
        with torch.no_grad():
            for sample, target in data:
                
                if type(model).__name__ == 'CNN3_TCN_FC1_pc':
                    sample_list = [sample[:, i, :, :].unsqueeze(1) for i in range(sample.shape[1])]
                    step += 1
                    sample, target = sample.to(device), target.to(device)
                    output, loss = _run_model_tcn(model, sample_list, target, criterion, class_number)                    
                else: 
                    step += 1
                    sample, target = sample.to(device), target.to(device)
                    output, loss = _run_model(model, sample, target, criterion, class_number)

                # Computing avereging for metrics
                avgloss.update(loss, sample.size(0)) 
                          
                target_roc = target.detach().numpy()
                # output_roc = output.detach().numpy()
                output_roc = nn.functional.softmax(output, dim=1).detach().numpy()
                roc_truth_stack = np.append(roc_truth_stack, target_roc) 
                
                # To perform roc_auc_score at the end
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
                
    return final_metrics