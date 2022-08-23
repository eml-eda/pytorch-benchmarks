import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, fmt='f', name='meter'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self):
        return float(self.avg)

    def __str__(self):
        fmtstr = '{:' + self.fmt + '}'
        return fmtstr.format(float(self.avg))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calculate_ae_accuracy(y_pred, y_true):
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        correct = np.sum(y_pred_binary == y_true)
        accuracy_tmp = 100 * correct / len(y_pred_binary)
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp
    return accuracy


def calculate_ae_pr_accuracy(y_pred, y_true):
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    n_normal = np.sum(y_true == 0)
    precision = np.zeros(len(thresholds))
    recall = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build matrix of TP, TN, FP and FN
        false_positive = np.sum((y_pred_binary[0:n_normal] == 1))
        true_positive = np.sum((y_pred_binary[n_normal:] == 1))
        false_negative = np.sum((y_pred_binary[n_normal:] == 0))
        # Calculate and store precision and recall
        precision[threshold_item] = true_positive / (true_positive + false_positive)
        recall[threshold_item] = true_positive / (true_positive + false_negative)
        # See if the accuracy has improved
        accuracy_tmp = 100 * (precision[threshold_item] + recall[threshold_item]) / 2
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp
    return accuracy


def calculate_ae_auc(y_pred, y_true):
    """
    Autoencoder ROC AUC calculation
    """
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.01, .01) * (np.amax(y_pred) - np.amin(y_pred))
    roc_auc = 0

    n_normal = np.sum(y_true == 0)
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(1, len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build TP and FP
        tpr[threshold_item] = np.sum((y_pred_binary[n_normal:] == 1)
                                     ) / float(len(y_true) - n_normal)
        fpr[threshold_item] = np.sum((y_pred_binary[0:n_normal] == 1)) / float(n_normal)

    # Force boundary condition
    fpr[0] = 1
    tpr[0] = 1

    # Integrate
    for threshold_item in range(len(thresholds) - 1):
        roc_auc += .5 * (tpr[threshold_item] + tpr[threshold_item + 1]) * (
            fpr[threshold_item] - fpr[threshold_item + 1])
    return roc_auc


def seed_all(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return seed
