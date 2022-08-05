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


def seed_all(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return seed
