import torch
import torch.nn as nn
from torch.nn import functional as F
from .CustomEnums import LossFunction, Norm
import numpy as np

cce_loss = nn.CrossEntropyLoss()

def get_loss(conf, loss_function, X, delta, logits, y, args=None):
    if loss_function == LossFunction.ce:
        return F.cross_entropy(logits, y)
    elif loss_function == LossFunction.ce_scaled:
        scaled_logits_adv = logits / torch.max(torch.abs(logits), 1, keepdim=True).values * 10
        loss = F.cross_entropy(scaled_logits_adv, y)
        return loss
    raise Exception("No valid loss function!")

def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    loss = (x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (x_sorted[:, -1] - .5 * x_sorted[:, -3] - .5 * x_sorted[:, -4] + 1e-12)
    return loss

def get_norm(X, norm):
    if norm == Norm.linf:
        n = torch.norm(X.view(X.shape[0], -1), p=float("inf"), dim=1)
    elif norm == Norm.l2:
        n = torch.norm(X.view(X.shape[0], -1), p=2, dim=1)
    return n
