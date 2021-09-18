import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
