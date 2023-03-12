import torch
from torch import nn


class modelWrapper(nn.Module):
    def __init__(self, model):
        super(modelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)

        return logits