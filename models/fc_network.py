import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class FullyConnected(nn.Module):
    """Simple fully connected neural network
    It requires sparse matrix rows as input"""
    def __init__(self, dimensions, drop_prob = 0.4):
        super(FullyConnected, self).__init__()
        num_blocks = len(dimensions) // 2
        layers = []
        for i in range(num_blocks-1):
            layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            layers.append(nn.BatchNorm1d())
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = drop_prob))
        self.end_layer = nn.Linear(dimensions[-2], dimensions[-1])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = self.end_layer(out)
        return out

