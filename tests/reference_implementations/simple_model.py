"""
Reference PyTorch implementation for simple_model.dl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.d1 = nn.Linear(784, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        # Input shape: (batch_size, 784)
        x = F.relu(self.d1(x))
        x = F.softmax(self.out(x), dim=1)
        return x