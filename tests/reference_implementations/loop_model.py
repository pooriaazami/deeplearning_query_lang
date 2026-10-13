"""
Reference PyTorch implementation for loop_model.dl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoopModel(nn.Module):
    def __init__(self):
        super(LoopModel, self).__init__()
        # Shared dense layer that gets reused 3 times
        self.shared_dense = nn.Linear(64, 64)
        # Final output layer
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        # Input shape: (batch_size, 64)
        # Apply shared dense layer 3 times with ReLU activation
        for _ in range(3):
            x = F.relu(self.shared_dense(x))
        # Final output layer with softmax
        x = F.softmax(self.output(x), dim=1)
        return x