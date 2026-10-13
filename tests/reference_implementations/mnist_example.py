"""
Reference PyTorch implementation for mnist_example.dl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, units, dropout_rate, input_name, prefix=''):
        super(DenseBlock, self).__init__()
        self.dense = nn.Linear(units, units)  # Will adjust based on actual input size
        self.batchnorm = nn.BatchNorm1d(units)
        self.dropout_rate = dropout_rate
        # We'll determine the actual input size from the first forward pass
        self.initialized = False

    def _initialize_linear(self, input_size):
        """Initialize the linear layer with correct input size"""
        if not self.initialized:
            self.dense = nn.Linear(input_size, self.dense.out_features)
            self.initialized = True

    def forward(self, x):
        # Initialize linear layer on first pass if needed
        if not self.initialized:
            self._initialize_linear(x.size(-1))

        x = self.dense(x)
        x = self.batchnorm(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(x)
        return x


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        # We'll handle the input size initialization in the first block
        self.block1 = DenseBlock(units=256, dropout_rate=0.3, input_name='input', prefix='block1_')
        self.block2 = DenseBlock(units=256, dropout_rate=0.3, input_name='block1_out', prefix='block2_')
        self.output_layer = nn.Linear(256, 10)  # Final output to 10 classes

    def forward(self, x):
        # x shape: (batch_size, 784) as defined in INPUT(shape=(784,), name='input')
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        output = F.softmax(self.output_layer(block2_out), dim=1)
        return output