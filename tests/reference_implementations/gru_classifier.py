"""
Reference PyTorch implementation for gru_classifier.dl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    def __init__(self):
        super(GRUClassifier, self).__init__()
        # Input shape: (sequence_length, batch_size, input_features) or (batch_size, sequence_length, input_features)
        # The DL doesn't specify batch_first, but we'll assume batch_first=True for convenience
        # INPUT(shape=(None, None), name='input_sequence') means variable length sequence, variable features

        # GRU layer: return_sequences=False means we only return the last output
        self.gru = nn.GRU(
            input_size=300,  # We'll need to determine this from context or make it flexible
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            return_sequences=False  # Only return last hidden state
        )

        # Classification layers
        self.dense1 = nn.Linear(128, 64)
        # Optional dropout: # dropout = Dropout([gru_out], rate=0.5, name='dropout')
        self.output_layer = nn.Linear(64, 10)  # num_classes = 10

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_features)
        # GRU output: (batch_size, hidden_size) when return_sequences=False
        gru_out, _ = self.gru(x)

        # Optional dropout: commented out in DL
        # gru_out = F.dropout(gru_out, p=0.5, training=self.training)

        # Classification layers
        dense1_out = F.relu(self.dense1(gru_out))
        # dropout2 = Dropout([dense1], rate=0.5, name='dropout2')  # Optional - commented out

        # Output layer with softmax
        output = F.softmax(self.output_layer(dense1_out), dim=1)
        return output