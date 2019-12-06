import torch
import torch.nn as nn


class ModelMLP(nn.Module):
    """docstring for ModelMLP"""

    def __init__(self, hidden_size=128):
        super(ModelMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.layers(x)
