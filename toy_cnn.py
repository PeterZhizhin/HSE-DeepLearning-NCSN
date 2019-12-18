import torch
from torch import nn
import cond_instance_norm_plus_plus
import sequential_with_sigmas


class ToyCNN(nn.Module):
    def __init__(self, n_channels, n_hidden_channels, num_sigmas):
        super().__init__()
        self.model = sequential_with_sigmas.SequentialWithSigmas(
            nn.Conv2d(n_channels, n_hidden_channels, kernel_size=1),
            cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(
                n_hidden_channels, num_sigmas
            ),
            nn.ELU(),
            nn.Conv2d(n_hidden_channels, n_hidden_channels, kernel_size=3, padding=1),
            cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(
                n_hidden_channels, num_sigmas
            ),
            nn.ELU(),
            nn.Conv2d(n_hidden_channels, n_channels, kernel_size=1),
        )

    def forward(self, x, sigmas):
        return self.model(x, sigmas)
