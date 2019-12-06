import torch
from torch import nn
import cond_instance_norm_plus_plus


class SequentialWithSigmas(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def forward(self, x: torch.Tensor, sigmas: torch.LongTensor):
        for module in self.modules:
            if isinstance(module, cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus):
                x = module(x, sigmas)
            else:
                x = module(x)
        return x
