import torch
from torch import nn
import cond_instance_norm_plus_plus


class SequentialWithSigmas(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x: torch.Tensor, sigmas: torch.LongTensor):
        for i in range(len(self.modules_list)):
            module = self.modules_list[i]
            if isinstance(module, cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus):
                x = module(x, sigmas)
            else:
                x = module(x)
        return x
