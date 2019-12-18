import torch
from torch import nn
import cond_instance_norm_plus_plus
import with_sigmas_mixin


class SequentialWithSigmas(nn.Module, with_sigmas_mixin.WithSigmasMixin):
    def __init__(self, *args):
        super().__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x: torch.Tensor, sigmas: torch.LongTensor):
        classes_with_sigma_params = (
            with_sigmas_mixin.WithSigmasMixin,
        )
        for i in range(len(self.modules_list)):
            module = self.modules_list[i]
            if isinstance(module, classes_with_sigma_params):
                x = module(x, sigmas)
            else:
                x = module(x)
        return x
