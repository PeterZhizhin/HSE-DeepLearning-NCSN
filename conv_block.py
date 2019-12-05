from cond_instance_norm_plus_plus import ConditionalInstanceNormalizationPlusPlus
import torch
from torch import nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_sigmas):
        self.norm1 = ConditionalInstanceNormalizationPlusPlus(num_sigmas, )