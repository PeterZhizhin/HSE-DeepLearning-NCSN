import torch
from torch import nn
import logging
from sequential_with_sigmas import SequentialWithSigmas
from cond_instance_norm_plus_plus import ConditionalInstanceNormalizationPlusPlus
import with_sigmas_mixin

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module, with_sigmas_mixin.WithSigmasMixin):
    """
       Convolutional Block, includes several sequential convolutional and activation layers.
    """

    def __init__(self, input_ch, output_ch, kernel_size, block_depth, num_sigmas, stride_last=1, check_output=True):
        """
            input_ch - input channels num
            output_ch - output channels num
            kernel_size - kernel size for convolution layers
            block_depth - number of convolution + activation repetitions
        """
        super().__init__()
        self.check_output = check_output
        logger.debug('ConvBlock: Creating block of size: ({}, {}, ks={}, depth={})'.format(
            input_ch, output_ch, kernel_size, block_depth))

        padding = kernel_size // 2
        conv_list = []
        for i in range(block_depth):
            conv_list += [
                nn.Conv2d(input_ch if i == 0 else output_ch, output_ch,
                          kernel_size,
                          padding=padding,
                          stride=1 if i != block_depth - 1 else stride_last),
                ConditionalInstanceNormalizationPlusPlus(output_ch, num_sigmas),
                nn.ELU(0.2),
            ]

        self.conv_net = SequentialWithSigmas(*conv_list)

    def forward(self, x, sigmas):
        logger.debug('ConvBlock: {}'.format(x.shape))
        input_shape = x.shape[2:]
        x = self.conv_net(x, sigmas)
        output_shape = x.shape[2:]
        if self.check_output:
            assert output_shape == input_shape, "Output has different shape: {}/{}".format(input_shape, output_shape)
        else:
            logger.debug('ConvBlock output: {}'.format(x.shape))
        return x
