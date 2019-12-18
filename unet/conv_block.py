import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
       Convolutional Block, includes several sequential convolutional and activation layers.
    """

    def __init__(self, input_ch, output_ch, kernel_size, block_depth, stride_last=1, check_output=True):
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
                nn.BatchNorm2d(output_ch),
                nn.LeakyReLU(0.2),
            ]

        self.conv_net = nn.Sequential(*conv_list)

    def forward(self, x):
        logger.debug('ConvBlock: {}'.format(x.shape))
        input_shape = x.shape[2:]
        x = self.conv_net(x)
        output_shape = x.shape[2:]
        if self.check_output:
            assert output_shape == input_shape, "Output has different shape: {}/{}".format(input_shape, output_shape)
        else:
            logger.debug('ConvBlock output: {}'.format(x.shape))
        return x
