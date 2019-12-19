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

    def __init__(self, input_ch, output_ch, kernel_size, block_depth, num_sigmas,
                 padding=None, conv_before=True,
                 activation=nn.ReLU, residual=False, pooling=False,
                 dilation=1, stride_last=1, check_output=True):
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

        if padding is None:
            padding = kernel_size // 2
        conv_list = []
        for i in range(block_depth):
            conv_layer = nn.Conv2d(input_ch if i == 0 else output_ch, output_ch,
                                   kernel_size,
                                   padding=padding,
                                   stride=1 if i != block_depth - 1 else stride_last,
                                   dilation=dilation)
            if conv_before:
                conv_list.append(conv_layer)
            conv_list += [
                ConditionalInstanceNormalizationPlusPlus(output_ch if conv_before or i > 0 else input_ch, num_sigmas),
                activation(),
            ]
            if not conv_before:
                conv_list.append(conv_layer)

        self.conv_net = SequentialWithSigmas(*conv_list)

        self.residual = residual
        self.maybe_skip_x_layer = lambda x: x
        if input_ch != output_ch:
            self.maybe_skip_x_layer = nn.Conv2d(input_ch, output_ch, 1)

        self.maybe_pooling = lambda x: x
        if pooling:
            self.maybe_pooling = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x, sigmas):
        skip_x = self.maybe_skip_x_layer(x)

        logger.debug('ConvBlock: {}'.format(x.shape))
        input_shape = x.shape[2:]
        x = self.conv_net(x, sigmas)
        output_shape = x.shape[2:]
        if self.check_output:
            assert output_shape == input_shape, "Output has different shape: {}/{}".format(input_shape, output_shape)
        else:
            logger.debug('ConvBlock output: {}'.format(x.shape))
        x = self.maybe_pooling(x)

        if self.residual:
            skip_x = self.maybe_pooling(skip_x)
            x = x + skip_x
        return x
