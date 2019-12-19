import torch
from torch import nn
import logging
import conv_block
from sequential_with_sigmas import SequentialWithSigmas
import with_sigmas_mixin

logger = logging.getLogger(__name__)


class DownBlock(nn.Module, with_sigmas_mixin.WithSigmasMixin):
    """
        Encoding block, includes pooling (for shape reduction) and Convolutional Block (ConvBlock)
    """

    def __init__(self, input_ch, output_ch, kernel_size, block_depth, num_sigmas):
        super().__init__()

        self.layers = SequentialWithSigmas(
            nn.MaxPool2d(2),
            conv_block.ConvBlock(input_ch, output_ch, kernel_size, block_depth, num_sigmas),
        )

    def forward(self, x, sigmas):
        logger.debug('DownBlock got shape: {}'.format(x.shape))
        x = self.layers(x, sigmas)
        return x


class UpBlock(nn.Module, with_sigmas_mixin.WithSigmasMixin):
    """
        Decoding block, includes upsampling and Convolutional Block (ConvBlock)
    """

    def __init__(self, input_ch, output_ch, kernel_size, block_depth, num_sigmas):
        super().__init__()

        self.conv_transposed = nn.ConvTranspose2d(input_ch, input_ch, stride=2, kernel_size=4, padding=1)
        self.conv_block = conv_block.ConvBlock(input_ch + output_ch, output_ch, kernel_size, block_depth, num_sigmas)

    def forward(self, copied_input, lower_input, sigmas):
        """
            copied_input - feature map from one of the encoder layers
            lower_input - feature map from previous decoder layer
        """
        x = self.conv_transposed(lower_input)
        logger.debug('UpBlock got shape: input={}, x={}, copied_input={}'.format(
            lower_input.shape, x.shape, copied_input.shape))
        assert x.shape[2:] == copied_input.shape[
                              2:], "Up block got different shapes: copied_input/lower_input/upconv_output {}/{}/{}".format(
            copied_input.shape, lower_input.shape, x.shape)
        x = torch.cat([x, copied_input], dim=1)
        logger.debug('UpBlock x axis after cat: {}'.format(x.shape))
        x = self.conv_block(x, sigmas)
        return x


class UNet(nn.Module, with_sigmas_mixin.WithSigmasMixin):
    def __init__(self, n_classes, feature_levels_num, input_ch_size,
                 hidden_ch_size, block_depth,
                 max_hidden_size,
                 num_sigmas,
                 output_block_depth=None,
                 filters_increase_factor=2, kernel_size=3):
        """
        Input:
            n_classes - number of classes
            feature_levels_num - number of down- and up- block levels
            input_ch_size - input number of channels (1 for gray images, 3 for rgb)
            hidden_ch_size - output number of channels of the first Convolutional Block (in the original paper - 32)
            block_depth - number of convolutions + activations in one Convolutional Block
            kernel_size - kernel size for all convolution layers
        """
        super(UNet, self).__init__()
        self.input_block = conv_block.ConvBlock(input_ch_size, hidden_ch_size, 1, block_depth, num_sigmas)
        self.down_blocks = []
        self.up_blocks = []
        self.feature_levels_num = feature_levels_num
        if output_block_depth is None:
            output_block_depth = block_depth

        prev_hidden_size = hidden_ch_size
        for _ in range(feature_levels_num):
            # your code
            # fill self.down_blocks and self.up_blocks with DownBlock/UpBlock
            # each DownBlock/UpBlock increase/decrease number of channels by 2 times
            next_hidden_size = min(prev_hidden_size * filters_increase_factor, max_hidden_size)
            self.down_blocks.append(DownBlock(prev_hidden_size, next_hidden_size,
                                              kernel_size, block_depth, num_sigmas))
            self.up_blocks.append(UpBlock(next_hidden_size, prev_hidden_size,
                                          kernel_size, block_depth, num_sigmas))
            prev_hidden_size = next_hidden_size

        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        self.output_block = conv_block.ConvBlock(hidden_ch_size, hidden_ch_size,
                                                 kernel_size, output_block_depth, num_sigmas)
        self.output_block_classification = nn.Conv2d(hidden_ch_size, n_classes, 1)

    def forward(self, x, sigmas):
        x = self.input_block(x, sigmas)

        down_blocks_outputs = [x]
        for down_block in self.down_blocks:
            x = down_block(x, sigmas)
            down_blocks_outputs.append(x)
        for i, x in enumerate(down_blocks_outputs):
            logger.debug('Down block #{} shape: {}'.format(i, x.shape))
        down_blocks_outputs = down_blocks_outputs[:-1]

        for i, up_block in enumerate(self.up_blocks[::-1]):
            correct_down_block_output = down_blocks_outputs[-i - 1]
            x = up_block(correct_down_block_output, x, sigmas)

        x = self.output_block(x, sigmas)
        x = self.output_block_classification(x)
        return x
