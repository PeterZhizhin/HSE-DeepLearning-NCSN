from torch import nn
from with_sigmas_mixin import WithSigmasMixin
from conv_block import ConvBlock
from refinenet.refineblock import RefineBlock
from sequential_with_sigmas import SequentialWithSigmas
from cond_instance_norm_plus_plus import ConditionalInstanceNormalizationPlusPlus


class RefineNet(nn.Module, WithSigmasMixin):
    def __init__(self, n_channles, n_filters, activation, num_sigmas, block_depth=2):
        super().__init__()
        self.activation = activation
        self.channels_to_filters = nn.Conv2d(n_channles, n_filters, kernel_size=3, padding=1)

        self.output_1_layer = ConvBlock(n_filters, n_filters,
                                        conv_before=False,
                                        activation=activation,
                                        num_sigmas=num_sigmas,
                                        kernel_size=3,
                                        residual=True,
                                        padding=1,
                                        block_depth=block_depth)
        self.output_2_layer = ConvBlock(n_filters, 2 * n_filters,
                                        conv_before=False,
                                        activation=activation,
                                        num_sigmas=num_sigmas,
                                        kernel_size=3,
                                        residual=True,
                                        block_depth=block_depth,
                                        pooling=True)
        self.output_3_layer = ConvBlock(2 * n_filters, 2 * n_filters,
                                        conv_before=False,
                                        activation=activation,
                                        num_sigmas=num_sigmas,
                                        kernel_size=3,
                                        dilation=2,
                                        padding=2,
                                        residual=True,
                                        block_depth=block_depth)
        self.output_4_layer = ConvBlock(2 * n_filters, 2 * n_filters,
                                        conv_before=False,
                                        activation=activation,
                                        num_sigmas=num_sigmas,
                                        kernel_size=3,
                                        dilation=4,
                                        padding=4,
                                        residual=True,
                                        block_depth=block_depth)

        self.refine_block4 = RefineBlock(2 * n_filters, 2 * n_filters, activation,
                                         num_sigmas, num_inputs=1)
        self.refine_block3 = RefineBlock(2 * n_filters, 2 * n_filters, activation,
                                         num_sigmas, num_inputs=2)
        self.refine_block2 = RefineBlock(2 * n_filters, 2 * n_filters, activation,
                                         num_sigmas, num_inputs=2)
        self.refine_block1 = RefineBlock(2 * n_filters, 2 * n_filters, activation,
                                         num_sigmas, num_inputs=2,
                                         in_channels_high=n_filters)

        self.output_layer = SequentialWithSigmas(
            ConditionalInstanceNormalizationPlusPlus(2 * n_filters, num_sigmas),
            activation(),
            nn.Conv2d(2 * n_filters, n_channles, kernel_size=3, padding=1)
        )

    def forward(self, x, sigmas):
        x = self.channels_to_filters(x)
        output1 = self.output_1_layer(x, sigmas)
        output2 = self.output_2_layer(output1, sigmas)
        output3 = self.output_3_layer(output2, sigmas)
        output4 = self.output_4_layer(output3, sigmas)

        refined_output4 = self.refine_block4([output4], sigmas)
        refined_output3 = self.refine_block3([output3, refined_output4], sigmas)
        refined_output2 = self.refine_block2([output2, refined_output3], sigmas)
        refined_output1 = self.refine_block1([output1, refined_output2], sigmas)

        output = self.output_layer(refined_output1, sigmas)
        return output
