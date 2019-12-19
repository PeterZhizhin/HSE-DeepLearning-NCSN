from torch import nn
from torch.nn import functional as F

from with_sigmas_mixin import WithSigmasMixin
from sequential_with_sigmas import SequentialWithSigmas
from cond_instance_norm_plus_plus import ConditionalInstanceNormalizationPlusPlus
from conv_block import ConvBlock


class RCUBlock(ConvBlock):
    def __init__(self, input_ch, output_ch, activation, num_sigmas, kernel_size=3, block_depth=2):
        super().__init__(input_ch, output_ch,
                         conv_before=False,
                         num_sigmas=num_sigmas,
                         kernel_size=kernel_size,
                         activation=activation,
                         block_depth=block_depth,
                         residual=True)

    @classmethod
    def create_rcu_block(cls, input_ch, output_ch, activation, num_sigmas, num_rcu):
        if num_rcu == 1:
            return cls(input_ch, output_ch, activation, num_sigmas)
        blocks = []
        for i in range(num_rcu):
            blocks.append(
                cls(input_ch if i == 0 else output_ch,
                    output_ch, activation, num_sigmas)
            )
        return SequentialWithSigmas(*blocks)


class MultiResolutionFusion(nn.Module, WithSigmasMixin):
    def __init__(self, in_channels, out_channels, num_sigmas):
        super().__init__()
        self.high_layer = SequentialWithSigmas(
            ConditionalInstanceNormalizationPlusPlus(in_channels, num_sigmas),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.low_layer = SequentialWithSigmas(
            ConditionalInstanceNormalizationPlusPlus(in_channels, num_sigmas),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x_high, x_low, sigmas):
        x_high = self.high_layer(x_high, sigmas)
        x_low = self.low_layer(x_low, sigmas)

        x_low_upsampled = F.interpolate(x_low, size=x_high.shape[2:])
        result = x_low_upsampled + x_high
        return result


class CRPBlock(nn.Module, WithSigmasMixin):
    def __init__(self, channels, activation, num_sigmas, n_blocks, kernel_size=3, padding=1, pool_size=5,
                 pool_padding=2):
        super().__init__()
        self.start_layer = SequentialWithSigmas(
            # ConditionalInstanceNormalizationPlusPlus(in_channels, num_sigmas),
            activation(),
        )

        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(
                SequentialWithSigmas(
                    ConditionalInstanceNormalizationPlusPlus(channels, num_sigmas),
                    nn.AvgPool2d(kernel_size=pool_size, padding=pool_padding, stride=1),
                    nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
                ))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x, sigmas):
        x = self.start_layer(x, sigmas)

        result = x
        for i in range(len(self.blocks)):
            rcu_block = self.blocks[i]
            x = rcu_block(x, sigmas)
            result = result + x

        return result


class RefineBlock(nn.Module, WithSigmasMixin):
    def __init__(self, in_channels, out_channels, activation, num_sigmas,
                 num_inputs,
                 kernel_size=3, pooling_size=5,
                 n_begin_rcu=2, n_end_rcu=1, n_crp=2,
                 in_channels_high=None):
        super().__init__()
        if in_channels_high is None:
            in_channels_high = in_channels
        self.begin_rcu_high = RCUBlock.create_rcu_block(in_channels_high, out_channels,
                                                        activation, num_sigmas, n_begin_rcu)
        self.end_rcu = RCUBlock.create_rcu_block(out_channels, out_channels, activation, num_sigmas, n_end_rcu)
        if num_inputs == 2:
            self.begin_rcu_low = RCUBlock.create_rcu_block(in_channels, out_channels,
                                                           activation, num_sigmas, n_begin_rcu)
            self.multi_res_fusion = MultiResolutionFusion(out_channels, out_channels, num_sigmas)

        self.crp = CRPBlock(out_channels, activation, num_sigmas, n_crp,
                            kernel_size=kernel_size, pool_size=pooling_size)

        self.num_inputs = num_inputs

    def forward(self, all_x, sigmas):
        if self.num_inputs == 2:
            x_high, x_low = all_x
        else:
            x_high = all_x[0]

        x_high_after_begin_rcu = self.begin_rcu_high(x_high, sigmas)
        if self.num_inputs == 2:
            x_low_after_begin_rcu = self.begin_rcu_low(x_low, sigmas)

            crp_input = self.multi_res_fusion(x_high_after_begin_rcu, x_low_after_begin_rcu, sigmas)
        else:
            crp_input = x_high_after_begin_rcu

        crp_output = self.crp(crp_input, sigmas)
        end_rcu_output = self.end_rcu(crp_output, sigmas)

        return end_rcu_output
