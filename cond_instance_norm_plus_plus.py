import torch
from torch import nn
import with_sigmas_mixin


class ConditionalInstanceNormalizationPlusPlus(nn.Module, with_sigmas_mixin.WithSigmasMixin):
    def __init__(self, num_channels, num_sigmas):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(num_sigmas, num_channels, 1, 1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(num_sigmas, num_channels, 1, 1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(num_sigmas, num_channels, 1, 1), requires_grad=True)

        self.register_parameter('gamma', self.gamma)
        self.register_parameter('beta', self.beta)
        self.register_parameter('alpha', self.alpha)

    def forward(self, x: torch.Tensor, sigma_values: torch.LongTensor) -> torch.Tensor:
        correct_gammas = self.gamma.index_select(dim=0, index=sigma_values)
        correct_betas = self.beta.index_select(dim=0, index=sigma_values)
        correct_alphas = self.alpha.index_select(dim=0, index=sigma_values)

        # [batch_size, num_channels, h, w]
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_var = x.var(dim=[2, 3], keepdim=True)

        # x_mean: [batch_size, num_channels, 1, 1]
        x_mean_mean = x_mean.mean(dim=1, keepdim=True)
        x_mean_var = x_mean.var(dim=1, keepdim=True)

        x_shifted = (x - x_mean) / torch.sqrt(x_var + 1e-8)
        x_mean_shifted = (x_mean - x_mean_mean) / torch.sqrt(x_mean_var + 1e-8)

        x_shifted_times_gamma = correct_gammas * x_shifted
        x_mean_shifted_times_alpha = correct_alphas * x_mean_shifted
        result = x_shifted_times_gamma + correct_betas + x_mean_shifted_times_alpha

        return result
