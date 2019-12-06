"""RefineNet-CRP-RCU-blocks in PyTorch

RefineNet-PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn as nn
import torch.nn.functional as F
import cond_instance_norm_plus_plus
import sequential_with_sigmas


def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes),
            nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes))


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages, num_sigmas):
        super(CRPBlock, self).__init__()
        self.stages = nn.ModuleList(
            [self.generate_stage(in_planes if i == 0 else out_planes, out_planes, num_sigmas) for i in range(n_stages)]
        )
        self.stride = 1
        self.n_stages = n_stages

    def generate_stage(self, in_planes, out_planes, num_sigmas):
        return sequential_with_sigmas.SequentialWithSigmas(
            cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(num_sigmas, in_planes),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            conv3x3(in_planes, out_planes, stride=1, bias=False),
        )

    def forward(self, x, sigma_idx):
        top = x
        for i in range(self.n_stages):
            top = self.stages[i](top, sigma_idx)
            x = top + x
        return x


class RCUBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_blocks, n_stages, num_sigmas):
        super(RCUBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            stages = nn.ModuleList()
            for j in range(n_stages):
                stages.append(
                    self.generate_stage(
                        in_planes if (i == 0) and (j == 0) else out_planes,
                        out_planes,
                        num_sigmas,
                        bias=(j == 0),
                    )
                )
            self.blocks.append(stages)

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages

    def generate_stage(self, in_planes, out_planes, num_sigmas, bias=False):
        return sequential_with_sigmas.SequentialWithSigmas(
            cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(num_sigmas, in_planes),
            nn.ELU(),
            conv3x3(in_planes, out_planes, stride=1, bias=bias)
        )

    def forward(self, x, sigmas_idx):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.blocks[i][j](x, sigmas_idx)
            x += residual
        return x
