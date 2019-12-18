from torch import nn
from unittest import TestCase
import numpy as np
import torch
import mock

import sequential_with_sigmas
import cond_instance_norm_plus_plus


class TestSequentialWithSigmas(TestCase):
    def test_forward_returns_correct_shape(self):
        test_class = sequential_with_sigmas.SequentialWithSigmas(
            nn.Conv2d(1, 1, 1),
            cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(1, 2),
            nn.Conv2d(1, 1, 1),
            cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(1, 2)
        )
        test_x = torch.ones(8, 1, 8, 8)
        sigmas = torch.from_numpy(np.array([0, 0, 1, 0, 1, 0, 1, 1])).long()

        test_y = test_class(test_x, sigmas)
        self.assertTupleEqual(test_y.shape, (8, 1, 8, 8))

    def test_forward_calls_cond_inst_norm_pp_with_sigmas(self):
        cond_inst_norm_pp = cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(1, 2)
        cond_inst_norm_pp.forward = mock.MagicMock(return_value=torch.ones(8, 1, 8, 8))
        test_class = sequential_with_sigmas.SequentialWithSigmas(
            cond_inst_norm_pp,
        )
        test_x = torch.ones(8, 1, 8, 8)
        sigmas = torch.from_numpy(np.array([0, 0, 1, 0, 1, 0, 1, 1])).long()

        test_y = test_class(test_x, sigmas)
        cond_inst_norm_pp.forward.assert_called_once_with(test_x, sigmas)

    def test_class_can_be_called_with_seq_with_sigmas_inside(self):
        test_class = sequential_with_sigmas.SequentialWithSigmas(
            nn.Conv2d(1, 1, 1),
            sequential_with_sigmas.SequentialWithSigmas(
                nn.Conv2d(1, 1, 1),
                cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(1, 2),
            ),
            cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(1, 2),
            nn.Conv2d(1, 1, 1),
            cond_instance_norm_plus_plus.ConditionalInstanceNormalizationPlusPlus(1, 2)
        )
        test_x = torch.ones(8, 1, 8, 8)
        sigmas = torch.from_numpy(np.array([0, 0, 1, 0, 1, 0, 1, 1])).long()

        test_y = test_class(test_x, sigmas)
        self.assertTupleEqual(test_y.shape, (8, 1, 8, 8))
