from unittest import TestCase
from cond_instance_norm_plus_plus import ConditionalInstanceNormalizationPlusPlus

import torch
import numpy as np


class TestConditionalInstanceNormalizationPlusPlus(TestCase):
    def test_forward_has_correct_shape(self):
        norm = ConditionalInstanceNormalizationPlusPlus(2, 3)
        test_x = torch.ones(8, 3, 16, 16)
        sigmas = torch.from_numpy(np.array([0, 0, 1, 0, 1, 0, 1, 1])).long()

        out = norm(test_x, sigmas)

        self.assertTupleEqual(test_x.shape, out.shape)

    def test_model_has_three_parameters(self):
        norm = ConditionalInstanceNormalizationPlusPlus(2, 3)
        self.assertEqual(len(list(norm.parameters())), 3)
        for param in norm.parameters():
            self.assertEqual(np.prod(param.size()), 2 * 3)
