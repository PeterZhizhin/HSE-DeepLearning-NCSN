from unittest import TestCase
import torch
import numpy as np

import toy_cnn


class TestToyCNN(TestCase):
    def test_model_returns_expected_image_shape(self):
        model = toy_cnn.ToyCNN(3, 6, 2)
        test_x = torch.ones(8, 3, 16, 16)
        sigmas = torch.from_numpy(np.array([0, 0, 1, 0, 1, 0, 1, 1])).long()

        test_y = model(test_x, sigmas)
        self.assertTupleEqual(test_y.shape, (8, 3, 16, 16))
