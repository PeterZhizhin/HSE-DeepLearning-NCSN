import torch
from torch import nn

from refinenet.refinenet import RefineNet
import unittest


class RefineNetTest(unittest.TestCase):
    def test_shapes(self):
        test_model = RefineNet(1, 5, nn.ELU, 2)
        test_image_batch = torch.randn((8, 1, 32, 32))
        test_sigmas = torch.zeros(8).long()

        result = test_model(test_image_batch, test_sigmas)
        assert result.shape == (8, 1, 32, 32)


if __name__ == "__main__":
    unittest.main()
