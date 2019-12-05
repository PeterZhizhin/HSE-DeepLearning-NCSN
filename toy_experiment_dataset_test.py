from unittest import TestCase
import numpy as np
import torch

import toy_experiment_dataset


class TestToyExperimentDataset(TestCase):
    def test_generated_data_has_expected_mean(self):
        probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
        means = np.array([[-5, -5],
                          [5, 5]], dtype=np.float)
        num = 10 ** 4

        ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
        ds_mean = ds.tensors[0].mean(axis=0).numpy()[0]

        expected_mean = np.array([3.0, 3.0])
        np.testing.assert_allclose(ds_mean, expected_mean, atol=0.1)
