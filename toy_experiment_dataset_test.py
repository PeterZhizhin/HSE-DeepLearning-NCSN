from unittest import TestCase
import numpy as np
import torch

import toy_experiment_dataset


class TestToyExperimentDataset(TestCase):
    def test_generated_data_has_expected_shape(self):
        probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
        means = np.array([[-5, -5],
                          [5, 5]], dtype=np.float)
        num = 10 ** 2
        ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
        self.assertTupleEqual(ds.tensors[0].shape, (10 ** 2, 2))

    def test_generated_data_has_expected_mean(self):
        probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
        means = np.array([[-5, -5],
                          [5, 5]], dtype=np.float)
        num = 10 ** 4

        ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
        ds_mean = ds.tensors[0].mean(axis=0).numpy()

        expected_mean = np.array([3.0, 3.0])
        np.testing.assert_allclose(ds_mean, expected_mean, atol=0.1)

    def test_can_compute_gradients(self):
        probs = np.array([1.0], dtype=np.float)
        means = np.array([[0, 0]], dtype=np.float)
        num = 1
        ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)

        test_dataset = np.array([[0.0, 0.0]])
        gradient = ds.compute_p_gradient(test_dataset)
        self.assertTupleEqual(gradient.shape, (1, 2))
        np.testing.assert_allclose(gradient.cpu().numpy(), np.array([[0, 0]]))

    def test_can_generate_on_many_samples_many_modes(self):
        probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
        means = np.array([[-5, -5],
                          [5, 5]], dtype=np.float)
        num = 1
        ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)

        test_dataset = np.ones((100, 2)) * np.array([-5, 5])
        gradient = ds.compute_p_gradient(test_dataset)
        self.assertTupleEqual(gradient.shape, (100, 2))
