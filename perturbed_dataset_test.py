from unittest import TestCase
import numpy as np
import torch
import torch.utils.data

import perturbed_dataset


class PerturbedDatasetTest(TestCase):
    def test_perturbed_dataset_returns_noisy_data(self):
        N = 100
        test_without_noise = torch.zeros(N, 3, 16, 16)
        test_non_noisy_dataset = torch.utils.data.TensorDataset(test_without_noise)
        sigmas = np.array([1.0])

        noisy_dataset = perturbed_dataset.PerturbedDataset(test_non_noisy_dataset, sigmas)
        all_data_dataloader = torch.utils.data.DataLoader(noisy_dataset, batch_size=N)
        all_data = next(iter(all_data_dataloader))

        self.assertEqual(len(all_data), 3)
        no_noise, with_noise, sigma_idx = all_data
        self.assertTupleEqual(no_noise.shape, (N, 3, 16, 16))
        self.assertTupleEqual(with_noise.shape, (N, 3, 16, 16))
        self.assertTupleEqual(sigma_idx.shape, (N, ))

        np.testing.assert_equal(test_without_noise.numpy(), no_noise.numpy())
        mean_noised = torch.mean(with_noise).numpy()
        std_noised = torch.std(with_noise).numpy()

        np.testing.assert_allclose(mean_noised, 0, atol=0.01)
        np.testing.assert_allclose(std_noised, 1, atol=0.01)

        np.testing.assert_equal(sigma_idx.numpy(), np.zeros(sigma_idx.shape, dtype=np.int32))

    def test_perturbed_dataset_selects_sigmas_uniformly(self):
        N = 1000
        test_without_noise = torch.zeros(N, 3, 16, 16)
        test_non_noisy_dataset = torch.utils.data.TensorDataset(test_without_noise)
        sigmas = np.array([1.0, 2.0, 3.0])

        noisy_dataset = perturbed_dataset.PerturbedDataset(test_non_noisy_dataset, sigmas)
        all_data_dataloader = torch.utils.data.DataLoader(noisy_dataset, batch_size=N)
        _, _, sigma_idx = next(iter(all_data_dataloader))
        fractions = [
            (sigma_idx == 0).float().mean().numpy(),
            (sigma_idx == 1).float().mean().numpy(),
            (sigma_idx == 2).float().mean().numpy(),
        ]
        np.testing.assert_allclose(fractions, np.ones((3, )) / 3.0, atol=0.1)
