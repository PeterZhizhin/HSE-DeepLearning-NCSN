import numpy as np
import torch
import torch.utils.data


class PerturbedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sigmas):
        assert isinstance(sigmas, np.ndarray)
        self.sigmas = sigmas
        self.n_sigmas = sigmas.shape[0]
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getattr__(self, item):
        return getattr(self._dataset, item)

    def __getitem__(self, item):
        obj = self._dataset[item]
        if isinstance(obj, tuple):
            assert len(obj) == 1, ('PerturbedDataset: Underlying dataset returned tuple'
                                   'with {} values, expected 1 value').format(len(obj))
            obj = obj[0]
        sigma_idx = np.random.randint(self.n_sigmas)

        sigma = self.sigmas[sigma_idx]
        obj_noise = torch.randn_like(obj) * sigma

        noisy_obj = obj + obj_noise
        return obj, noisy_obj, torch.tensor(sigma_idx)
