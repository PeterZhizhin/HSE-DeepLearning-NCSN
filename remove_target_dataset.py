import torch.utils.data


class DatasetWithoutTarget(torch.utils.data.Dataset):
    def __init__(self, dataset_with_target: torch.utils.data.Dataset):
        self._dataset_with_target = dataset_with_target

    def __getattr__(self, item):
        return getattr(self._dataset_with_target, item)

    def __len__(self):
        return len(self._dataset_with_target)

    def __getitem__(self, item):
        obj, unused_target = self._dataset_with_target[item]
        return obj
