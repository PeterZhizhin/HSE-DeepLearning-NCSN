import torch.utils.data


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, desired_shape, random_func):
        self.desired_shape = desired_shape
        self.random_func = random_func

    def __len__(self):
        return self.desired_shape[0]

    def __getitem__(self, unused_item):
        rand_shape = self.desired_shape[1:]
        return self.random_func(rand_shape)
