import numpy as np
import torch.utils.data
import torch


def gather(self, dim, index):
    """
    Gathers values along an axis specified by ``dim``.

    Taken from this StackOverflow question:
    https://stackoverflow.com/questions/46868056/how-to-gather-elements-of-specific-indices-in-numpy/46868165

    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    Parameters
    ----------
    dim:
        The axis along which to index
    index:
        A tensor of indices of elements to gather

    Returns
    -------
    Output Tensor
    """
    idx_xsection_shape = index.shape[:dim] + \
                         index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)


class ToyExperimentDataset(torch.utils.data.TensorDataset):
    @classmethod
    def generate_numpy_dataset(cls,
                               component_probs,
                               component_means,
                               number_of_elements,
                               random):
        np.testing.assert_allclose(np.sum(component_probs), 1.0)
        num_components = component_probs.shape[0]
        assert component_means.shape[0] == num_components

        desired_shape = (number_of_elements,) + component_means.shape

        std_norm = random.normal(size=desired_shape)
        norm = std_norm + component_means[None]
        chosen_components = random.choice(
            np.arange(num_components, dtype=np.int),
            replace=True, p=component_probs,
            size=(number_of_elements,)
        )
        chosen_components = chosen_components.reshape(-1, 1, 1)
        chosen_values = np.take_along_axis(norm, chosen_components, axis=1)
        return chosen_values

    def __init__(self,
                 component_probs: np.array,
                 component_means: np.array,
                 number_of_elements: int,
                 random_seed=1337):
        self.random = np.random.RandomState(random_seed)
        numpy_dataset = self.generate_numpy_dataset(component_probs, component_means, number_of_elements, self.random)
        self.torch_dataset = torch.from_numpy(numpy_dataset)
        super().__init__(self.torch_dataset)
