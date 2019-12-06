import numpy as np
import torch.utils.data
import torch
import torch.distributions as td
import typing


class ToyExperimentDataset(torch.utils.data.TensorDataset):
    @classmethod
    def generate_numpy_dataset(cls,
                               component_probs,
                               component_means,
                               number_of_elements,
                               random,
                               sigma=1):
        np.testing.assert_allclose(np.sum(component_probs), 1.0)
        num_components = component_probs.shape[0]
        assert component_means.shape[0] == num_components
        desired_shape = (number_of_elements,) + component_means.shape

        std_norm = random.normal(size=desired_shape)
        norm = std_norm * sigma + component_means[None]
        chosen_components = random.choice(
            np.arange(num_components, dtype=np.int),
            replace=True, p=component_probs,
            size=(number_of_elements,)
        )
        chosen_components = chosen_components.reshape(-1, 1, 1)
        chosen_values = np.take_along_axis(norm, chosen_components, axis=1)
        return chosen_values[:, 0, :]

    def __init__(self,
                 component_probs: np.array,
                 component_means: np.array,
                 number_of_elements: int,
                 random_seed=1337,
                 sigma=1):
        self.random = np.random.RandomState(random_seed)
        self.component_probs = component_probs
        self.component_means = component_means
        self.sigma = sigma
        numpy_dataset = self.generate_numpy_dataset(component_probs, component_means, number_of_elements, self.random)
        self.torch_dataset = torch.from_numpy(numpy_dataset)
        super().__init__(self.torch_dataset)

    def compute_log_p(self, x: typing.Union[np.array, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, requires_grad=True).double()
        log_probs = []
        for mean, prob in zip(self.component_means, self.component_probs):
            dist = td.multivariate_normal.MultivariateNormal(
                torch.tensor(mean, requires_grad=True).double(),
                torch.eye(mean.shape[0], requires_grad=True).double() * self.sigma
            )
            log_prob_comp = torch.ones(1, requires_grad=True).double() * np.log(prob)
            log_prob_dist = dist.log_prob(x)
            log_probs.append(log_prob_dist + log_prob_comp)
        log_probs = torch.stack(log_probs, dim=0)
        total_log_prob = torch.logsumexp(log_probs, dim=0)
        return total_log_prob

    def compute_p_gradient(self, x: typing.Union[np.array, torch.Tensor]):
        x = torch.tensor(x, requires_grad=True).double()
        total_log_prob = self.compute_log_p(x)
        # total_prob = torch.exp(total_log_prob)
        # Only one x contribues to each part of the sum
        total_prob_sum = torch.sum(total_log_prob)

        gradient = torch.autograd.grad(outputs=total_prob_sum, inputs=x)[0]
        return gradient
