import toy_experiment_dataset

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def plot_2d_tensor(x: torch.Tensor):
    x = x.cpu().numpy()
    assert x.ndim == 2, str(x.shape)
    assert x.shape[1] == 2, str(x.shape)

    sns.scatterplot(x[:, 0], x[:, 1])


def main():
    probabilities = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
    means = np.array([[-5, -5],
                      [5, 5]], dtype=np.float)
    num_generate = 1280
    toy_ds = toy_experiment_dataset.ToyExperimentDataset(probabilities, means, num_generate)
    plot_2d_tensor(toy_ds.tensors[0])
    plt.savefig('generated_toy_data.png')

    xs = np.linspace(-8, 8, 20)
    ys = np.linspace(-8, 8, 20)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.flatten()
    yy = yy.flatten()
    grid_values = np.stack((xx, yy), axis=1)
    evaluated_gradient = toy_ds.compute_p_gradient(grid_values).numpy()
    evaluated_gradient /= np.sqrt(np.max(np.sum(evaluated_gradient ** 2, axis=1)))
    plt.quiver(grid_values[:, 0], grid_values[:, 1], evaluated_gradient[:, 0], evaluated_gradient[:, 1])
    plt.savefig('gradient_plot.png')


if __name__ == "__main__":
    main()
