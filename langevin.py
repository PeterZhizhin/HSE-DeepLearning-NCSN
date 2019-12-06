from tqdm import trange, tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sliced_sm import sliced_score_estimation_vr
import model_mlp
import toy_experiment_dataset


def langevin(probs, means, num, input, lr=0.01, step=1000):
    ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, 1)
    for i in trange(step):
        # print(input.dtype)
        input += lr * ds.compute_p_gradient(input).float().detach() / 2
        input += torch.randn_like(input) * np.sqrt(lr)
    # print(input)
    return input


def anneal_langevin(probs, means, num, input, sigmas, lr=0.01, step=1000):
    for s in sigmas:
        ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, 1, sigma=s)
        for i in trange(step):
            # print(s, sigmas[-1])
            lr_new = lr * np.power(s / sigmas[-1], 2)
            input += lr_new * ds.compute_p_gradient(input).float().detach() / 2
            input += torch.randn_like(input) * np.sqrt(lr_new)
    # print(input)
    return input


def toy_generate():
    probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
    means = np.array([[-5, -5], [5, 5]], dtype=np.float)
    num = 1280

    ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)

    dataset = ds.tensors[0].numpy()
    # Fig3a samples
    d1 = (dataset[:, 0] < 0).sum() / 1280
    d2 = (dataset[:, 0] > 0).sum() / 1280
    plt.scatter(dataset[dataset[:, 0] < 0, 0], dataset[dataset[:, 1] < 0, 1], s=1, label='Доля = ' + str(d1))
    plt.scatter(dataset[dataset[:, 0] > 0, 0], dataset[dataset[:, 1] > 0, 1], s=1, label='Доля = ' + str(d2))
    plt.title('Samples')
    plt.legend()
    plt.savefig("figures/fig3a.svg")
    plt.show()

    # Fig3b

    # ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
    start_point = torch.rand(1280, 2) * 16 - 8
    # print(start_point)
    after_lan = langevin(probs, means, num, start_point, lr=0.1, step=1000).detach().numpy()
    d1 = (after_lan[:, 0] < 0).sum() / 1280
    d2 = (after_lan[:, 0] > 0).sum() / 1280
    plt.scatter(after_lan[after_lan[:, 0] < 0, 0], after_lan[after_lan[:, 1] < 0, 1], s=1, label='Доля = ' + str(d1))
    plt.scatter(after_lan[after_lan[:, 0] > 0, 0], after_lan[after_lan[:, 1] > 0, 1], s=1, label='Доля = ' + str(d2))
    plt.title('Langevin')
    plt.legend()
    plt.savefig("figures/fig3b.svg")
    plt.show()

    # Fig3c

    sigmas = np.geomspace(20, 0.7,
                          10)  # np.geomspace(2, 0.1, 10)#np.exp(np.linspace(np.log(20), 0., 10))#np.geomspace(10, 0.1, 10)
    # print(sigmas)
    # ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
    start_point = torch.rand(1280, 2) * 16 - 8
    after_lan = anneal_langevin(probs, means, num, start_point, sigmas, lr=0.1, step=100).detach().numpy()
    d1 = (after_lan[:, 0] < 0).sum() / 1280
    d2 = (after_lan[:, 0] > 0).sum() / 1280
    plt.scatter(after_lan[after_lan[:, 0] < 0, 0], after_lan[after_lan[:, 1] < 0, 1], s=1, label='Доля = ' + str(d1))
    plt.scatter(after_lan[after_lan[:, 0] > 0, 0], after_lan[after_lan[:, 1] > 0, 1], s=1, label='Доля = ' + str(d2))
    plt.title('Annealed Langevin')
    plt.legend()
    plt.savefig("figures/fig3c.svg")
    plt.show()


def create_model_from_save(save_path):
    model = model_mlp.ModelMLP(128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if save_path:
        try:
            save_file = torch.load(save_path)
            model.load_state_dict(save_file['model'])
            optimizer.load_state_dict(save_file['optimizer'])
        except FileNotFoundError:
            print('Model file does not exist yet.')
    return model, optimizer


def save_model_opt(model, optimizer, save_path):
    if save_path:
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(save_dict, save_path)


def train(model, optimizer):
    probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
    means = np.array([[-5, -5], [5, 5]], dtype=np.float)
    num = 128 * 10 ** 4
    batch_size = 128
    toy_ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
    toy_dl = torch.utils.data.DataLoader(toy_ds, batch_size=batch_size)
    for points in tqdm(toy_dl, desc='Training model', total=num // batch_size):
        # input = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
        # print(points)
        loss, _, _ = sliced_score_estimation_vr(model, points[0].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def parse_args(args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--save_path', nargs='?', default='')
    arg_parser.add_argument('--train', action='store_true', default=False)
    return arg_parser.parse_args(args)


def generate_field_map(model):
    probabilities = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
    means = np.array([[-5, -5],
                      [5, 5]], dtype=np.float)
    toy_ds = toy_experiment_dataset.ToyExperimentDataset(probabilities, means, 1)

    n = 20
    xs = np.linspace(-8, 8, n)
    ys = np.linspace(-8, 8, n)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.flatten()
    yy = yy.flatten()
    grid_values = np.stack((xx, yy), axis=1)

    log_density = toy_ds.compute_log_p(grid_values).detach().numpy()
    density_color_grid = log_density.reshape(n, n)

    true_gradient = toy_ds.compute_p_gradient(grid_values).numpy()
    plt.pcolormesh(xs, ys, density_color_grid)
    plt.quiver(grid_values[:, 0], grid_values[:, 1], true_gradient[:, 0], true_gradient[:, 1])
    plt.title('Data scores')
    plt.savefig('figures/fig2a.svg')
    plt.show()

    grid_values_torch = torch.tensor(grid_values).float()
    with torch.no_grad():
        estimated_gradient = model(grid_values_torch).numpy()

    plt.pcolormesh(xs, ys, density_color_grid)
    plt.quiver(grid_values[:, 0], grid_values[:, 1], estimated_gradient[:, 0], estimated_gradient[:, 1])
    plt.title('Estimated scores')
    plt.savefig('figures/fig2b.svg')
    plt.show()


def main(args=None):
    if isinstance(args, str):
        args = [__name__] + args.split()
    args = parse_args(args)
    model, optimizer = create_model_from_save(args.save_path)
    if args.train:
        train(model, optimizer)
        save_model_opt(model, optimizer, args.save_path)
    generate_field_map(model)
    toy_generate()


if __name__ == '__main__':
    main()
