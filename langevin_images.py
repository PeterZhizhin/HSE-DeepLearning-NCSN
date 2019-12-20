import argparse
import logging
import torchvision

import langevin_training_loop
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--dataset_folder', default='dataset')
    parser.add_argument('--download_dataset', action='store_true', default=False)
    parser.add_argument('--sigma_start', type=float, default=1)
    parser.add_argument('--sigma_end', type=float, default=0.01)
    parser.add_argument('--num_sigmas', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_path', nargs='?', default='langevin_model')
    parser.add_argument('--log', nargs='?', default='INFO')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--show_every', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--show_grid_size', type=int, default=8)
    parser.add_argument('--image_dim', type=int, default=32)
    parser.add_argument('--n_processes', nargs='?', default=0)
    parser.add_argument('--target_device', nargs='?', default='cpu')
    return parser.parse_args()


def create_dataset(args):
    dataset_parameters_map = {
        'mnist': (torchvision.datasets.mnist.MNIST, 1, 28, 64),
        'cifar': (torchvision.datasets.cifar.CIFAR10, 3, 32, 128),
    }
    dataset_func, num_channels, image_dim, num_filters = dataset_parameters_map[args.dataset]
    dataset = dataset_func(
        args.dataset_folder,
        train=True,
        download=args.download_dataset,
        transform=torchvision.transforms.ToTensor(),
    )
    return dataset, num_channels, image_dim, num_filters


def main():
    args = parse_args()

    log_level = getattr(logging, args.log.upper())
    logging.basicConfig(level=log_level)

    dataset, num_channels, image_dim, num_filters = create_dataset(args)
    image_shape = (num_channels, image_dim, image_dim)

    sigmas = np.geomspace(args.sigma_start, args.sigma_end, num=args.num_sigmas)
    langevin_model_with_loop = langevin_training_loop.LangevinCNN(num_filters=num_filters,
                                                                  sigmas=sigmas,
                                                                  images_dataset=dataset,
                                                                  image_shape=image_shape,
                                                                  target_device=args.target_device,
                                                                  checkpoint_dir=args.model_path,
                                                                  n_processes=args.n_processes,
                                                                  batch_size=args.batch_size,
                                                                  save_every=args.save_every,
                                                                  show_every=args.show_every,
                                                                  show_grid_size=args.show_grid_size)
    langevin_model_with_loop.train(args.n_epochs)


if __name__ == "__main__":
    main()
