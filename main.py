import argparse
import torchvision
import torch.utils.data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', default='dataset')
    parser.add_argument('--download_dataset', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = torchvision.datasets.mnist.MNIST(
        args.dataset_folder,
        train=True,
        download=args.download_dataset,
        transform=torchvision.transforms.ToTensor(),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = 

    for images, _ in dataloader:
        break


if __name__ == "__main__":
    main()
