import numpy as np
import torch
import torch.utils.data
import torch.utils.tensorboard
from tqdm import tqdm
import logging

from unet import unet_model
import remove_target_dataset
import perturbed_dataset
import checkpointer
import generate

logger = logging.getLogger(__name__)


class LangevinCNN(object):
    def __init__(self, n_channels,
                 sigmas: np.array, images_dataset,
                 image_shape: tuple,
                 checkpoint_dir, target_device='cpu',
                 n_processes=0,
                 batch_size=4,
                 save_every=1,
                 show_every=1,
                 show_grid_size=8,
                 ):
        self.target_device = torch.device(target_device)
        self.save_every = save_every
        self.show_every = show_every
        self.show_grid_size = show_grid_size
        assert sigmas.ndim == 1
        self.sigmas = torch.tensor(sigmas).float()
        self.n_sigmas = sigmas.shape[0]
        self.image_shape = (n_channels, ) + image_shape

        images_dataset_no_target = remove_target_dataset.DatasetWithoutTarget(images_dataset)
        # images_perturbed_dataset = perturbed_dataset.PerturbedDataset(images_dataset_no_target, sigmas)
        self.dataloader = torch.utils.data.DataLoader(
            images_dataset_no_target,
            batch_size=batch_size, shuffle=True,
            num_workers=n_processes,
            pin_memory=self.target_device.type == 'cuda',
        )

        self.model = unet_model.UNet(
            num_sigmas=self.n_sigmas,
            n_classes=n_channels,
            feature_levels_num=4,
            input_ch_size=n_channels,
            filters_increase_factor=2,
            hidden_ch_size=64,
            max_hidden_size=512,
            block_depth=1,
            output_block_depth=2)
        self.model = self.model.to(self.target_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Page 15 of paper
        self.lambda_sigma_pow = 2

        self.checkpointer = checkpointer.Checkpointer(checkpoint_dir)
        self.start_epoch, self.start_niter, tensorboard_dir = self.restore_with_checkpointer()

        logger.info('Using {} as TensorBoard directory'.format(tensorboard_dir))
        self.summary_writer = torch.utils.tensorboard.SummaryWriter(tensorboard_dir)

    def restore_with_checkpointer(self):
        checkpoint_file = self.checkpointer.get_latest_checkpoint_file()
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file, map_location=self.target_device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            return checkpoint['epoch'], checkpoint['niter'], checkpoint['tensorboard_dir']
        else:
            return 1, 0, None

    def save_with_checkpointer(self, epoch, niter):
        checkpoint_file = self.checkpointer.get_checkpoint_file(epoch)
        checkpoint = {
            'epoch': epoch,
            'niter': niter,
            'tensorboard_dir': self.summary_writer.log_dir,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        logger.info('Saving checkpoint {}: {}'.format(checkpoint_file, checkpoint))
        torch.save(checkpoint, checkpoint_file)
        self.checkpointer.checkpoint_saved(epoch)

    def denoising_score_matching_loss(self, images, noisy_images, sigmas, predicted_grad):
        grad_q = -(noisy_images - images) / sigmas.view(-1, 1, 1, 1)
        square_error = (predicted_grad - grad_q) ** 2
        loss = torch.sum(square_error, dim=[1, 2, 3]) / 2.0  # Keep batch dimension for sigma multiplication

        lambda_sigma = sigmas ** self.lambda_sigma_pow
        loss_times_lambda_sigma = loss * lambda_sigma
        total_loss = torch.mean(loss_times_lambda_sigma)
        return total_loss

    def generate_and_show_images(self, step):
        image_to_show = generate.generate_MNIST_anneal(self.model, self.sigmas, self.show_grid_size,
                                                       image_shape=self.image_shape,
                                                       device=self.target_device)
        self.summary_writer.add_image('generated_annealed_image', image_to_show, step)

    def train(self, n_epochs=1):
        num_epochs_iter = len(self.dataloader.dataset) // self.dataloader.batch_size
        niter = self.start_niter
        for epoch in range(self.start_epoch, n_epochs + 1):
            self.model.train()
            denoising_losses = []
            for i, (images) in enumerate(tqdm(
                    self.dataloader, desc='Training epoch {}/{}'.format(epoch, n_epochs), total=num_epochs_iter), 1):
                sigmas_idx = torch.randint(low=0, high=self.n_sigmas, size=(images.shape[0],))
                sigmas = torch.gather(self.sigmas, index=sigmas_idx, dim=0)

                images_noise = torch.randn_like(images) * sigmas.view(-1, 1, 1, 1)
                noisy_images = images + images_noise

                images = images.to(self.target_device)
                noisy_images = noisy_images.to(self.target_device)
                sigmas_idx = sigmas_idx.to(self.target_device)
                sigmas = sigmas.to(self.target_device)

                predicted_grad = self.model(noisy_images, sigmas_idx)

                self.optimizer.zero_grad()
                denoising_loss = self.denoising_score_matching_loss(images, noisy_images, sigmas, predicted_grad)
                denoising_loss.backward()
                self.optimizer.step()

                denoising_loss = denoising_loss.detach().item()
                self.summary_writer.add_scalars('denoising_loss_iter', {'iter': denoising_loss}, niter)
                denoising_losses.append(denoising_loss)

                niter += 1
            num_epochs_iter = i + 1

            mean_denoising_loss = np.mean(denoising_losses)
            self.summary_writer.add_scalars('denoising_loss_iter', {'epoch': mean_denoising_loss}, niter)
            self.summary_writer.add_scalar('denoising_loss_epoch', mean_denoising_loss, epoch)

            if epoch % self.save_every == 0:
                self.save_with_checkpointer(epoch + 1, niter)
            if epoch % self.show_every == 0:
                self.generate_and_show_images(epoch + 1)

        logger.info('Showing last images')
        self.generate_and_show_images(n_epochs + 1)

        self.summary_writer.close()
