import numpy as np
import torch
import torch.utils.data
import torch.utils.tensorboard
from tqdm import tqdm
from pathlib import Path
import logging
import torchvision

from unet import unet_model
from refinenet.refinenet import RefineNet
import remove_target_dataset
import perturbed_dataset
import checkpointer
import generate
import random_dataset

logger = logging.getLogger(__name__)


class LangevinCNN(object):
    def __init__(self,
                 num_filters: int,
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
        self.image_shape = image_shape

        images_dataset_no_target = remove_target_dataset.DatasetWithoutTarget(images_dataset)
        # images_perturbed_dataset = perturbed_dataset.PerturbedDataset(images_dataset_no_target, sigmas)
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(
            images_dataset_no_target,
            batch_size=batch_size, shuffle=True,
            num_workers=n_processes,
            pin_memory=self.target_device.type == 'cuda',
        )

        self.model = RefineNet(image_shape[0], num_filters, torch.nn.ELU, num_sigmas=self.n_sigmas, block_depth=2)
        self.model = self.model.to(self.target_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Page 15 of paper
        self.lambda_sigma_pow = 2

        self.checkpoint_dir = checkpoint_dir
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
        }
        logger.info('Saving checkpoint {}: {}'.format(checkpoint_file, checkpoint))
        checkpoint.update({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        })
        torch.save(checkpoint, checkpoint_file)
        self.checkpointer.checkpoint_saved(epoch)

    def denoising_score_matching_loss(self, images, noisy_images, sigmas, predicted_grad):
        grad_q = -(noisy_images - images) / (sigmas ** 2).view(-1, 1, 1, 1)
        square_error = (predicted_grad - grad_q) ** 2
        loss = torch.sum(square_error, dim=[1, 2, 3]) / 2.0  # Keep batch dimension for sigma multiplication

        lambda_sigma = sigmas ** self.lambda_sigma_pow
        loss_times_lambda_sigma = loss * lambda_sigma
        total_loss = torch.mean(loss_times_lambda_sigma)
        return total_loss

    def generate_and_show_images(self, step):
        image_to_show, all_images = generate.generate_MNIST_anneal(self.model, self.sigmas, self.show_grid_size,
                                                                   image_shape=self.image_shape,
                                                                   device=self.target_device)
        self.summary_writer.add_image('generated_annealed_image', image_to_show, step)
        for i, image in enumerate(all_images):
            self.summary_writer.add_image('generated_annealed_image_process', image, step + i)

    def train(self, n_epochs=1):
        num_epochs_iter = len(self.dataloader.dataset) // self.dataloader.batch_size
        niter = self.start_niter
        torch.autograd.set_detect_anomaly(True)
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

    def generate_images(self, num_images):
        random_shape = (num_images,) + self.image_shape
        start_points_dataset = random_dataset.RandomDataset(random_shape, torch.rand)
        start_points_dataloader = torch.utils.data.DataLoader(
            start_points_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.target_device.type == 'cuda',
        )
        self.model.eval()
        total = num_images // self.batch_size
        current_image_i = 1

        images_path = Path(self.checkpoint_dir) / 'generated_images'
        images_path.mkdir(parents=True, exist_ok=True)
        for i, start_points_batch in tqdm(enumerate(start_points_dataloader),
                                          desc='Generating desired images',
                                          total=total):
            start_points_batch = start_points_batch.to(self.target_device)
            final_images, _ = generate.data_anneal_lavgevin(
                start_points_batch, self.model, self.sigmas, lr=5 * 1e-5, step=100, device=self.target_device)

            for image_i in range(final_images.shape[0]):
                image_i_path = images_path / "{:09}.png".format(current_image_i)
                torchvision.utils.save_image(final_images[image_i], str(image_i_path))
                current_image_i += 1

    def generate_image_generation_process_picture(self, n_images):
        start_point = torch.rand(n_images, *self.image_shape)
        start_point = start_point.to(self.target_device)

        _, generation_process = generate.data_anneal_lavgevin(
            start_point, self.model, self.sigmas, lr=5 * 1e-5, step=100, device=self.target_device)
        generation_process_stacked = torch.stack(generation_process)

        image_then_generation_process = generation_process_stacked.transpose(1, 0)
        image_then_generation_process = image_then_generation_process.reshape(
            -1, *image_then_generation_process.shape[2:])

        generation_image_grid = torchvision.utils.make_grid(
            image_then_generation_process, nrow=self.n_sigmas)

        save_path = Path(self.checkpoint_dir) / 'generation_process.png'
        torchvision.utils.save_image(generation_image_grid, str(save_path))
