import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=300,
                 use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        # Get generated data
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size) # G(z)
        # Calculate the output of the discriminator on real and generated data
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data) # D(x)
        d_generated = self.D(generated_data) # D(G(z))
        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())
        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()
        # Record loss
        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        self.G_opt.zero_grad()
        # Get generated data
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size)
        # Calculate loss and optimize
        d_generated = self.D(generated_data) # D(G(z))
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()
        # Record loss
        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.shape[0]
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated.requires_grad = True
        if self.use_cuda:
            interpolated = interpolated.cuda()
        # Feed the interpolated examples to the discriminator
        D_interpolated = self.D(interpolated)
        # Calculate gradients of the output of the discriminator with respect to examples
        gradients = torch_grad(outputs=D_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(D_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               D_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]
        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data[0])
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data[0])

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = self.G.sample_latent(64)
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images_magnitude = []
            training_progress_images_phase = []
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)
            if save_training_gif:
                # Generate batch of images and convert to grid
                images = self.G(fixed_latents).detach().cpu().data
                # magnitudes = torch.sqrt(images[:,0:1,:,:]**2 + images[:,1:2,:,:]**2)
                # phases = torch.atan2(images[:,1:2,:,:], images[:,0:1,:,:] + 1e-5)
                magnitudes = images[:,0:1,:,:]
                phases = images[:,1:2,:,:]
                img_grid_mag = make_grid(magnitudes)
                img_grid_phase = make_grid(phases)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid_mag = np.transpose(img_grid_mag.numpy(), (1, 2, 0)) * 255
                img_grid_mag = img_grid_mag.astype(np.uint8)
                img_grid_phase = np.transpose(img_grid_phase.numpy(), (1, 2, 0)) * 255
                img_grid_phase = img_grid_phase.astype(np.uint8)
                # Add image grid to training progress
                training_progress_images_magnitude.append(img_grid_mag)
                training_progress_images_phase.append(img_grid_phase)
                imageio.imwrite('./training_{}_epoch_magnitude.png'.format(epoch),
                            img_grid_mag)
                imageio.imwrite('./training_{}_epoch_phase.png'.format(epoch),
                            img_grid_phase)
        if save_training_gif:
            imageio.mimsave('./training_{}_epochs_magnitude.gif'.format(epochs),
                            training_progress_images_magnitude)
            imageio.mimsave('./training_{}_epochs_phase.gif'.format(epochs),
                            training_progress_images_phase)

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data
