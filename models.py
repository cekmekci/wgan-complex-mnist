import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):

    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size # in (C,H,W) format
        self.img_size_vectorized = np.prod(self.img_size)
        self.latent_to_image = nn.Sequential(
            # layer 1
            nn.Linear(latent_dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 2
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 3
            nn.Linear(dim * 2, dim * 4),
            nn.BatchNorm1d(dim * 4),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 4
            nn.Linear(dim * 4, dim * 8),
            nn.BatchNorm1d(dim * 8),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 5
            nn.Linear(dim * 8, self.img_size_vectorized),
            nn.Sigmoid()
        )

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

    def forward(self, input_data):
        # Map latent to image
        x = self.latent_to_image(input_data)
        # Reshape
        x = x.view(-1, self.img_size[0] , self.img_size[1], self.img_size[2])
        # Return generated image
        return x


class Discriminator(nn.Module):

    def __init__(self, img_size, dim):
        super(Discriminator, self).__init__()
        self.img_size = img_size # in (C,H,W) format
        self.img_size_vectorized = np.prod(self.img_size)
        self.image_to_scalar = nn.Sequential(
            # layer 1
            nn.Linear(self.img_size_vectorized, dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 2
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 3
            nn.Linear(dim, 1)
        )

    def forward(self, input_data):
        batch_size = input_data.shape[0]
        x = input_data.view(batch_size, -1)
        x = self.image_to_scalar(x)
        return x
