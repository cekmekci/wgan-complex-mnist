import torch
import random
import numpy as np
import torch.optim as optim
from dataloaders import get_mnist_dataloaders
from models import Generator, Discriminator
from trainer import Trainer

SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_loader, _ = get_mnist_dataloaders(batch_size=64)
img_size = (1, 28, 28)

generator = Generator(img_size=img_size, latent_dim=100, dim=128)
discriminator = Discriminator(img_size=img_size, dim=256)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 2e-4
betas = (0.5, 0.999)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 2000
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

# Save models
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
