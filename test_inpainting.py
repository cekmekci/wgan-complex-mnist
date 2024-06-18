import torch
from models import Generator
from dataloaders import get_mnist_dataloaders
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import imageio
from sampler import MALA_Sampler
import random

# Fix the random seed
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Forward operator
def A(x):
    # x is (1,1,H,W)
    mask = torch.ones_like(x)
    mask[:,:,:x.shape[2]//2,:] = 0.0
    out = mask * x
    return out

# Adjoint of the forward operator
def AT(x):
    return A(x)

# Generate the measurement, whose shape is (1,1,H,W)
noise_std = 0.1
_, test_dataloader = get_mnist_dataloaders(batch_size=64)
gt = next(iter(test_dataloader))[0][:1,:,:,:]
measurement = A(gt)
measurement = measurement + noise_std * torch.randn_like(measurement)
if torch.cuda.is_available():
    gt, measurement = gt.cuda(), measurement.cuda()

# Obtain the generator
generator = Generator(img_size=(1, 28, 28), latent_dim=100, dim=128)
if torch.cuda.is_available():
    generator.load_state_dict(torch.load("./pretrained/gen_mnist_model.pt"))
    generator = generator.cuda()
else:
    generator.load_state_dict(torch.load("./pretrained/gen_mnist_model.pt", map_location=torch.device('cpu')))
generator.eval()

# Initialize z randomly
z_init = torch.randn(1, 100)

# Create the MALA sampler
mala_sampler = MALA_Sampler(A, AT, noise_std, generator, measurement, z_init,
                num_iter=1000, step_size=1e-5, burn_in=100, use_cuda=torch.cuda.is_available())

# Generate samples
x_samples = mala_sampler.generate_samples()

# Save the samples as a gif
x_samples_uint = [(image[0,:,:] * 255).astype(np.uint8) for image in x_samples]
imageio.mimsave('./inpainting_samples.gif', x_samples_uint)
