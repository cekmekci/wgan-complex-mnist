import torch
from models import Generator
from dataloaders import get_complex_mnist_dataloaders
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import imageio
from sampler import MALA_Sampler
import random
from utils_ptycho import ptycho_forward_op, ptycho_adjoint_op, cartesian_scan_pattern
import bz2
from scipy.ndimage import zoom


# Fix the random seed
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Obtain the probe
print('Loading the probe...')
probe_amplitude = 200
probe_shape = (8, 8)
with bz2.open('./probes/siemens-star-small.npz.bz2') as f:
    archive = np.load(f)
    probe = archive['probe'][0]
    # reshape the probe
    probe = np.squeeze(probe, (0,1,2))
    probe = zoom(probe, (probe_shape[0]/probe.shape[0], probe_shape[1]/probe.shape[1]), order=3)
    # adjust the amplitude
    probe = probe_amplitude * probe
    # adjust the shape so that it is (1,2,H2,W2)
    probe = np.expand_dims(probe, 0)
    probe = np.stack((np.real(probe), np.imag(probe)), 1)
    probe = torch.from_numpy(probe).float()

# Obtain the scan
print('Creating the scan pattern...')
object_size = (64, 64)
scan = cartesian_scan_pattern(object_size, probe.shape, step_size = 4, sigma = 0.25)

# Obtain a test image
_, test_dataloader = get_complex_mnist_dataloaders(batch_size=64)
gt = next(iter(test_dataloader))[0][:1,:,:,:]

# Obtain the forward operator and its adjoint
A = lambda x: ptycho_forward_op(x, scan, probe)
AH = lambda x: ptycho_adjoint_op(x, scan, probe, object_size)

# Obtain the diffraction patterns
print('Obtaining the difraction patterns...')
farplane = A(gt) # (1,S,2,H2,W2)
intensity = np.square(np.abs(farplane))
intensity = np.random.poisson(intensity)

# Obtain the generator
latent_dim = 100
generator = Generator(img_size=(2, object_size[0], object_size[1]), latent_dim=latent_dim, dim=128)
if torch.cuda.is_available():
    generator.load_state_dict(torch.load("./pretrained/gen_mnist_model.pt"))
    generator = generator.cuda()
else:
    generator.load_state_dict(torch.load("./pretrained/gen_mnist_model.pt", map_location=torch.device('cpu')))
generator.eval()

# Initialize z randomly. See Unser's paper for a better initialization technique.
z_init = torch.randn(1, latent_dim)

# Create the MALA sampler
mala_sampler = MALA_Sampler(A, AH, generator, measurement, z_init,
                num_iter=1000, step_size=1e-5, burn_in=100, use_cuda=torch.cuda.is_available())

# Generate samples
x_samples = mala_sampler.generate_samples()

# Save the samples as a gif
x_samples_uint = [(image[0,:,:] * 255).astype(np.uint8) for image in x_samples]
imageio.mimsave('./inpainting_samples.gif', x_samples_uint)
