import torch
from models import Generator
from dataloaders import get_complex_mnist_dataloaders
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import imageio
from sampler import MALA_Poisson_Sampler, optimize_latent_variable
import random
from utils_ptycho import ptycho_forward_op, ptycho_adjoint_op, cartesian_scan_pattern, rPIE, create_disk_probe
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
probe_amplitude = 100
probe_shape = (16, 16)
probe = create_disk_probe(size = probe_shape, width=8.0, magnitude = probe_amplitude)
probe = np.expand_dims(probe, 0)
probe = np.stack((np.real(probe), np.imag(probe)), 1)
probe = torch.from_numpy(probe).float()

# Obtain the test image
print('Generating the test image...')
object_size = (64, 64)
_, test_dataloader = get_complex_mnist_dataloaders(batch_size=1, image_size=object_size[0])
gt = next(iter(test_dataloader))[0][:1,:,:,:]
gt_real = gt[:,0:1,:,:] * torch.cos(gt[:,1:2,:,:])
gt_imag = gt[:,0:1,:,:] * torch.sin(gt[:,1:2,:,:])
gt = torch.cat((gt_real, gt_imag), 1)

# Obtain the scan
print('Creating the scan pattern...')
scan = cartesian_scan_pattern(object_size, probe.shape, step_size = 2, sigma = 0.1)

# Obtain the forward operator and its adjoint
A = lambda x: ptycho_forward_op(x, scan, probe)
AH = lambda x: ptycho_adjoint_op(x, scan, probe, object_size)

# Obtain the diffraction patterns
print('Obtaining the difraction patterns...')
farplane = A(gt) # (1,S,2,H2,W2)
intensity = torch.sum(farplane**2, 2, keepdim=True) # (1,S,1,H2,W2)
intensity = torch.poisson(intensity)
print("Min intensity:", intensity.min(), "Maximum intensity:", intensity.max())

# Obtain the generator
latent_dim = 100
generator = Generator(img_size=(2, object_size[0], object_size[1]), latent_dim=latent_dim, dim=128)
if torch.cuda.is_available():
    generator.load_state_dict(torch.load("./pretrained/gen_mnist_model.pt"))
    generator = generator.cuda()
else:
    generator.load_state_dict(torch.load("./pretrained/gen_mnist_model.pt", map_location=torch.device('cpu')))
generator.eval()

# Initialize the latent variable
rpie_rec = rPIE(intensity, object_size, scan, probe)
z_init = optimize_latent_variable(generator, rpie_rec, z_dim=latent_dim, lr=1e-4, num_steps=1000, verbose=False)

# Create the MALA sampler
mala_sampler = MALA_Poisson_Sampler(A, AH, generator, intensity, z_init,
                num_iter=1000, step_size=1e-5, burn_in=100, use_cuda=torch.cuda.is_available())

# Generate samples
x_samples = mala_sampler.generate_samples()

# Save the samples as a gif
x_samples_uint = [(image[0,:,:] * 255).astype(np.uint8) for image in x_samples]
imageio.mimsave('./inpainting_samples.gif', x_samples_uint)
