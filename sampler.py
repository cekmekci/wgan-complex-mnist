import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class MALA_Poisson_Sampler():

    def __init__(self, A, AT, generator, measurement, z_init,
                    num_iter=10000, step_size=1e-5, burn_in=500, use_cuda=False):
        self.A = A
        self.AT = AT
        self.generator = generator
        self.measurement = measurement
        self.z_init = z_init
        self.num_iter = num_iter
        self.step_size = step_size
        self.burn_in = burn_in
        self.use_cuda = use_cuda
        self.eps = 1e-5
        if self.use_cuda:
            self.generator = self.generator.cuda()
            self.z_init = self.z_init.cuda()
            self.measurement = self.measurement.cuda()

    def log_p_z_given_y(self, z):
        # ignoring logp(y)
        AGz = self.A(self.generator(z))
        AGz_squared_sum = torch.sum(AGz**2, 2, keepdim=True)
        log_likelihood = torch.sum(self.measurement * torch.log(AGz_squared_sum + self.eps) - AGz_squared_sum)
        log_prior = -1 / 2 * torch.sum(z**2)
        return log_likelihood + log_prior

    def grad_log_p_z_given_y(self, z):
        # gradient of log_p_y_given_z
        AGz = self.A(self.generator(z)) # (1,S,2,H2,W2)
        vec = self.AT(AGz * (self.measurement / (torch.sum(AGz**2,2,keepdim=True) + self.eps)  - 1)) # (1,2,64,64)
        _, vjp = torch.autograd.functional.vjp(self.generator, z, v=vec, create_graph=False, strict=True) # (1, latent_dim)
        grad = 2 * vjp
        # add the gradient of log_p_z, which is -z
        grad = grad - z
        return grad

    def log_q_zbar_given_ztilde(self, z_bar, z_tilde):
        grad = self.grad_log_p_z_given_y(z_tilde)
        return -0.25 / self.step_size * torch.sum((z_bar - z_tilde - self.step_size * grad)**2)

    def sample(self):
        z = self.z_init
        z_samples = []
        for k in range(self.num_iter):
            # Compute the gradient of log p(z|y)
            grad = self.grad_log_p_z_given_y(z)
            # Determine the candidate
            z_new = z + self.step_size * grad + (2 * self.step_size)**0.5 * torch.randn_like(z)
            # Determine the transition probability
            term_1 = self.log_p_z_given_y(z_new)
            term_2 = self.log_q_zbar_given_ztilde(z, z_new)
            term_3 = self.log_p_z_given_y(z)
            term_4 = self.log_q_zbar_given_ztilde(z_new, z)
            log_prob = term_1 + term_2 - term_3 - term_4
            prob = torch.exp(log_prob)
            prob = torch.clamp(prob, min=0.0, max=1.0)
            print(prob)
            # Transition
            if np.random.uniform() < prob.item():
                z = z_new
                # If we pass the burn-in period, collect the sample.
                if k > self.burn_in:
                    z_samples.append(z.detach().cpu())
        return z_samples

    def generate_samples(self):
        z_samples = self.sample()
        x_samples = []
        for z_sample in z_samples:
            if self.use_cuda:
                z_sample = z_sample.cuda()
            x_sample = self.generator(z_sample).detach().cpu().numpy()[0,:,:,:]
            x_samples.append(x_sample)
        return x_samples


def optimize_latent_variable(G, x, z_dim, lr=1e-4, num_steps=1000, verbose=False):
    """
    Finds the latent variable z that best approximates the given image x.

    Args:
        G (nn.Module): Generative model.
        x (torch.Tensor): Target image tensor.
        z_dim (int): Dimension of the latent space.
        lr (float): Learning rate for the optimizer.
        num_steps (int): Number of optimization steps.

    Returns:
        torch.Tensor: Optimized latent variable.
    """
    # Ensure x is on the same device as G
    device = next(G.parameters()).device
    x = x.to(device)
    # Initialize the latent variable z with requires_grad=True
    z = torch.randn(1, z_dim, device=device, requires_grad=True)
    # Define the optimizer
    optimizer = optim.Adam([z], lr=lr)
    # Define the loss function
    loss_fn = nn.MSELoss()
    for step in range(num_steps):
        optimizer.zero_grad()
        # Generate image from z
        generated_image = G(z)
        # Compute the loss
        loss = loss_fn(generated_image, x)
        # Backpropagate the loss
        loss.backward()
        # Update z
        optimizer.step()
        if verbose and (step % 100 == 0):
            print(f'Step {step}/{num_steps}, Loss: {loss.item()}')
    return z.detach()


# Numerically test if the gradient calculation is correct
if __name__ == '__main__':

    from utils_ptycho import ptycho_forward_op, ptycho_adjoint_op, cartesian_scan_pattern
    from models import Generator
    import matplotlib.pyplot as plt

    object_size = (64,64)
    probe_size = (1,2,16,16)
    latent_dim = 100

    complex_object = torch.randn(1,2,*object_size)
    scan = cartesian_scan_pattern(object_size, probe_size, step_size = 4, sigma = 0.1)
    probe = torch.randn(probe_size)

    A = lambda x: ptycho_forward_op(x, scan, probe)
    AH = lambda x: ptycho_adjoint_op(x, scan, probe, object_size)

    measurement = A(complex_object)
    measurement = torch.sum(measurement**2, 2, keepdim=True) # (1,S,1,H2,W2)
    measurement = torch.poisson(measurement)

    z = torch.randn((1,latent_dim))

    generator = Generator(img_size=(2, object_size[0], object_size[1]), latent_dim=latent_dim, dim=128)
    generator.eval()

    sampler = MALA_Poisson_Sampler(A, AH, generator, measurement, z,
                                    num_iter=10000, step_size=1e-5, burn_in=500,
                                    use_cuda=False)

    # Our closed form gradient
    grad = sampler.grad_log_p_z_given_y(z)

    errors = []
    eps_list = np.logspace(-3, 2, 25)
    for eps in eps_list:
        # Numerical gradient
        numerical_grad = torch.zeros_like(grad)
        for i in range(latent_dim):
            e_i = torch.zeros_like(z)
            e_i[:,i] = 1.0
            z_plus = z + eps * e_i
            z_minus = z - eps * e_i
            num_grad = (sampler.log_p_z_given_y(z_plus) - sampler.log_p_z_given_y(z_minus)) / (2*eps)
            numerical_grad[:,i] = num_grad
        # Check the difference
        error = torch.mean(torch.abs(grad - numerical_grad))
        errors.append(error)

    plt.figure(figsize=(10, 6))
    plt.plot(eps_list, errors, marker='o', color='white', linestyle='-', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')

    # Set plot style
    ax = plt.gca()
    fig = plt.gcf()

    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')  # Set the figure background to black
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    # Set grid style
    plt.grid(color='white', linestyle='--', linewidth=0.5)

    # Set axis color
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')

    # Also, set the color of the ticks and labels
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Set labels and title
    plt.xlabel('Eps Values')
    plt.ylabel('Gradient Error')
    plt.title('Gradient Error vs. Perturbation Parameter (Eps)')

    # Show plot
    plt.savefig("gradient_errors.png")
