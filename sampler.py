import torch
import numpy as np

class MALA_Sampler():

    def __init__(self, A, AT, noise_std, generator, measurement, z_init,
                    num_iter=10000, step_size=1e-5, burn_in=500, use_cuda=False):
        self.A = A
        self.AT = AT
        self.noise_std = noise_std
        self.generator = generator
        self.measurement = measurement
        self.z_init = z_init
        self.num_iter = num_iter
        self.step_size = step_size
        self.burn_in = burn_in
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.generator = self.generator.cuda()
            self.z_init = self.z_init.cuda()
            self.measurement = self.measurement.cuda()

    def log_p_z_given_y(self, z):
        # ignoring logp(y)
        return -0.5 / (self.noise_std**2) * torch.sum((self.A(self.generator(z)) - self.measurement)**2) - 0.5 * torch.sum(z**2)

    def grad_log_p_z_given_y(self, z):
        vec = self.AT(self.A(self.generator(z)) - self.measurement)
        _, vjp = torch.autograd.functional.vjp(self.generator, z, v=vec, create_graph=False, strict=True)
        grad = -1 / (self.noise_std**2) * vjp - z
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
