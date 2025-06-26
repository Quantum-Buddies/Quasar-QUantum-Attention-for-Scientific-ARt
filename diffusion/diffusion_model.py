"""quantum_transformers.diffusion.diffusion_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A PyTorch Lightning wrapper for training and sampling from a
denoising diffusion model. This module handles the noising (forward process),
denoising (reverse process), and loss calculation.

It is designed to work with any denoiser network, particularly our
UDiT and hybrid Q-UDiT models.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        denoiser_model: nn.Module,
        timesteps: int = 1000,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['denoiser_model'])
        self.model = denoiser_model
        self.timesteps = timesteps
        self.learning_rate = learning_rate

        # --- Setup Noise Schedule (DDPM) ---
        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))


    def forward_process(self, x0, t, noise=None):
        """
        Adds noise to an image x0 at timestep t.
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1-alpha_cumprod_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        noisy_image = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_image

    def training_step(self, batch, batch_idx):
        x0, y = batch
        batch_size = x0.shape[0]

        # 1. Sample a random timestep for each image in the batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

        # 2. Generate noise and create the noisy image
        noise = torch.randn_like(x0)
        xt = self.forward_process(x0, t, noise)

        # 3. Predict the noise using the model
        predicted_noise = self.model(xt, t, y)

        # 4. Calculate loss
        loss = F.mse_loss(noise, predicted_noise)
        
        self.log('train_loss', loss)
        return loss

    @torch.no_grad()
    def sample(self, num_samples, image_size, y=None):
        """
        The reverse (denoising) process. Starts from pure noise and generates samples.
        """
        self.model.eval()
        img = torch.randn((num_samples, self.model.in_channels, image_size, image_size), device=self.device)
        
        if y is None:
            # Unconditional generation
            y = torch.randint(0, self.model.num_classes, (num_samples,), device=self.device)

        for t in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            predicted_noise = self.model(img, t_tensor, y)
            
            # Denoising step formula (from DDPM paper)
            alpha_t = 1. - self.betas[t]
            alphas_cumprod_t = self.alphas_cumprod[t]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            
            # Predict x0 from xt and predicted noise
            x0_pred = (img - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / torch.sqrt(alphas_cumprod_t)
            x0_pred.clamp_(-1., 1.)

            # Compute mean of the posterior q(x_{t-1} | x_t, x_0)
            posterior_mean = (self.betas[t] * torch.sqrt(self.alphas_cumprod_prev[t]) / (1. - alphas_cumprod_t)) * x0_pred + \
                             ((1. - self.alphas_cumprod_prev[t]) * torch.sqrt(alpha_t) / (1. - alphas_cumprod_t)) * img
            
            if t == 0:
                img = posterior_mean
            else:
                posterior_variance_t = self.posterior_variance[t]
                noise = torch.randn_like(img)
                img = posterior_mean + torch.sqrt(posterior_variance_t) * noise
        
        self.model.train()
        return img

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
