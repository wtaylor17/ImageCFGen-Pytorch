import torch
import torch.nn as nn
import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions.conditional import ConditionalTransform
from tqdm import tqdm

from .training_utils import batchify, init_weights


class VAEEncoder(nn.Module):
    def __init__(self, parent_dim=13):
        super().__init__()
        self.upstream_layers = nn.Sequential(
            nn.Conv2d(1, 64, (4, 4), (2, 2), 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, (4, 4), (2, 2), 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.1)
        )
        self.mean_linear = nn.Linear(100 + parent_dim, 16)
        self.log_var_linear = nn.Linear(100 + parent_dim, 16)

    def forward(self, x, c):
        upstream_e = self.upstream_layers(x)
        features = torch.concat([upstream_e, c], dim=-1)
        return self.mean_linear(features), self.log_var_linear(features)

    def sample(self, x, c, device='cpu'):
        mean, log_var = self(x, c)
        var = torch.exp(log_var)
        return mean + torch.randn(mean.shape).to(device) * var


class VAEDecoder(nn.Module):
    def __init__(self, parent_dim=13):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16 + parent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 7 * 7),
            nn.BatchNorm1d(64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, (4, 4), (2, 2), (1, 1))
        )

    def forward(self, z, c=None):
        if c is not None:
            features = torch.concat([z, c], dim=-1)
        else:
            features = z
        return self.layers(features)


class MNISTDecoderTransformation(ConditionalTransform):
    def __init__(self, decoder: nn.Module, log_var=-5,
                 device='cpu'):
        self.decoder = decoder
        self.scale = torch.exp(torch.ones((28 * 28,)) * log_var).to(device)

    def condition(self, context):
        bias = self.decoder(context).reshape((-1, 28 * 28))
        return T.AffineTransform(bias, self.scale)


class MorphoMNISTVAE(nn.Module):
    def __init__(self, parent_dim=13, device='cpu'):
        super().__init__()
        self.encoder = VAEEncoder(parent_dim).to(device)
        self.decoder = VAEDecoder(parent_dim).to(device)
        self.base = dist.MultivariateNormal(torch.zeros((28 * 28,)).to(device),
                                            covariance_matrix=torch.eye(28 * 28).to(device))
        self.dec_transform = MNISTDecoderTransformation(self.decoder,
                                                        device=device)

        alpha = 0.05
        num_bits = 8
        a1 = T.AffineTransform(0., (1. / 2 ** num_bits))

        # Map into unconstrained space as done in RealNVP
        a2 = T.AffineTransform(alpha, (1 - alpha))

        s = T.SigmoidTransform()
        preprocess_transform = T.ComposeTransform([a1, a2, s.inv])

        self.dist = dist.ConditionalTransformedDistribution(self.base,
                                                            [self.dec_transform,
                                                             preprocess_transform.inv])

    def forward(self, x, c, num_samples=10):
        return self.elbo(x, c, num_samples=num_samples)

    def elbo(self, x, c, num_samples=10, device='cpu'):
        z_mean, z_log_var = self.encoder(x, c)
        z_std = torch.exp(z_log_var * .5)
        lp = 0
        x_reshaped = x.reshape((-1, 28 * 28))
        for _ in range(num_samples):
            z = z_mean + torch.randn(z_mean.shape).to(device) * z_std
            z_c = torch.concat([z, c], dim=1)
            lp = lp + self.dist.condition(z_c).log_prob(x_reshaped)
        lp = lp / num_samples
        dkl = .5 * (torch.square(z_std) +
                    torch.square(z_mean) -
                    1 - 2 * torch.log(z_std)).sum(dim=1)
        return (lp - dkl).mean()


def train(x_train: torch.Tensor,
          a_train: torch.Tensor,
          x_test=None,
          a_test=None,
          scale_a_after=10,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          save_images_every=1,
          image_output_path='.',
          num_samples_per_step=4):
    vae = MorphoMNISTVAE(device=device)
    vae.encoder.apply(init_weights)
    vae.decoder.apply(init_weights)
    optimizer = torch.optim.Adam(vae.parameters(),
                                 lr=l_rate)

    for epoch in range(n_epochs):
        epoch_elbo = 0
        vae.train()

        num_batches = 0
        for i, (images, attrs) in tqdm(list(enumerate(batchify(x_train, a_train)))):
            num_batches += 1
            images = images.reshape((-1, 1, 28, 28)).float().to(device) / 255
            c = torch.clone(attrs.reshape((-1, 13))).float().to(device)
            c_min, c_max = a_train[:, scale_a_after:].min(dim=0).values, a_train[:, scale_a_after:].max(dim=0).values
            c[:, scale_a_after:] = (c[:, scale_a_after:] - c_min) / (c_max - c_min)

            optimizer.zero_grad()
            elbo_loss = -vae.elbo(images, c,
                                  num_samples=num_samples_per_step,
                                  device=device)
            elbo_loss.backward()
            optimizer.step()
            epoch_elbo = epoch_elbo + elbo_loss.item()

        print(epoch_elbo / num_batches)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            n_show = 10
            vae.eval()

            with torch.no_grad():
                # generate images from same class as real ones
                xdemo = x_test[:n_show]
                ademo = a_test[:n_show]
                x = xdemo.reshape((-1, 1, 28, 28)).float().to(device) / 255
                c = torch.clone(ademo.reshape((-1, 13))).float().to(device)
                c_min, c_max = a_train[:, scale_a_after:].min(dim=0).values, a_train[:, scale_a_after:].max(dim=0).values
                c[:, scale_a_after:] = (c[:, scale_a_after:] - c_min) / (c_max - c_min)

                z_mean = torch.zeros((len(x), 16)).float()

                gener = 0
                for _ in range(32):
                    z = torch.normal(z_mean, z_mean + 1).to(device)
                    context = torch.concat([z, c], dim=1)
                    gener = gener + vae.dist.condition(context).sample()
                gener = gener.cpu().detach().numpy().reshape((n_show, 28, 28)) / 32

                recon = 0
                for i in range(32):
                    z = vae.encoder.sample(x, c, device)
                    context = torch.concat([z, c], dim=1)
                    recon = recon + vae.dist.condition(context).sample()

                recon = recon.cpu().detach().numpy().reshape((n_show, 28, 28)) / 32

                real = xdemo.cpu().numpy()

                if save_images_every is not None:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
                    fig.subplots_adjust(wspace=0.05, hspace=0)
                    plt.rcParams.update({'font.size': 20})
                    fig.suptitle('Epoch {}'.format(epoch + 1))
                    fig.text(0.01, 0.75, 'G(z, c)', ha='left')
                    fig.text(0.01, 0.5, 'x', ha='left')
                    fig.text(0.01, 0.25, 'G(E(x, c), c)', ha='left')

                    for i in range(n_show):
                        ax[0, i].imshow(gener[i], cmap='gray', vmin=0, vmax=1)
                        ax[0, i].axis('off')
                        ax[1, i].imshow(real[i], cmap='gray', vmin=0, vmax=255)
                        ax[1, i].axis('off')
                        ax[2, i].imshow(recon[i], cmap='gray', vmin=0, vmax=1)
                        ax[2, i].axis('off')
                    plt.savefig(f'{image_output_path}/epoch-{epoch + 1}.png')
                    plt.close()

    return vae, optimizer
