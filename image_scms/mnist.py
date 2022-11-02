import torch
import torch.nn as nn
from tqdm import tqdm

from .training_utils import attributes_image
from .training_utils import AdversariallyLearnedInference
from .training_utils import init_weights
from .training_utils import batchify


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 32, (5, 5), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (1, 1), (1, 1))
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X, a):
        return self.layers(attributes_image(X, a, device=self.device))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(512 + 13),
            nn.ConvTranspose2d(512 + 13, 256, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, (4, 4), (1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, (1, 1), (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, z, a):
        a = a.reshape((-1, 13, 1, 1))
        return self.layers(torch.concat([z, a], dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(512, 512, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 512, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dx = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(2, 32, (5, 5), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 256, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 512, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dxz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1, (1, 1), (1, 1)),
            nn.Sigmoid()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X, z, a):
        dx = self.dx(attributes_image(X, a, device=self.device))
        dz = self.dz(z)
        return self.dxz(torch.concat([dx, dz], dim=1)).reshape((-1, 1))


def train(x_train: torch.Tensor,
          a_train: torch.Tensor,
          x_test=None,
          a_test=None,
          scale_a_after=10,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          save_images_every=10,
          image_output_path=''):
    E = Encoder().to(device)
    G = Generator().to(device)
    D = Discriminator().to(device)

    E.apply(init_weights)
    G.apply(init_weights)
    D.apply(init_weights)

    optimizer_E = torch.optim.Adam(list(E.parameters()) + list(G.parameters()),
                                   lr=l_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(),
                                   lr=l_rate, betas=(0.5, 0.999))

    loss_calc = AdversariallyLearnedInference(E, G, D)

    for epoch in range(n_epochs):
        D_score = 0.
        EG_score = 0.
        D.train()
        E.train()
        G.train()

        num_batches = 0
        for i, (images, attrs) in tqdm(list(enumerate(batchify(x_train, a_train)))):
            num_batches += 1
            images = images.reshape((-1, 1, 28, 28)).float().to(device) / 255
            c = torch.clone(attrs.reshape((-1, 13))).float().to(device)
            c_min, c_max = c[:, scale_a_after:].min(dim=0).values, c[:, scale_a_after:].max(dim=0).values
            c[:, scale_a_after:] = (c[:, scale_a_after:] - c_min) / (c_max - c_min)

            z_mean = torch.zeros((len(images), 512, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(device)

            # Discriminator training
            optimizer_D.zero_grad()
            loss_D = loss_calc.discriminator_loss(images, z, c)
            loss_D.backward()
            optimizer_D.step()

            # Encoder & Generator training
            optimizer_E.zero_grad()
            loss_EG = loss_calc.generator_loss(images, z, c)
            loss_EG.backward()
            optimizer_E.step()

            Gz = G(z, c).detach()
            EX = E(images, c).detach()
            DG = D(Gz, z, c)
            DE = D(images, EX, c)
            D_score += DG.mean().item()
            EG_score += DE.mean().item()

        print(D_score / num_batches, EG_score / num_batches)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            n_show = 10
            D.eval()
            E.eval()
            G.eval()

            with torch.no_grad():
                # generate images from same class as real ones
                xdemo = x_test[:n_show]
                ademo = a_test[:n_show]
                x = xdemo.reshape((-1, 1, 28, 28)).float().to(device) / 255
                c = torch.clone(ademo.reshape((-1, 13))).float().to(device)
                c_min, c_max = a_train[:, scale_a_after:].min(dim=0).values, a_train[:, scale_a_after:].max(dim=0).values
                c[:, scale_a_after:] = (c[:, scale_a_after:] - c_min) / (c_max - c_min)

                z_mean = torch.zeros((len(x), 512, 1, 1)).float()
                z = torch.normal(z_mean, z_mean + 1)
                z = z.to(device)

                gener = G(z, c).reshape(n_show, 28, 28).cpu().numpy()
                recon = G(E(x, c), c).reshape(n_show, 28, 28).cpu().numpy()
                real = xdemo

                if save_images_every is not None:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
                    fig.subplots_adjust(wspace=0.05, hspace=0)
                    plt.rcParams.update({'font.size': 20})
                    fig.suptitle('Epoch {}'.format(epoch + 1))
                    fig.text(0.04, 0.75, 'G(z, c)', ha='left')
                    fig.text(0.04, 0.5, 'x', ha='left')
                    fig.text(0.04, 0.25, 'G(E(x, c), c)', ha='left')

                    for i in range(n_show):
                        ax[0, i].imshow(gener[i], cmap='gray')
                        ax[0, i].axis('off')
                        ax[1, i].imshow(real[i], cmap='gray')
                        ax[1, i].axis('off')
                        ax[2, i].imshow(recon[i], cmap='gray')
                        ax[2, i].axis('off')
                    plt.savefig(f'{image_output_path}/epoch-{epoch + 1}.png')
                    plt.close()

    return E, G, D, optimizer_D, optimizer_E


def load_model(tar_path, device='cpu', return_raw=False):
    obj = torch.load(tar_path, map_location=device)
    E = Encoder()
    G = Generator()
    D = Discriminator()

    E.load_state_dict(obj['E_state_dict'])
    G.load_state_dict(obj['G_state_dict'])
    D.load_state_dict(obj['D_state_dict'])
    if return_raw:
        return E, G, D, obj
    return E, G, D
