import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict
import numpy as np
from argparse import ArgumentParser
import os

from image_scms.training_utils import init_weights
from image_scms.training_utils import batchify, batchify_dict


class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (2, 2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (4, 4), (2, 2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (4, 4), (2, 2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (4, 4), (2, 2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, latent_dim, (1, 1), (2, 2)),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X: torch.Tensor):
        return self.layers(X)


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, (3, 3), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, (4, 4)),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor):
        return self.layers(z)


parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--steps', type=int,
                    help='number of epochs to train the distributions',
                    default=200)
parser.add_argument('--cls', type=int, default=None)
parser.add_argument('--output-path', type=str,
                    default='morphomnist_ae.tar')
parser.add_argument('--latent-dim', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=1e-4)


if __name__ == '__main__':
    args = parser.parse_args()

    cls = args.cls

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-train.npy')
    )).float().to(device)
    x_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().to(device)

    y_train = a_train[:, :10].float()
    y_test = a_test[:, :10].float()

    E = Encoder(args.latent_dim).to(device)
    G = Generator(args.latent_dim).to(device)

    if cls is not None:
        x_train = x_train[y_train.argmax(1) == cls]
        x_test = x_test[y_test.argmax(1) == cls]

    opt = torch.optim.Adam(list(E.parameters()) + list(G.parameters()),
                           lr=args.learning_rate)
    for epoch in range(args.steps):
        print(f'Epoch {epoch + 1}/{args.steps}')
        tq = tqdm(batchify(x_train, batch_size=args.batch_size))
        cur_loss = 0
        for i, (x,) in enumerate(tq):
            opt.zero_grad()
            x = 2 * x.reshape((-1, 1, 28, 28)) / 255.0 - 1
            z = E(x)
            xr = G(z)
            loss = (xr - x).square().mean()
            loss.backward()
            opt.step()
            cur_loss += loss.item()
            tq.set_postfix(mse=cur_loss / (i + 1))
        with torch.no_grad():
            xt = 2 * x_test.reshape((-1, 1, 28, 28)) / 255.0 - 1
            xr = G(E(xt))
            loss = (xr - x).square().mean().item()
            print('Test loss:', loss)

    torch.save({
        'E': E,
        'G': G
    }, args.output_path)
