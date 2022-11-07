from argparse import ArgumentParser
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deepscm_vae import mnist

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--model-file',
                    type=str,
                    help='file (.tar) for saved cnn models')
parser.add_argument('--latent-dim',
                    type=int,
                    default=16)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()
    latent_dim = args.latent_dim

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

    vae = mnist.MorphoMNISTVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(args.model_file,
                                   map_location=device)['vae_state_dict'])
    vae = vae.to(device)
    vae.eval()

    n_show = 10
    inds = np.random.permutation(len(x_test))[:n_show]

    with torch.no_grad():
        # generate images from same class as real ones
        xdemo = x_test[inds]
        ademo = a_test[inds]
        x = xdemo.reshape((-1, 1, 28, 28)).to(device) / 255.0
        c = ademo.reshape((-1, 13))
        cr = ademo.reshape((-1, 13)).cpu().numpy()
        c_min, c_max = a_train[:, 10:].min(dim=0).values, a_train[:, 10:].max(dim=0).values
        c[:, 10:] = (c[:, 10:] - c_min) / (c_max - c_min)

        z_mean = torch.zeros((len(x), latent_dim)).float()
        c = c.to(device)
        gener = 0
        recon = 0
        for _ in range(32):
            z = torch.normal(z_mean, z_mean + 1).to(device)
            gener = gener + vae.decoder(z, c)
            recon = recon + vae.decoder(vae.encoder.sample(x, c, device), c)
        gener = gener.reshape(n_show, 28, 28).cpu().numpy() / 32
        recon = recon.reshape(n_show, 28, 28).cpu().numpy() / 32
        real = xdemo.reshape((n_show, 28, 28)).cpu().numpy() / 255.0

        fig, ax = plt.subplots(3, n_show, figsize=(16, 5))
        fig.subplots_adjust(wspace=0.05, hspace=0)
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('Training complete')
        fig.text(0.03, 0.75, r'Sampled', ha='left')
        fig.text(0.05, 0.5, 'Real', ha='left')
        fig.text(0.01, 0.25, r'Reconstructed', ha='left')

        for i in range(n_show):
            ax[0, i].imshow(gener[i], cmap='gray', vmin=0, vmax=1)
            ax[0, i].set_title(
                f'c = {cr[i, :10].argmax()}, t = {round(float(cr[i, 10]), 2)}\ni'
                f' = {round(float(cr[i, 11]), 2)}, s = {round(float(cr[i, 12]), 2)}',
                fontsize=8)
            ax[0, i].axis('off')
            ax[1, i].imshow(real[i], cmap='gray', vmin=0, vmax=1)
            ax[1, i].axis('off')
            ax[2, i].imshow(recon[i], cmap='gray', vmin=0, vmax=1)
            ax[2, i].axis('off')
        plt.savefig('mnist-vae-eval.png')
        plt.close()
