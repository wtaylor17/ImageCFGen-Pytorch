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
parser.add_argument('--steps', type=int,
                    help='number of epochs to train the distributions',
                    default=2000)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()

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

    vae, optimizer = mnist.train(x_train,
                                 a_train,
                                 x_test=x_test,
                                 a_test=a_test,
                                 scale_a_after=10,
                                 n_epochs=args.steps,
                                 device=device)
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'models_state_dict_VAE_MNIST_tis.tar')

    n_show = 10
    inds = np.random.permutation(len(x_test))[:n_show]
    vae.eval()

    with torch.no_grad():
        # generate images from same class as real ones
        xdemo = x_test[inds]
        ademo = a_test[inds]
        x = xdemo.reshape((-1, 1, 28, 28)).to(device)
        c = ademo.reshape((-1, 13)).to(device)
        cr = ademo.reshape((-1, 13)).cpu().numpy()
        c[:, 10] = (c[:, 10] - c[:, 10].min()) / (c[:, 10].max() - c[:, 10].min())
        c[:, 11] = (c[:, 11] - c[:, 11].min()) / (c[:, 11].max() - c[:, 11].min())
        c[:, 12] = (c[:, 12] - c[:, 12].min()) / (c[:, 12].max() - c[:, 12].min())
        z_mean = torch.zeros((len(x), 512, 1, 1)).float()
        z = torch.normal(z_mean, z_mean + 1)
        z = z.to(device)

        gener = vae.decoder(z, c).reshape(n_show, 28, 28).cpu().numpy()
        recon = vae.decoder(vae.encoder(x, c), c).reshape(n_show, 28, 28).cpu().numpy()
        real = xdemo

        fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.05, hspace=0)
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('Training complete')
        fig.text(0.01, 0.75, 'G(z, c)', ha='left')
        fig.text(0.01, 0.5, 'x', ha='left')
        fig.text(0.01, 0.25, 'G(E(x, c), c)', ha='left')

        for i in range(n_show):
            ax[0, i].imshow(gener[i], cmap='gray')
            ax[0, i].set_title(
                f'c = {cr[i, :10].argmax()}, t = {round(float(cr[i, 10]), 2)}\ni'
                f' = {round(float(cr[i, 11]), 2)}, s = {round(float(cr[i, 12]), 2)}',
                fontsize=8)
            ax[0, i].axis('off')
            ax[1, i].imshow(real[i], cmap='gray')
            ax[1, i].axis('off')
            ax[2, i].imshow(recon[i], cmap='gray')
            ax[2, i].axis('off')
        plt.show()
