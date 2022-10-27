from argparse import ArgumentParser
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from image_scms import mnist

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--model-file',
                    type=str,
                    help='file (.tar) for saved cnn models')

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

    E, G, D = mnist.load_model(args.model_file,
                               device=device)
    E = E.to(device)
    G = G.to(device)
    D = D.to(device)

    n_show = 10
    inds = np.random.permutation(len(x_test))[:n_show]
    D.eval()
    E.eval()
    G.eval()

    with torch.no_grad():
        # generate images from same class as real ones
        xdemo = x_test[inds]
        ademo = a_test[inds]
        x = xdemo.reshape((-1, 1, 28, 28)).to(device) / 255.0
        c = ademo.reshape((-1, 13))
        cr = ademo.reshape((-1, 13)).cpu().numpy()
        c_min, c_max = c[:, 10:].min(dim=0).values, c[:, 10:].max(dim=0).values
        c[:, 10:] = (c[:, 10:] - c_min) / (c_max - c_min)
        z_mean = torch.zeros((len(x), 512, 1, 1)).float()
        z = torch.normal(z_mean, z_mean + 1)
        z = z.to(device)
        c = c.to(device)

        gener = G(z, c).reshape(n_show, 28, 28).cpu().numpy()
        recon = G(E(x, c), c).reshape(n_show, 28, 28).cpu().numpy()
        real = xdemo.reshape((n_show, 28, 28)).cpu().numpy() / 255.0

        fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.05, hspace=0)
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('Training complete')
        fig.text(0.01, 0.75, 'G(z, c)', ha='left')
        fig.text(0.01, 0.5, 'x', ha='left')
        fig.text(0.01, 0.25, 'G(E(x, c), c)', ha='left')

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
        plt.savefig('mnist-imagecfgen-eval.png')
        plt.close()
