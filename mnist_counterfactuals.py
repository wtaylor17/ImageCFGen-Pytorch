from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from image_scms import mnist
from morphomnist.morpho import ImageMorphology
from morphomnist.perturb import SetThickness, SetSlant


def sigmoid_inverse(x_):
    return torch.log(x_) - torch.log(1 - x_)


class MorphoMNISTSCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_noise_dist = torch.distributions.Gamma(10, 5)
        self.i_noise_dist = torch.distributions.Normal(0, 1)
        self.s_noise_dist = torch.distributions.Normal(0, 0.1)

    def generate_t(self, n=1, noise=None):
        if noise is None:
            et = self.t_noise_dist.sample((n, 1))
        else:
            et = noise
        return (et + 0.5).float()

    def generate_s(self, n=1, noise=None):
        if noise is None:
            es = self.s_noise_dist.sample((n, 1))
        else:
            es = noise
        return np.pi * es

    def generate_i(self, t, noise=None):
        if noise is None:
            ei = torch.randn(t.shape)
        else:
            ei = noise
        return 191 * torch.sigmoid(.5 * ei + 2 * t - 5) + 64

    def generate(self, n=1):
        t = self.generate_t(n)
        i_ = self.generate_i(t)
        s = self.generate_s(n)
        return t, i_, s

    def counterfactual_t(self, x_, a_, t_new_):
        cls = a_[:, :10]
        t_, i_, s_ = a_[:, 10], a_[:, 11], a_[:, 12]
        i_noise = 2 * (sigmoid_inverse((i - 64) / 191) + 5 - 2 * t_)
        i_out = self.generate_i(t_new_, noise=i_noise)

        attrs_out = torch.concat([cls, t_new_.reshape((-1, 1)),
                                  i_out.reshape((-1, 1)),
                                  s_.reshape((-1, 1))], dim=1)
        imgs_out = np.zeros((x_.size(0), 1, 28, 28))
        for idx, (im_, th_, in_, sl_) in enumerate(zip(x_.reshape((-1, 28, 28)).cpu().numpy(),
                                                       t_new_.flatten().cpu().numpy(),
                                                       i_out.flatten().cpu().numpy())):
            morph = ImageMorphology(im_.reshape((28, 28)), scale=16)
            new_img = np.float32(SetThickness(th_)(morph))
            new_img = morph.downscale(np.float32(SetSlant(sl_)(ImageMorphology(new_img))))
            img_min, img_max = new_img.min(), new_img.max()
            current_intensity = np.median(new_img[new_img >= img_min + (img_max - img_min) * .5])
            mult = in_ / current_intensity
            new_img = np.clip(new_img * mult, 0, 255)
            imgs_out[idx, 0, :, :] = new_img
        return torch.from_numpy(imgs_out), attrs_out


parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--image-model-file',
                    type=str,
                    help='file (.tar) for saved cnn models')
parser.add_argument('--attr-model-file',
                    type=str,
                    help='file (.tar) for saved attribute models')

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
        c_min, c_max = a_train[:, 10:].min(dim=0).values, a_train[:, 10:].max(dim=0).values
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
