from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from image_scms import mnist as image_mnist
from attribute_scms import mnist as attribute_mnist
from attribute_scms.mnist import MNISTCausalGraph
from tqdm import tqdm
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
        i_noise = 2 * (sigmoid_inverse((i_ - 64) / 191) + 5 - 2 * t_)
        i_out = self.generate_i(t_new_, noise=i_noise.reshape((-1, 1)))
        attrs_out = torch.concat([cls.reshape((-1, 10)), t_new_.reshape((-1, 1)),
                                  i_out.reshape((-1, 1)),
                                  s_.reshape((-1, 1))], dim=1)
        imgs_out = np.zeros((x_.size(0), 1, 28, 28))
        for idx, (im_, th_, in_, sl_) in tqdm(enumerate(zip(x_.reshape((-1, 28, 28)).cpu().numpy(),
                                                            t_new_.flatten().cpu().numpy(),
                                                            i_out.flatten().cpu().numpy(),
                                                            s_.cpu().numpy()))):
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
    )).float().to(device)[:20]
    x_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-train.npy')
    )).float().to(device)
    x_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().to(device)[:20]

    ground_truth_scm = MorphoMNISTSCM()

    E, G, D = image_mnist.load_model(args.image_model_file,
                                     device=device)
    E = E.to(device)
    G = G.to(device)
    D = D.to(device)

    graph: MNISTCausalGraph = torch.load(args.attr_model_file)["graph"]

    n_show = 10
    inds = np.random.permutation(len(x_test))
    D.eval()
    E.eval()
    G.eval()
    with torch.no_grad():
        x = x_test.reshape((-1, 1, 28, 28)).to(device) / 255.0
        c = torch.clone(a_test.reshape((-1, 13)))
        c_raw = a_test.reshape((-1, 13))
        c_min, c_max = a_train[:, 10:].min(dim=0).values, a_train[:, 10:].max(dim=0).values
        c[:, 10:] = (c[:, 10:] - c_min) / (c_max - c_min)

        t_new = c_raw[:, 10:11] + 2

        cf = graph.sample_cf({
            "label": c[:, 10:].argmax(1),
            "thickness": c_raw[:, 10:11],
            "intensity": c_raw[:, 11:12],
            "slant": c_raw[:, 12:13]
        }, {
            "thickness": t_new
        })

        true_image_out, true_new_attributes = ground_truth_scm.counterfactual_t(x_test, c_raw, t_new)

        codes = E(x, c)
        c_cf = torch.clone(c_raw)
        c_cf[:, 10] = t_new.flatten()
        c_cf[:, 11] = cf["intensity"].flatten()
        c_cf[:, 10:] = (c_cf[:, 10:] - c_min) / (c_max - c_min)
        pred_image_out = G(codes, c_cf)

        real = x_test.reshape((-1, 28, 28)).cpu().numpy() / 255.0
        true_image_out = true_image_out / 255.0

        mae = torch.abs(true_image_out - pred_image_out).mean()
        print('MAE (pixel):', 255 * mae.item())

        fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.05, hspace=0)
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('ImageCFGen Morpho-MNIST Thickness Counterfactuals')
        fig.text(0.01, 0.75, 'Original', ha='left')
        fig.text(0.01, 0.5, 'do(t+2) GT', ha='left')
        fig.text(0.01, 0.25, 'do(t+2) pred', ha='left')

        for i in range(n_show):
            j = inds[i]
            ax[0, i].imshow(real[j], cmap='gray', vmin=0, vmax=1)
            # ax[0, i].set_title(
            #     f'c = {c_raw[j, :10].argmax().item()}, t = {round(float(c_raw[j, 10].item()), 2)}\ni'
            #     f' = {round(float(c_raw[j, 11].item()), 2)}, s = {round(float(c_raw[j, 12].item()), 2)}',
            #     fontsize=8)
            ax[0, i].axis('off')
            ax[1, i].imshow(true_image_out[j].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
            ax[1, i].axis('off')
            ax[2, i].imshow(pred_image_out[j].reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
            ax[2, i].axis('off')
        plt.show()
