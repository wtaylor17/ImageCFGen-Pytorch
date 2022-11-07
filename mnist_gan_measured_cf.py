from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from image_scms import mnist as image_mnist
from attribute_scms import mnist as attribute_mnist
from attribute_scms.training_utils import nf_forward, nf_inverse
from tqdm import tqdm
from morphomnist.morpho import ImageMorphology, ImageMoments
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

    def counterfactual_i(self, x_, a_, i_new_):
        cls = a_[:, :10]
        t_, i_, s_ = a_[:, 10], a_[:, 11], a_[:, 12]
        i_out = i_new_
        attrs_out = torch.concat([cls.reshape((-1, 10)), t_.reshape((-1, 1)),
                                  i_out.reshape((-1, 1)),
                                  s_.reshape((-1, 1))], dim=1)
        imgs_out = np.zeros((x_.size(0), 1, 28, 28))
        for idx, (im_, th_, in_, sl_) in tqdm(enumerate(zip(x_.reshape((-1, 28, 28)).cpu().numpy(),
                                                            t_.flatten().cpu().numpy(),
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

    def counterfactual_s(self, x_, a_, s_new_):
        cls = a_[:, :10]
        t_, i_, s_ = a_[:, 10], a_[:, 11], a_[:, 12]
        s_out = s_new_
        attrs_out = torch.concat([cls.reshape((-1, 10)), t_.reshape((-1, 1)),
                                  i_.reshape((-1, 1)),
                                  s_out.reshape((-1, 1))], dim=1)
        imgs_out = np.zeros((x_.size(0), 1, 28, 28))
        for idx, (im_, th_, in_, sl_) in tqdm(enumerate(zip(x_.reshape((-1, 28, 28)).cpu().numpy(),
                                                            t_.flatten().cpu().numpy(),
                                                            i_.flatten().cpu().numpy(),
                                                            s_out.cpu().numpy()))):
            morph = ImageMorphology(im_.reshape((28, 28)), scale=16)
            new_img = np.float32(SetThickness(th_)(morph))
            new_img = morph.downscale(np.float32(SetSlant(sl_)(ImageMorphology(new_img))))
            img_min, img_max = new_img.min(), new_img.max()
            current_intensity = np.median(new_img[new_img >= img_min + (img_max - img_min) * .5])
            mult = in_ / current_intensity
            new_img = np.clip(new_img * mult, 0, 255)
            imgs_out[idx, 0, :, :] = new_img
        return torch.from_numpy(imgs_out), attrs_out


def extract_observed_attributes(image):
    if type(image) is torch.Tensor:
        image = image.detach().cpu().numpy()
    image = image.reshape((28, 28))
    morph = ImageMorphology(image, scale=16)
    thickness = morph.mean_thickness
    img_min, img_max = image.min(), image.max()
    intensity = np.median(image[image >= img_min + (img_max - img_min) * .5])
    moments = ImageMoments(image)
    slant = moments.horizontal_shear
    return np.array([thickness, intensity, slant])


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
parser.add_argument('--cf-attribute',
                    type=str,
                    default='thickness')

np.random.seed(42)
torch.manual_seed(42)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)[:1000]
    x_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-train.npy')
    )).float().to(device)
    x_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().to(device)[:1000]

    ground_truth_scm = MorphoMNISTSCM()

    E, G, D = image_mnist.load_model(args.image_model_file,
                                     device=device)
    E = E.to(device)
    G = G.to(device)
    D = D.to(device)

    t_dist, i_given_t_dist, s_dist = attribute_mnist.load_model(args.attr_model_file,
                                                                device=device)

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

        if args.cf_attribute == 'thickness':
            i_noise_pred = nf_inverse(i_given_t_dist.condition(c_raw[:, 10:11]),
                                      c_raw[:, 11:12])
            t_new = t_dist.sample((len(c_raw),))
            i_new = nf_forward(i_given_t_dist.condition(t_new), i_noise_pred)

            true_image_out, true_new_attributes = ground_truth_scm.counterfactual_t(x_test, c_raw, t_new)
            attr_idx = 0
            s_new = c_raw[:, 12:13]
        elif args.cf_attribute == 'intensity':
            t_new = c_raw[:, 10:11]
            s_new = c_raw[:, 12:13]
            i_new = i_given_t_dist.condition(t_new).sample((len(c_raw),))
            attr_idx = 1
            true_image_out, true_new_attributes = ground_truth_scm.counterfactual_i(x_test, c_raw, i_new)
        elif args.cf_attribute == 'slant':
            t_new = c_raw[:, 10:11]
            i_new = c_raw[:, 11:12]
            s_new = s_dist.sample((len(c_raw),))
            attr_idx = 2
            true_image_out, true_new_attributes = ground_truth_scm.counterfactual_s(x_test, c_raw, s_new)
        else:
            raise ValueError

        measured_real = np.row_stack([
            extract_observed_attributes(true_image)
            for true_image in tqdm(true_image_out)
        ])

        codes = E(x, c)
        c_cf = torch.clone(c_raw)
        c_cf[:, 10] = t_new.flatten()
        c_cf[:, 11] = i_new.flatten()
        c_cf[:, 12] = s_new.flatten()
        c_cf[:, 10:] = (c_cf[:, 10:] - c_min) / (c_max - c_min)
        pred_image_out = G(codes, c_cf)

        measured_pred = np.row_stack([
            extract_observed_attributes(255 * pred_image)
            for pred_image in tqdm(pred_image_out)
        ])

        a_range = torch.range(float(a_train[:, 10 + attr_idx].min().item()) - 1,
                              float(a_train[:, 10 + attr_idx].max().item()) + 1, 0.01)

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(a_range, a_range, 'k--')
        axs[1].plot(a_range, a_range, 'k--')

        axs[0].scatter(true_new_attributes[:, 10 + attr_idx].cpu().numpy(),
                       measured_pred[:, attr_idx])
        axs[0].set_ylabel('ImageCFGen Measured')
        axs[0].set_xlabel('Target')
        axs[1].scatter(true_new_attributes[:, 10 + attr_idx].cpu().numpy(),
                       measured_real[:, attr_idx])
        axs[1].set_ylabel('REAL Measured')
        axs[1].set_xlabel('Target')
        fig.suptitle(args.cf_attribute.capitalize())

        plt.show()

        pred_error = torch.abs(true_new_attributes[:, -3:] - measured_pred)

        print(f'do({args.cf_attribute}) median absolute error:', pred_error[:, attr_idx].median().item())
