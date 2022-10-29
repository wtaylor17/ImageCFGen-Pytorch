import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser

from morphomnist.perturb import SetThickness, SetSlant
from morphomnist.morpho import ImageMorphology


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
        i = self.generate_i(t)
        s = self.generate_s(n)
        return t, i, s


# dataset preparation
def load_dataset(batch_size=128, root=r'.\datasets'):
    test_dataset = torchvision.datasets.MNIST(root=root + r'\MNIST', train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def add_attributes(loader, thicknesses, intensities):
    mnist_data = [], []

    model = MorphoMNISTSCM()
    loader = list(loader)
    images = np.concatenate([instance[0] for instance in loader], axis=0)
    labels = np.concatenate([instance[1] for instance in loader], axis=0)
    slants = model.generate_s(len(images))

    for i, (image, label, thickness, intensity, slant) in enumerate(tqdm(list(zip(images,
                                                                                  labels,
                                                                                  thicknesses,
                                                                                  intensities,
                                                                                  slants)))):
        thickness, intensity, slant = thickness.item(), intensity.item(), slant.item()
        morph = ImageMorphology(image.reshape((28, 28)), scale=16)
        new_img = np.float32(SetThickness(thickness)(morph))
        new_img = morph.downscale(np.float32(SetSlant(slant)(ImageMorphology(new_img))))
        img_min, img_max = new_img.min(), new_img.max()
        current_intensity = np.median(new_img[new_img >= img_min + (img_max - img_min) * .5])
        mult = intensity / current_intensity
        new_img = np.clip(new_img * mult, 0, 255)

        # make one-hot embedding from labels
        c = np.zeros((13,), dtype=np.float32)
        c[label] = 1
        c[10] = thickness
        c[11] = intensity
        c[12] = slant
        mnist_data[0].append(new_img)
        mnist_data[1].append(c)

    return mnist_data


parser = ArgumentParser()
parser.add_argument('--deepscm-csv',
                    help='path to CSV data with deepSCM test attributes')

if __name__ == '__main__':
    mnist_test = load_dataset()

    csv_path = parser.parse_args().deepscm_csv
    df = pd.read_csv(csv_path)

    th = torch.from_numpy(df['thickness'].to_numpy()).float()
    in_ = torch.from_numpy(df['intensity'].to_numpy()).float()

    x_test, a_test = add_attributes(mnist_test, th, in_)
    np.save('mnist-x-test.npy', np.stack(x_test, axis=0))
    np.save('mnist-a-test.npy', np.stack(a_test, axis=0))
