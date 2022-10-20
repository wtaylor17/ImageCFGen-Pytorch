import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.transforms import Compose, Lambda
from torchvision.datasets import mnist
from torch.utils.data import DataLoader

from morphomnist.perturb import RandomThickening
from morphomnist.morpho import ImageMorphology


def load_mnist_train(batch_size=32):
    return DataLoader(
        mnist.MNIST('./data', train=True, download=True,
                    transform=Compose([
                        Lambda(lambda x_: np.asarray(x_) / 255.0)
                    ])),
        batch_size=batch_size, shuffle=True)


def add_thicknesses(imgs: np.ndarray):
    T = []
    imgs_out = []
    for im in imgs:
        morph = ImageMorphology(im)
        th = RandomThickening()
        imgs_out.append(th(morph))
        T.append(th.last_thickness)

    return np.array(imgs_out), np.array(T)


if __name__ == '__main__':
    x, y = next(iter(load_mnist_train()))

    fig, axs = plt.subplots(4, 4)
    
    for i in range(4):
        xi = x[i]
        axs[i][0].imshow(ImageMorphology(xi).binary_image)
        axs[i][0].set_title('Original')
        axs[i][0].axis('off')
        xt, t = add_thicknesses(np.array([xi, xi, xi]))
        for j in range(1, 4):
            axs[i][j].imshow(xt[j-1])
            axs[i][j].set_title(f'do(t = {t[j-1]})')
            axs[i][j].axis('off')
    plt.show()
