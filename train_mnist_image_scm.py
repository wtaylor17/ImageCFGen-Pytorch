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
parser.add_argument('--steps', type=int,
                    help='number of epochs to train the distributions',
                    default=200)


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

    E, G, D, optimizer_D, optimizer_E = mnist.train(x_train,
                                                    a_train,
                                                    x_test=x_test,
                                                    a_test=a_test,
                                                    scale_a_after=10,
                                                    n_epochs=args.steps,
                                                    device=device)
    torch.save({
        'D_state_dict': D.state_dict(),
        'E_state_dict': E.state_dict(),
        'G_state_dict': G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'optimizer_EG_state_dict': optimizer_E.state_dict()
    }, 'models_state_dict_ImageCFGen_MNIST_tis.tar')
