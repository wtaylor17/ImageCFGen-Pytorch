from argparse import ArgumentParser

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from attribute_scms import mnist


parser = ArgumentParser()
parser.add_argument('--data', type=str,
                    help='path to .npy file for MNIST attributes',
                    default='./mnist-a-train.py')
parser.add_argument('--steps', type=int,
                    help='number of epochs to train the distributions',
                    default=2000)


if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(args.data)).float().to(device)

    t_dist, i_given_t_dist, s_dist, l_dist, opt = mnist.train(a_train,
                                                              device=device,
                                                              steps=args.steps)

    torch.save({
        't_dist': t_dist,
        'i_given_t_dist': i_given_t_dist,
        's_dist': s_dist,
        'optimizer_state_dict': opt.state_dict()
    }, 'MorphoMNIST_attribute_scm.tar')

    t_true = a_train[:, 10]
    t_gen = t_dist.sample((t_true.size(0), 1)).cpu().numpy().flatten()
    sns.histplot(t_true, label='observed', color='b', alpha=0.3, kde=True, stat='density')
    sns.histplot(t_gen, label='learned', color='r', alpha=0.3, kde=True, stat='density')
    plt.legend()
    plt.title('samples from p(t)')
    plt.xlabel('t')
    plt.savefig('MorphoMNIST_thickness_distribution.png')
    plt.close()

    t = torch.Tensor([[3.0]]).to(device)
    i_gen = i_given_t_dist.condition(t).sample((10_000, 1)).cpu().numpy().reshape((-1,))
    ei = torch.randn((10_000, 1)).to(device)
    i_true = (191 * torch.sigmoid(.5 * ei + 2 * t - 5) + 64).cpu().numpy().reshape((-1,))
    sns.histplot(i_true, label='observed', color='b', alpha=0.3, kde=True, stat='density')
    sns.histplot(i_gen, label='learned', color='r', alpha=0.3, kde=True, stat='density')
    plt.legend()
    plt.title('samples from p(i|t=3)')
    plt.xlabel('i')
    plt.savefig('MorphoMNIST_intensity_distribution.png')
    plt.close()

    s_true = a_train[:, 12]
    s_gen = s_dist.sample((10_000, 1)).cpu().numpy().flatten()
    sns.histplot(s_true, label='observed', color='b', alpha=0.3, kde=True, stat='density')
    sns.histplot(s_gen, label='learned', color='r', alpha=0.3, kde=True, stat='density')
    plt.legend()
    plt.title('samples from p(s)')
    plt.xlabel('s')
    plt.savefig('MorphoMNIST_slant_distribution.png')
    plt.close()
