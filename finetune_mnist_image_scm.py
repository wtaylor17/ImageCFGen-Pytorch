from argparse import ArgumentParser
import os

import torch
from pytorch_msssim import ssim
import numpy as np
import seaborn as sns
from image_scms import mnist
from image_scms.training_utils import AdversariallyLearnedInference, batchify
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--steps', type=int,
                    help='number of epochs to train the distributions',
                    default=20)
parser.add_argument('--model-file',
                    type=str,
                    help='file (.tar) for saved cnn models')
parser.add_argument('--metric',
                    type=str,
                    default='ssim')
parser.add_argument('--latent-loss',
                    action='store_true')
parser.add_argument('--lr',
                    type=float,
                    default=1e-5)
parser.add_argument('--latent-scale',
                    type=float,
                    default=0.01)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()
    use_latent_loss = args.latent_loss

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

    E, G, D, model_dict = mnist.load_model(args.model_file,
                                           device=device,
                                           return_raw=True)
    E = E.to(device)
    G = G.to(device)
    D = D.to(device)

    E.train()
    opt = torch.optim.Adam(E.parameters(), lr=args.lr)
    loss_calc = AdversariallyLearnedInference(E, G, D)

    for i in range(args.steps):
        R, L = 0, 0
        n_batches = 0
        for x, a in tqdm(list(batchify(x_train, a_train, device=device))):
            x = x.reshape((-1, 1, 28, 28)) / 255.0
            c = torch.clone(a.reshape((-1, 13)))
            c_min, c_max = c[:, 10:].min(dim=0).values, c[:, 10:].max(dim=0).values
            c[:, 10:] = (c[:, 10:] - c_min) / (c_max - c_min)
            opt.zero_grad()
            codes = E(x, c)
            xr = G(codes, c)
            if args.metric == 'ssim':
                rec_loss = 1 - ssim(x, xr, data_range=1.0).mean()
            else:
                rec_loss = torch.square(x - xr).mean()
            loss = rec_loss
            if use_latent_loss:
                latent = torch.square(codes).sum(dim=1).mean()
                L += latent.item()
                loss = loss + args.latent_scale * latent
            R += rec_loss.item()
            loss.backward()
            opt.step()
            n_batches += 1
        print(f'Epoch {i + 1}/{args.steps}: {args.metric}={round(R / n_batches, 4)} ', end='')
        if use_latent_loss:
            print(f'latent loss (znorm) ={round(L / n_batches, 4)}')
        else:
            print()

    model_dict['E_state_dict'] = E.state_dict()
    torch.save(model_dict, f'models_state_dict_ImageCFGen_MNIST_tis_fine_tuned_{args.metric}.tar')
