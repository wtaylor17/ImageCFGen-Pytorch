from argparse import ArgumentParser
import os

import torch
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

    opt = torch.optim.Adam(E.parameters())
    loss_calc = AdversariallyLearnedInference(E, G, D)

    for i in range(args.steps):
        loss = 0
        n_batches = 0
        for x, a in tqdm(list(batchify(x_train, a_train, device=device))):
            x = x.reshape((-1, 1, 28, 28)).to(device) / 255.0
            c = a.reshape((-1, 13)).to(device)
            opt.zero_grad()
            rec = loss_calc.rec_loss(x, c, metric=args.metric)
            rec.backward()
            opt.step()
            loss += rec.item()
            n_batches += 1
        print(f'Epoch {i+1}/{args.steps}: {round(loss / n_batches, 4)}')

    torch.save({
        'D_state_dict': D.state_dict(),
        'E_state_dict': E.state_dict(),
        'G_state_dict': G.state_dict(),
        'optimizer_E_state_dict': opt.state_dict(),
    }, f'models_state_dict_ImageCFGen_MNIST_tis_fine_tuned_{args.metric}.tar')