from argparse import ArgumentParser
import os

import torch
from pytorch_msssim import ssim
import numpy as np
import seaborn as sns
from image_scms.training_utils import batchify, batchify_dict
from image_scms.audio_mnist import AudioMNISTData, VALIDATION_RUNS, IMAGE_SHAPE, ATTRIBUTE_DIMS, Encoder, Generator
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--data', type=str,
                    help='path to ZIP',
                    default='')
parser.add_argument('--steps', type=int,
                    help='number of epochs to train the distributions',
                    default=20)
parser.add_argument('--model-file',
                    type=str,
                    help='file (.tar) for saved cnn models')
parser.add_argument('--metric',
                    type=str,
                    default='mse')
parser.add_argument('--lr',
                    type=float,
                    default=1e-5)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = AudioMNISTData(args.data, device=device)

    spect_mean, spect_ss, n_batches = 0, 0, 0

    print('Computing spectrogram statistics...')
    for batch in data.stream(batch_size=64,
                             excluded_runs=VALIDATION_RUNS):
        n_batches += 1
        spect_mean = spect_mean + batch["audio"].mean(dim=(0, 1)).reshape((1, 1, -1))
        spect_ss = spect_ss + batch["audio"].square().mean(dim=(0, 1)).reshape((1, 1, -1))

    spect_mean = (spect_mean / n_batches).float().to(device)  # E[X]
    spect_ss = (spect_ss / n_batches).float().to(device)  # E[X^2]
    spect_std = torch.sqrt(spect_ss - spect_mean.square())
    stds_kept = 3

    def spect_to_img(spect_):
        spect_ = (spect_ - spect_mean) / (spect_std + 1e-6)
        return torch.clip(spect_, -stds_kept, stds_kept) / float(stds_kept)

    def img_to_spect(img_):
        return img_ * stds_kept * (spect_std + 1e-6) + spect_mean

    model_dict = torch.load(args.model_file, map_location=device)
    E = Encoder().to(device)
    E.load_state_dict(model_dict["E_state_dict"])
    G = Generator().to(device)
    G.load_state_dict(model_dict["G_state_dict"])

    E.train()
    G.eval()
    opt = torch.optim.Adam(E.parameters(), lr=args.lr)

    for i in range(args.steps):
        R, L = 0, 0
        n_batches = 0

        for batch in tqdm(data.stream(batch_size=64,
                                      excluded_runs=VALIDATION_RUNS)):
            images = batch["audio"].reshape((-1, 1, *IMAGE_SHAPE)).float().to(device)
            a = {k: torch.clone(batch[k]).float().to(device)
                 for k in ATTRIBUTE_DIMS
                 if k != "audio"}
            x = spect_to_img(images)

            opt.zero_grad()
            codes = E(x, a)
            xr = G(codes, a)
            if args.metric == 'ssim':
                rec_loss = 1 - ssim(x, xr, data_range=1.0).mean()
            else:
                rec_loss = torch.square(x - xr).mean()
            loss = rec_loss
            latent = torch.square(codes).mean()
            L += latent.item()
            loss = loss + latent
            R += rec_loss.item()
            loss.backward()
            opt.step()
            n_batches += 1
        print(f'Epoch {i + 1}/{args.steps}: {args.metric}={round(R / n_batches, 4)} ', end='')
        print(f'latent loss ={round(L / n_batches, 4)}')

    model_dict["E_state_dict"] = E.state_dict()
    model_dict["G_state_dict"] = G.state_dict()
    torch.save(model_dict, f'audio-mnist-bigan-finetuned-{args.metric}.tar')
