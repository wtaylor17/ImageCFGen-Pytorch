from argparse import ArgumentParser
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from image_scms import mnist
from deepscm_vae.training_utils import batchify_dict, batchify

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')

if __name__ == '__main__':
    np.random.seed(42)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = torch.load('mnist-vae.tar', map_location=device)["vae"]
    bigan_model_dict = torch.load('mnist-bigan.tar', map_location=device)
    bigan_ft_model_dict = torch.load('mnist-bigan-finetuned-mse.tar', map_location=device)

    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_test = 2 * torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().to(device).reshape((-1, 1, 28, 28)) / 255.0 - 1

    a_train = {
        "digit": a_train[:, :10].int(),
        "thickness": a_train[:, 10:11].float(),
        "intensity": a_train[:, 11:12].float(),
        "slant": a_train[:, 12:13].float()
    }

    attr_stats = {
        k: (v.min(dim=0).values, v.max(dim=0).values)
        for k, v in a_train.items()
        if k != "digit"
    }

    a_test = {
        "digit": a_test[:, :10].float(),
        "thickness": a_test[:, 10:11].float(),
        "intensity": a_test[:, 11:12].float(),
        "slant": a_test[:, 12:13].float()
    }

    with torch.no_grad():
        for k in attr_stats:
            if k != "digit":
                a_test[k] = 2 * (a_test[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1

        vae_rec = vae.decoder(vae.encoder(x_test, a_test)[0], a_test)
        bigan_rec = bigan_model_dict["G"](bigan_model_dict["E"](x_test, a_test), a_test)
        bigan_ft_rec = bigan_ft_model_dict["G"](bigan_ft_model_dict["E"](x_test, a_test), a_test)
        inds = np.random.randint(0, len(x_test), size=(4,))

        fig, axs = plt.subplots(4, 4)
        axs[0, 0].set_title('Original')
        axs[0, 1].set_title('ImageCFGen')
        axs[0, 2].set_title('ImageCFGen (fine-tuned)')
        axs[0, 3].set_title('DeepSCM')

        for j, i in enumerate(inds):
            axs[j, 0].imshow(x_test[i].cpu().numpy().reshape((28, 28)), vmin=-1, vmax=1, cmap="gray")
            axs[j, 0].set_xticks([])
            axs[j, 0].set_yticks([])
            axs[j, 1].imshow(bigan_rec[i].cpu().numpy().reshape((28, 28)), vmin=-1, vmax=1, cmap="gray")
            axs[j, 1].set_xticks([])
            axs[j, 1].set_yticks([])
            axs[j, 2].imshow(bigan_ft_rec[i].cpu().numpy().reshape((28, 28)), vmin=-1, vmax=1, cmap="gray")
            axs[j, 2].set_xticks([])
            axs[j, 2].set_yticks([])
            axs[j, 3].imshow(vae_rec[i].cpu().numpy().reshape((28, 28)), vmin=-1, vmax=1, cmap="gray")
            axs[j, 3].set_xticks([])
            axs[j, 3].set_yticks([])
        fig.suptitle('Morpho-MNIST reconstructions', fontsize=14)
    plt.show()
