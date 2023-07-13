from argparse import ArgumentParser
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from attribute_scms.mnist import MNISTCausalGraph


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

    graph: MNISTCausalGraph = torch.load('mnist-attribute-scm.tar', map_location=device)["graph"]
    graph.get_module('thickness').eval()

    ind = np.random.permutation(len(x_test))[0]
    a_train = {
        "digit": a_train[:, :10].float(),
        "thickness": a_train[:, 10:11].float(),
        "intensity": a_train[:, 11:12].float(),
        "slant": a_train[:, 12:13].float()
    }
    attr_stats = {
        k: (v.min(dim=0).values, v.max(dim=0).values)
        for k, v in a_train.items()
        if k != "digit"
    }

    x_test = x_test[ind:ind + 1]

    a_test = {
        "digit": a_test[ind:ind+1, :10].float(),
        "thickness": a_test[ind:ind+1, 10:11].float(),
        "intensity": a_test[ind:ind+1, 11:12].float(),
        "slant": a_test[ind:ind+1, 12:13].float()
    }
    a_test_scaled = {"digit": a_test["digit"]}
    for k in attr_stats:
        if k != "digit":
            a_test_scaled[k] = 2 * (a_test[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1

    with torch.no_grad():
        t_cf = {"thickness": a_test["thickness"] + 2}
        new_attrs = graph.sample_cf(a_test, t_cf)
        a_cf_scaled = {"digit": new_attrs["digit"]}
        for k in attr_stats:
            if k != "digit":
                a_cf_scaled[k] = 2 * (new_attrs[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
        print(a_test_scaled)
        print(a_cf_scaled)
        bigan_cf = bigan_model_dict["G"](bigan_model_dict["E"](x_test, a_test_scaled), a_cf_scaled)
        vae_cf = vae.decoder(vae.encoder(x_test, a_test_scaled)[0], a_cf_scaled)
        bigan_ft_cf = bigan_ft_model_dict["G"](bigan_ft_model_dict["E"](x_test, a_test_scaled), a_cf_scaled)

        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(x_test.reshape((28, 28)), vmin=-1, vmax=1, cmap="gray")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title('Original')
        axs[1].imshow(bigan_cf.reshape((28, 28)), vmin=-1, vmax=1, cmap="gray")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title('ImageCFGen')
        axs[2].imshow(bigan_ft_cf.reshape((28, 28)), vmin=-1, vmax=1, cmap="gray")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_title('ImageCFGen (fine-tuned)')
        axs[3].imshow(vae_cf.reshape((28, 28)), vmin=-1, vmax=1, cmap="gray")
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        axs[3].set_title('DeepSCM')
        fig.suptitle('Morpho-MNIST do(t+2) counterfactuals')
        plt.show()
