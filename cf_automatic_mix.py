from argparse import ArgumentParser
import os

from explain.cf_example import DeepCounterfactualExplainer

import torch
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--original', type=int, default=3)
parser.add_argument('--target', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)

if __name__ == '__main__':
    args = parser.parse_args()
    original_dig = args.original
    target_dig = args.target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_test = 2 * torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).reshape((-1, 1, 28, 28)).float().to(device) / 255 - 1
    np.random.seed(args.seed)
    inds = np.random.permutation(len(x_test))

    a_train = {
        "digit": a_train[:, :10].float(),
        "thickness": a_train[:, 10:11].float(),
        "intensity": a_train[:, 11:12].float(),
        "slant": a_train[:, 12:13].float()
    }
    a_test = {
        "digit": a_test[inds, :10].float(),
        "thickness": a_test[inds, 10:11].float(),
        "intensity": a_test[inds, 11:12].float(),
        "slant": a_test[inds, 12:13].float()
    }
    x_test = x_test[inds]

    attr_stats = {
        k: (v.min(dim=0).values, v.max(dim=0).values)
        for k, v in a_train.items()
        if k != "digit"
    }

    vae = torch.load('mnist-vae.tar', map_location='cpu')['vae']
    clf = torch.load('mnist_clf.tar', map_location='cpu')['clf']
    graph = torch.load('mnist-attribute-scm.tar', map_location='cpu')['graph']
    graph.get_module('thickness').eval()

    mask = a_test["digit"].argmax(1) == original_dig
    eye = torch.eye(10)
    c = {
        k: 2 * (a_test[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
        for k in attr_stats
    }
    c["digit"] = a_test["digit"].float()
    c = {
        k: v[mask][0:1]
        for k, v in c.items()
    }
    x = x_test[mask][0:1]

    explainer = DeepCounterfactualExplainer(
        vae.encoder.sample,
        vae.decoder,
        clf, "digit"
    )

    with torch.no_grad():
        rec = vae.decoder(vae.encoder.sample(x, c), c).cpu().numpy()
        samples, mse = explainer.explain(x, c, target_class=target_dig, sample_points=1000)
        best_image = samples[0].reshape((28, 28)).cpu().numpy()
        preds = clf(samples.reshape((-1, 1, 28, 28))).softmax(1).cpu().numpy()[0]
        x = x.reshape((28, 28)).cpu().numpy()
        diff_map = (best_image - x) / 2
        diff_map[diff_map < -.1] = -1
        diff_map[diff_map > .1] = 1

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(x, vmin=-1, vmax=1)
    axs[0].set_title(f'Original ({original_dig})')
    axs[1].imshow(best_image, vmin=-1, vmax=1)
    axs[1].set_title(f'CF ({target_dig}) (mixture = {round(mse[0].item(), 4)})')
    axs[2].imshow(diff_map, vmin=-1, vmax=1)
    axs[2].set_title("Difference")
    axs[3].bar(range(10), preds)
    axs[3].set_title(f'Predicted softmax probabilities')
    axs[3].set_xticks(list(range(10)))
    plt.show()
