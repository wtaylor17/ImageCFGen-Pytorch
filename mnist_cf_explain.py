from attribute_scms import mnist
from image_scms.mnist import load_model as load_image_scm
from train_mnist_clf import MNISTClassifier
from explain import simple_cf_distance
import torch
from argparse import ArgumentParser
import os
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument("--classifier",
                    type=str,
                    help="path to .tar file holding classifier",
                    required=True)
parser.add_argument("--image-scm",
                    type=str,
                    help="path to .tar file holding ALI model",
                    required=True)
parser.add_argument("--data-dir",
                    type=str,
                    help="path to folder containing morpho-mnist data",
                    required=True)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()
    idx = np.random.permutation(10_000)[0]
    clf: torch.nn.Module = torch.load(args.classifier, map_location=device)["clf"]
    E, G, D = load_image_scm(args.image_scm, device=device)
    E.eval()
    G.eval()
    D.eval()

    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().reshape((-1, 1, 28, 28)).to(device) / 255.0
    c_min, c_max = a_train[:, 10:].min(dim=0).values, a_train[:, 10:].max(dim=0).values
    a_raw = torch.clone(a_test)
    a_test[:, 10:] = (a_test[:, 10:] - c_min) / (c_max - c_min)

    with torch.no_grad():
        criterion = torch.nn.CrossEntropyLoss()
        base_idx = idx
        base_attrs = a_test[base_idx:base_idx + 1]
        base_img = x_test[base_idx:base_idx + 1]
        base_codes = E(base_img, base_attrs)
        base_rec = G(base_codes, base_attrs)
        base_logits = clf(base_rec)
        best_loss = float('inf')
        best_idx = None
        best_rec = None
        best_logits = None
        for i in tqdm(list(range(x_test.size(0)))):
            new_attrs = a_test[i:i+1]
            new_rec = G(base_codes, new_attrs)
            new_logits = clf(new_rec)
            # want low attr dist, high pred dist
            attr_dist = (new_attrs - base_attrs).square().mean().item()
            pred_dist = criterion(new_logits, base_logits).item()
            loss = attr_dist - 0.1 * pred_dist
            if loss < best_loss:
                best_loss = loss
                best_idx = i
                best_rec = new_rec
                best_logits = new_logits
        best_ds_logits = clf(x_test[best_idx:best_idx+1])

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(base_img.reshape((28, 28)).detach().cpu(), vmin=0, vmax=1)
    axs[0].set_title(f"Original (pred = {base_logits.softmax(1).argmax(1).item()})")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(x_test[best_idx].reshape((28, 28)).detach().cpu(), vmin=0, vmax=1)
    axs[1].set_title(f"CF attrs in data (pred = {best_ds_logits.softmax(1).argmax(1).item()})")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].imshow(best_rec.reshape((28, 28)).detach().cpu(), vmin=0, vmax=1)
    axs[2].set_title(f"CF found (pred = {best_logits.softmax(1).argmax(1).item()})")
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    plt.show()
