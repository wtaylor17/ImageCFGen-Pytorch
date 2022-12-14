from attribute_scms import mnist
from image_scms.mnist import load_model as load_image_scm
from train_mnist_clf import MNISTClassifier
import torch
from argparse import ArgumentParser
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument("--graph",
                    type=str,
                    help="path to .tar file holding causal graph",
                    required=True)
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
parser.add_argument("-n", "--num-samples", type=int, default=10)
parser.add_argument("cf_attribute", type=str)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()
    idx = np.random.permutation(10_000)[0]
    graph: mnist.MNISTCausalGraph = torch.load(args.graph, map_location=device)["graph"]
    clf: torch.nn.Module = torch.load(args.classifier, map_location=device)["clf"]
    E, G, D = load_image_scm(args.image_scm, device=device)
    E.eval()
    G.eval()
    D.eval()

    n = args.num_samples
    cf_attribute = args.cf_attribute

    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)[idx:idx+1]
    x_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().reshape((-1, 1, 28, 28)).to(device)[idx:idx+1] / 255.0
    c_min, c_max = a_train[:, 10:].min(dim=0).values, a_train[:, 10:].max(dim=0).values
    a_raw = torch.clone(a_test)
    a_test[:, 10:] = (a_test[:, 10:] - c_min) / (c_max - c_min)

    codes = E(x_test, a_test)
    rec = G(codes, a_test)
    y_orig = clf(rec).softmax(1)

    graph.get_module("thickness").td.transforms[0].training = False

    attr_inds = {
        "thickness": 10,
        "intensity": 11,
        "slant": 12
    }
    obs = {
        "label": a_raw[:, :10].argmax(dim=1),
        "thickness": a_raw[:, 10:11],
        "intensity": a_raw[:, 11:12],
        "slant": a_raw[:, 12:13]
    }

    attr_vals = torch.linspace(c_min[attr_inds[cf_attribute] - 10],
                               c_max[attr_inds[cf_attribute] - 10],
                               100).reshape((100, 1))
    cfs = []
    clfs = []

    for val in attr_vals:
        obs_cond = {
            k: v
            for k, v in obs.items()
            if k != cf_attribute
        }
        obs_int = {
            cf_attribute: val.reshape((1, 1))
        }
        obs_cf = graph.sample_cf(obs, obs_int)
        a_cf = torch.concat([a_test[:, :10],
                             obs_cf["thickness"].reshape((-1, 1)),
                             obs_cf["intensity"].reshape((-1, 1)),
                             obs_cf["slant"].reshape((-1, 1))], dim=1)
        a_cf[:, 10:] = (a_cf[:, 10:] - c_min) / (c_max - c_min)
        x_cf = G(codes, a_cf)
        cfs.append(x_cf)
        y_cf = clf(x_cf).softmax(1)
        clfs.append(y_cf)

    cfs = torch.concat(cfs, dim=0)
    clfs = torch.concat(clfs, dim=0)

    fig, axs = plt.subplots(2, 2)
    axs[0][0].imshow(rec.detach().cpu().numpy().reshape((28, 28)), vmin=0, vmax=1)
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])
    original_val = a_raw[:, attr_inds[cf_attribute]].flatten().detach().cpu().item()
    axs[0][0].set_title(f"Original ({cf_attribute} = {round(original_val, 4)})")
    for i in range(10):
        scores = clfs[:, i].detach().cpu().numpy().reshape((100, 1))
        axs[0][1].plot(attr_vals.detach().cpu().numpy(), scores, label=f"Class {i}")
    axs[0][1].vlines(original_val, -0.1, 1.1, ['black'])
    axs[0][1].set_ylim((-0.1, 1.1))
    axs[0][1].set_xlabel(cf_attribute)
    axs[0][1].set_ylabel(rf"$P(y|X_c)$")
    axs[0][1].legend(bbox_to_anchor=(1.3, 1.0))
    axs[1][0].imshow(cfs[0].detach().cpu().numpy().reshape((28, 28)), vmin=0, vmax=1)
    axs[1][0].set_xticks([])
    axs[1][0].set_yticks([])
    axs[1][0].set_title(r"$X_c^{min}$")
    axs[1][1].imshow(cfs[-1].detach().cpu().numpy().reshape((28, 28)), vmin=0, vmax=1)
    axs[1][1].set_xticks([])
    axs[1][1].set_yticks([])
    axs[1][1].set_title(r"$X_c^{max}$")
    fig.suptitle(f"Class scores as a function of {cf_attribute}")
    fig.tight_layout()
    plt.show()
