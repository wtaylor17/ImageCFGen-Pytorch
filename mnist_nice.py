from attribute_scms import mnist
from image_scms.mnist import load_model as load_image_scm
from train_mnist_clf import MNISTClassifier
from explain import nice
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
    a_train[:, 10:] = (a_train[:, 10:] - c_min) / (c_max - c_min)
    a_test[:, 10:] = (a_test[:, 10:] - c_min) / (c_max - c_min)

    explainer = nice.LocalCFNICE(E, G, clf, a_train, fixed_feat=list(range(10)))

    x_cf, a_cf = explainer.explain(x_test, a_test, from_logits=True)
    print(x_cf.size(), a_cf.size())
