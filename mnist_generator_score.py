from argparse import ArgumentParser
import os

import torch
import numpy as np
import seaborn as sns
from image_scms import mnist
from attribute_scms.mnist import MNISTCausalGraph
from deepscm_vae.training_utils import batchify_dict

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument("-m", "--image-model", type=str)
parser.add_argument("-c", "--classifier", type=str)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()
    model_path = args.image_model
    clf_path = args.classifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(model_path, map_location=device)
    if 'vae' in model_path:
        generator = model_dict["vae"].decoder
        encoder = lambda *arg: model_dict["vae"].encoder(*arg)[0]
    else:
        generator = model_dict["G"]
        encoder = model_dict["E"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)

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

    digit_clf = torch.load(clf_path, map_location=device)["clf"]

    with torch.no_grad():
        digit_acc = 0

        for k in attr_stats:
            a_test[k] = 2 * (a_test[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1

        for attr_batch in batchify_dict(a_test):
            z = torch.randn(size=(len(attr_batch["digit"]), mnist.LATENT_DIM, 1, 1))
            gen = generator(z, attr_batch)
            preds = digit_clf(gen).argmax(1)
            true = attr_batch["digit"].argmax(1)
            digit_acc += (true == preds).sum().cpu().item()

        print("Digit accuracy (generated):", digit_acc / len(a_test["digit"]))
