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
parser.add_argument("-a", "--attribute-model", type=str)
parser.add_argument("-c", "--classifier", type=str)
parser.add_argument("-n", "--num-samples", type=int, default=10_000)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()
    model_path = args.image_model
    attr_path = args.attribute_model
    num_samples = args.num_samples
    clf_path = args.classifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator: mnist.Generator = torch.load(model_path, map_location=device)["G"].to(device)
    attrs_graph: MNISTCausalGraph = torch.load(attr_path, map_location=device)["graph"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
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

    digit_clf = torch.load(clf_path, map_location=device)["clf"]

    with torch.no_grad():
        generator.eval()

        digit_acc = 0
        attrs = attrs_graph.sample(n=num_samples)
        attrs["digit"] = torch.eye(10)[attrs["digit"].flatten()]
        for k in attr_stats:
            attrs[k] = 2 * (attrs[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1

        for attr_batch in batchify_dict(attrs):
            z = torch.randn(size=(len(attr_batch["digit"]), mnist.LATENT_DIM, 1, 1))
            gen = generator(z, attr_batch)
            preds = digit_clf(gen).argmax(1)
            true = attr_batch["digit"].argmax(1)
            digit_acc += (true == preds).sum().cpu().item()

        print("Digit accuracy (generated):", digit_acc / num_samples)
