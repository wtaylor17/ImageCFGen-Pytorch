from argparse import ArgumentParser
import os

import torch
import numpy as np
from image_scms import mnist
from attribute_scms.mnist import MNISTCausalGraph
from deepscm_vae.training_utils import batchify_dict, batchify
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument("-m", "--image-model", type=str)
parser.add_argument("-a", "--attribute-model", type=str)
parser.add_argument("-c", "--classifier", type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    model_path = args.image_model
    attr_path = args.attribute_model
    clf_path = args.classifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = torch.load(model_path, map_location=device)['vae']
    attrs_graph: MNISTCausalGraph = torch.load(attr_path, map_location=device)["graph"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_test = 2 * torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().to(device).reshape((-1, 1, 28, 28)) / 255 - 1

    a_train = {
        "digit": a_train[:, :10].float(),
        "thickness": a_train[:, 10:11].float(),
        "intensity": a_train[:, 11:12].float(),
        "slant": a_train[:, 12:13].float()
    }
    a_test = {
        "digit": a_test[:, :10].float(),
        "thickness": a_test[:, 10:11].float(),
        "intensity": a_test[:, 11:12].float(),
        "slant": a_test[:, 12:13].float()
    }

    attr_stats = {
        k: (v.min(dim=0).values, v.max(dim=0).values)
        for k, v in a_train.items()
        if k != "digit"
    }

    digit_clf = torch.load(clf_path, map_location=device)["clf"]

    with torch.no_grad():
        vae.eval()

        digit_acc = 0
        x_batches = batchify(x_test, batch_size=128)
        a_batches = batchify_dict(a_test, batch_size=128)
        for (x,), a in tqdm(zip(x_batches, a_batches)):
            cf_obs = dict(**a)
            c_obs = dict(**a)
            c_obs = {
                k: 2 * (v - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
                for k, v in c_obs.items()
                if k != "digit"
            }
            c_obs["digit"] = a["digit"]
            codes = vae.encoder.sample(x, c_obs)

            cf_obs["digit"] = a["digit"].argmax(1)
            batch_digits = cf_obs["digit"].clone()
            while (cf_obs["digit"] == batch_digits).sum().item() > 0:
                new_sample = attrs_graph.sample(obs_in={
                    k: v
                    for k, v in cf_obs.items()
                    if k != "digit"
                })
                mask = new_sample["digit"] != batch_digits
                cf_obs["digit"][mask] = new_sample["digit"][mask]
            cf_obs["digit"] = torch.eye(10)[cf_obs["digit"].flatten()]
            for k in c_obs:
                if k != "digit":
                    cf_obs[k] = c_obs[k]
            # print("\n".join(f"{k}: {v.min()}, {v.max()}" for k, v in cf_obs.items()))
            rec = vae.decoder(codes, cf_obs)
            pred = digit_clf(rec).argmax(1)
            digit_acc += (pred == cf_obs["digit"].argmax(1)).sum()

        print("Digit accuracy (test data w/ interventions):", digit_acc / len(x_test))
