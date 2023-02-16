from argparse import ArgumentParser
import os

from explain.cf_example import DeepCounterfactualExplainer

import torch
import numpy as np
import pickle
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--seed', type=int, default=42)

if __name__ == '__main__':
    args = parser.parse_args()

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

    eye = torch.eye(10)
    c = {
        k: 2 * (a_test[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
        for k in attr_stats
    }
    c["digit"] = a_test["digit"].float()

    results = []
    n = 1000
    for i in tqdm(range(n), total=n):
        cur_c = {
            k: v[i:i + 1]
            for k, v in c.items()
        }
        cur_x = x_test[i:i + 1]
        explainer = DeepCounterfactualExplainer(
            vae.encoder.sample,
            vae.decoder,
            clf, "digit"
        )

        result_i = {
            "mse": {},
            "ssim": {},
            "mixture": {}
        }
        with torch.no_grad():
            codes = vae.encoder.sample(cur_x, cur_c)
            rec = vae.decoder(codes, cur_c)
            pred = clf(rec).argmax(1).cpu().item()
            for metric in result_i:
                for tgt in range(10):
                    if tgt != pred:
                        try:
                            samples, metrics = explainer.explain(cur_x, cur_c,
                                                                 target_class=tgt,
                                                                 sample_points=100,
                                                                 metric=metric)
                            result_i[metric][tgt] = (samples[0], metrics.flatten()[0].cpu().item())
                        except:
                            result_i[metric][tgt] = None
            results.append(result_i)

    with open("vae-cf-matrix.pkl", "wb") as fp:
        pickle.dump(results, fp)
