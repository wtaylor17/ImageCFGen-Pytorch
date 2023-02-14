from argparse import ArgumentParser
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
    seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-train.npy')
    )).float().to(device)
    x_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().to(device)

    a_train = {
        "digit": a_train[:, :10].int(),
        "thickness": a_train[:, 10:11].float(),
        "intensity": a_train[:, 11:12].float(),
        "slant": a_train[:, 12:13].float()
    }
    a_test = {
        "digit": a_test[:, :10].int(),
        "thickness": a_test[:, 10:11].float(),
        "intensity": a_test[:, 11:12].float(),
        "slant": a_test[:, 12:13].float()
    }

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
    c = {
        k: v[0:1]
        for k, v in c.items()
    }

    fig, axs = plt.subplots(1, 3)

    torch.manual_seed(seed)
    z = torch.randn(size=(1, 512, 1, 1))

    c["digit"] = eye[original_dig].reshape((1, 10))
    gen = vae.decoder(z, c)
    gen = gen.detach().numpy()
    axs[0].imshow(gen.reshape((28, 28)), vmin=-1, vmax=1)
    axs[0].set_title(f"Original ({original_dig})")
    axs[0].axis('off')
    c["digit"] = eye[original_dig].reshape((1, 10))
    gen = vae.decoder(z, c)
    gen = gen.detach().numpy()
    im = axs[1].imshow(gen.reshape((28, 28)), vmin=-1, vmax=1)
    txt = axs[1].text(14, 35, "prediction", fontsize=12)
    axs[1].axis('off')
    axs[1].set_title("Combination")

    slider_digit = Slider(ax=fig.add_axes([0.4, 0.85, 0.2, 0.04]),
                          label="Digit",
                          valmin=0,
                          valmax=1,
                          valinit=1,
                          orientation="horizontal")


    def update_digit(v):
        c["digit"] = (v * eye[original_dig] + (1 - v) * eye[target_dig]).reshape((1, 10))
        gen_ = vae.decoder(z, c)
        pred = clf(gen_).softmax(1).detach().numpy().tolist()[0]
        gen_ = gen_.detach().numpy()

        im.set_data(gen_.reshape((28, 28)))
        txt.set_text(f"p({original_dig})={round(pred[original_dig], 4)},"
                     f" p({target_dig})={round(pred[target_dig], 4)}")
        fig.canvas.draw_idle()


    slider_digit.on_changed(update_digit)

    slider_thickness = Slider(ax=fig.add_axes([0.4, 0.9, 0.2, 0.04]),
                              label="Thickness",
                              valmin=0,
                              valmax=1,
                              valinit=(c["thickness"][0, 0].item() + 1) / 2,
                              orientation="horizontal")


    def update_thickness(v):
        global c
        c_raw = {  # scale current attrs to original
            k: (val + 1) / 2 * (attr_stats[k][1] - attr_stats[k][0]) + attr_stats[k][0]
            for k, val in c.items()
            if k != "digit"
        }
        c_raw["digit"] = c["digit"]
        cf_raw = {  # intervention on thickness for cf on original scale
            "thickness": v * torch.ones((1, 1))
            * (attr_stats["thickness"][1] - attr_stats["thickness"][0])
            + attr_stats["thickness"][0]
        }
        c = graph.sample_cf(c_raw, cf_raw)
        c = {
            k: 2 * (c[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
            for k in c
            if k != "digit"
        }
        c["digit"] = c_raw["digit"]
        gen_ = vae.decoder(z, c)
        pred = clf(gen_).softmax(1).detach().numpy().tolist()[0]
        gen_ = gen_.detach().numpy()

        im.set_data(gen_.reshape((28, 28)))
        txt.set_text(f"p({original_dig})={round(pred[original_dig], 4)}, "
                     f"p({target_dig})={round(pred[target_dig], 4)}")
        slider_intensity.set_val((c["intensity"][0, 0].item() + 1) / 2)
        fig.canvas.draw_idle()


    slider_thickness.on_changed(update_thickness)

    slider_intensity = Slider(ax=fig.add_axes([0.4, 0.95, 0.2, 0.04]),
                              label="Intensity",
                              valmin=0,
                              valmax=1,
                              valinit=(c["intensity"][0, 0].item() + 1) / 2,
                              orientation="horizontal")

    def update_intensity(v):
        global c
        c_raw = {  # scale current attrs to original
            k: (val + 1) / 2 * (attr_stats[k][1] - attr_stats[k][0]) + attr_stats[k][0]
            for k, val in c.items()
            if k != "digit"
        }
        c_raw["digit"] = c["digit"]
        cf_raw = {  # intervention on intensity for cf on original scale
            "intensity": v * torch.ones((1, 1))
            * (attr_stats["intensity"][1] - attr_stats["intensity"][0])
            + attr_stats["intensity"][0]
        }
        c = graph.sample_cf(c_raw, cf_raw)
        c = {
            k: 2 * (c[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
            for k in c
            if k != "digit"
        }
        c["digit"] = c_raw["digit"]
        gen_ = vae.decoder(z, c)
        pred = clf(gen_).softmax(1).detach().numpy().tolist()[0]
        gen_ = gen_.detach().numpy()

        im.set_data(gen_.reshape((28, 28)))
        txt.set_text(f"p({original_dig})={round(pred[original_dig], 4)}, "
                     f"p({target_dig})={round(pred[target_dig], 4)}")
        fig.canvas.draw_idle()

    slider_intensity.on_changed(update_intensity)

    c["digit"] = eye[target_dig].reshape((1, 10))
    gen = vae.decoder(z, c)
    gen = gen.detach().numpy()
    axs[2].imshow(gen.reshape((28, 28)), vmin=-1, vmax=1)
    axs[2].axis('off')
    axs[2].set_title(f"Target ({target_dig})")
    fig.tight_layout()
    c["digit"] = eye[original_dig].reshape((1, 10))
    plt.show()
