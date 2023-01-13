from argparse import ArgumentParser
import torch
import numpy as np
import os
from image_scms.audio_mnist import ATTRIBUTE_DIMS, AudioMNISTData, Generator
from image_scms.training_utils import batchify_dict
from attribute_scms.audio_mnist import AudioMNISTCausalGraph

parser = ArgumentParser()
parser.add_argument("-m", "--image-model", type=str)
parser.add_argument("-a", "--attribute-model", type=str)
parser.add_argument("--gender-clf", type=str, default=None)
parser.add_argument("--digit-clf", type=str, default=None)
parser.add_argument("--accent-clf", type=str, default=None)
parser.add_argument("-d", "--data", type=str, default="AudioMNIST-data.zip")
parser.add_argument("-n", "--num-samples", type=int, default=10_000)
parser.add_argument("-r", "--mc-rounds", type=int, default=4)


if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.image_model
    attr_path = args.attribute_model
    num_samples = args.num_samples
    mc_rounds = args.mc_rounds
    gender_clf = args.gender_clf
    digit_clf = args.digit_clf
    accent_clf = args.accent_clf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(model_path, map_location=device)
    G = Generator().to(device)
    G.load_state_dict(model_dict["G_state_dict"])
    attrs_graph: AudioMNISTCausalGraph = torch.load(attr_path, map_location=device)["graph"]

    if gender_clf:
        gender_clf = torch.load(gender_clf, map_location=device)["model"].to(device)
    if digit_clf:
        digit_clf = torch.load(digit_clf, map_location=device)["model"].to(device)
    if accent_clf:
        accent_clf = torch.load(accent_clf, map_location=device)["model"].to(device)

    with torch.no_grad():
        G.eval()

        print("loading data...")
        data = AudioMNISTData(args.data, device=device)
        print("done.")

        spect_mean, spect_ss, n_batches = 0, 0, 0

        print('Computing spectrogram statistics...')
        for batch in data.stream(batch_size=128):
            n_batches += 1
            spect_mean = spect_mean + batch["audio"].mean(dim=(0, 1)).reshape((1, 1, -1))
            spect_ss = spect_ss + batch["audio"].square().mean(dim=(0, 1)).reshape((1, 1, -1))

        spect_mean = (spect_mean / n_batches).float().to(device)  # E[X]
        spect_ss = (spect_ss / n_batches).float().to(device)  # E[X^2]
        spect_var = torch.relu(spect_ss - spect_mean.square())
        spect_std = torch.sqrt(spect_var)
        stds_kept = 3


        def spect_to_img(spect_):
            spect_ = (spect_ - spect_mean) / (spect_std + 1e-6)
            return torch.clip(spect_, -stds_kept, stds_kept) / float(stds_kept)


        def img_to_spect(img_):
            return img_ * stds_kept * (spect_std + 1e-6) + spect_mean

        attrs = {
            k: torch.eye(ATTRIBUTE_DIMS[k])[v.flatten()]
            for k, v in attrs_graph.sample(n=num_samples).items()
        }

        print("\n".join(f"{k}: {v.shape}" for k, v in attrs.items()))

        gender_acc, digit_acc, accent_acc = 0, 0, 0

        for attr_batch in batchify_dict(attrs, device=device):
            gen = 0
            for _ in range(mc_rounds):
                z = torch.randn(size=(len(attr_batch["gender"]), 512, 1, 1))
                gen = gen + G(z, attrs)
            gen = gen / mc_rounds
            spect = img_to_spect(gen)

            if gender_clf:
                gender_preds = gender_clf(spect).argmax(1)
                gender_acc += (gender_preds == attr_batch["gender"].argmax(1)).sum()
            if digit_clf:
                digit_preds = digit_clf(spect).argmax(1)
                digit_acc += (digit_preds == attr_batch["digit"].argmax(1)).sum()
            if accent_clf:
                accent_preds = accent_clf(spect).argmax(1)
                accent_acc += (accent_preds == attr_batch["accent"].argmax(1)).sum()

        gender_acc = gender_acc / num_samples
        digit_acc = digit_acc / num_samples
        accent_acc = accent_acc / num_samples

        if gender_clf:
            print("Gender accuracy =", gender_acc)
        if digit_clf:
            print("Digit accuracy =", digit_acc)
        if accent_clf:
            print("Accent accuracy =", accent_acc)
