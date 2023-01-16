from argparse import ArgumentParser
import torch
import numpy as np
import os
from image_scms.audio_mnist import ATTRIBUTE_DIMS, AudioMNISTData, Generator, Encoder, VALIDATION_RUNS
from image_scms.training_utils import batchify_dict
from attribute_scms.audio_mnist import AudioMNISTCausalGraph

parser = ArgumentParser()
parser.add_argument("-m", "--image-model", type=str)
parser.add_argument("-a", "--attribute-model", type=str)
parser.add_argument("--gender-clf", type=str, default=None)
parser.add_argument("--digit-clf", type=str, default=None)
parser.add_argument("--accent-clf", type=str, default=None)
parser.add_argument("--cf-attr", type=str, default="digit")
parser.add_argument("-d", "--data", type=str, default="AudioMNIST-data.zip")
parser.add_argument("-r", "--mc-rounds", type=int, default=4)


if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.image_model
    attr_path = args.attribute_model
    mc_rounds = args.mc_rounds
    gender_clf = args.gender_clf
    digit_clf = args.digit_clf
    accent_clf = args.accent_clf
    cf_attr = args.cf_attr

    print(f"Computing scores using CFs on attribute {cf_attr}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(model_path, map_location=device)
    E = Encoder().to(device)
    G = Generator().to(device)
    E.load_state_dict(model_dict["E_state_dict"])
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
        for batch in data.stream(batch_size=128,
                                 excluded_runs=VALIDATION_RUNS):
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

        n_valid = 0
        for batch in data.stream(batch_size=128,
                                 excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS))):
            n_valid += len(batch["digit"])
        attrs_cf = {
            cf_attr: torch.eye(ATTRIBUTE_DIMS[cf_attr])[
                attrs_graph.sample(n=n_valid)[cf_attr].flatten()
            ]
        }

        gender_acc, digit_acc, accent_acc = 0, 0, 0

        for batch in data.stream(batch_size=128,
                                 excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS))):
            spect = batch["audio"]
            img = spect_to_img(spect)
            codes = E(img, batch)

            gender_score, digit_score, accent_score = 0, 0, 0
            for _ in range(mc_rounds):
                cond = dict(**batch)
                cf_sample = {
                    k: v.argmax(1)
                    for k, v in batch.items()
                }
                del cf_sample["audio"]
                del cond["audio"], cond[cf_attr]

                cond = {
                    k: v.argmax(1)
                    for k, v in cond.items()
                }
                batch_attr = batch[cf_attr].argmax(1)
                while (cf_sample[cf_attr] == batch_attr).sum().item() > 0:
                    new_sample = attrs_graph.sample(obs_in=cond)
                    mask = new_sample[cf_attr] != batch_attr
                    cf_sample[cf_attr][mask] = new_sample[cf_attr][mask]

                cf_sample = {
                    k: torch.eye(ATTRIBUTE_DIMS[k])[v.flatten()]
                    for k, v in cf_sample.items()
                    if k in ATTRIBUTE_DIMS
                }
                rec = G(codes, cf_sample)
                rec = img_to_spect(rec)
                print(rec.shape)
                print("\n".join(f"{k}: {v.shape}" for k, v in cf_sample.items()))
                print(gender_score)
                if gender_clf:
                    gender_preds = gender_clf(rec).argmax(1)
                    gender_score += (gender_preds == cf_sample["gender"].argmax(1)).sum()
                if digit_clf:
                    digit_preds = digit_clf(rec).argmax(1)
                    digit_score += (digit_preds == cf_sample["digit"].argmax(1)).sum()
                if accent_clf:
                    accent_preds = accent_clf(rec).argmax(1)
                    accent_score += (accent_preds == cf_sample["accent"].argmax(1)).sum()
            gender_acc += gender_score
            digit_acc += digit_score
            accent_acc += accent_score

        gender_acc = gender_acc / (mc_rounds * n_valid)  # scale to [0,1]
        digit_acc = digit_acc / (mc_rounds * n_valid)
        accent_acc = accent_acc / (mc_rounds * n_valid)

        if gender_clf:
            print("Gender accuracy =", gender_acc)
        if digit_clf:
            print("Digit accuracy =", digit_acc)
        if accent_clf:
            print("Accent accuracy =", accent_acc)
