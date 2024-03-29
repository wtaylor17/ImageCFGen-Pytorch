from argparse import ArgumentParser
import torch
import numpy as np
import os
import json
from image_scms.audio_mnist import ATTRIBUTE_DIMS, AudioMNISTData, Generator
from attribute_scms.audio_mnist import AudioMNISTCausalGraph
from scipy.io.wavfile import write as write_wav

parser = ArgumentParser()
parser.add_argument("-m", "--image-model", type=str)
parser.add_argument("-a", "--attribute-model", type=str)
parser.add_argument("--gender-clf", type=str, default=None)
parser.add_argument("--digit-clf", type=str, default=None)
parser.add_argument("--accent-clf", type=str, default=None)
parser.add_argument("-d", "--data", type=str, default="AudioMNIST-data.zip")
parser.add_argument("-n", "--num-samples", type=int, default=10)
parser.add_argument("-o", "--outdir", type=str, default=".")
parser.add_argument("-r", "--mc-rounds", type=int, default=1)


if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.image_model
    attr_path = args.attribute_model
    num_samples = args.num_samples
    mc_rounds = args.mc_rounds
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
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

        attr_strings = {
            k: data.inv_transforms[k](v)
            for k, v in attrs.items()
        }

        print("\n".join(f"{k}: {v.shape}" for k, v in attrs.items()))

        gen = 0
        for _ in range(mc_rounds):
            z = torch.randn(size=(num_samples, 512, 1, 1))
            gen = gen + G(z, attrs)
        gen = gen / mc_rounds
        spect = img_to_spect(gen)

        gender_preds, digit_preds, accent_preds = None, None, None
        if gender_clf:
            gender_preds = gender_clf(spect)
            gender_preds = torch.eye(
                ATTRIBUTE_DIMS["gender"]
            )[gender_preds.argmax(1).flatten()].cpu().numpy()
            gender_preds = data.inv_transforms["gender"](gender_preds)
        if digit_clf:
            digit_preds = digit_clf(spect)
            digit_preds = torch.eye(
                ATTRIBUTE_DIMS["digit"]
            )[digit_preds.argmax(1).flatten()].cpu().numpy()
            digit_preds = data.inv_transforms["digit"](digit_preds)
        if accent_clf:
            accent_preds = accent_clf(spect)
            accent_preds = torch.eye(
                ATTRIBUTE_DIMS["accent"]
            )[accent_preds.argmax(1).flatten()].cpu().numpy()
            accent_preds = data.inv_transforms["accent"](accent_preds)

        wavs = data.inv_transforms["audio"](spect.cpu().numpy()).cpu().numpy().reshape((num_samples, -1))
        for i in range(num_samples):
            curwav = np.int16(wavs[i] / np.max(np.abs(wavs[i])) * 32767)
            write_wav(os.path.join(outdir, f"sample-{i}.wav"), 8000, curwav)
            attrs_json = {
                k: str(v[i, 0])
                for k, v in attr_strings.items()
            }
            if digit_preds is not None:
                attrs_json["digit_pred"] = str(digit_preds[i].flatten()[0])
            if accent_preds is not None:
                attrs_json["accent_pred"] = str(accent_preds[i].flatten()[0])
            if gender_preds is not None:
                attrs_json["gender_pred"] = str(gender_preds[i].flatten()[0])
            json_str = json.dumps(attrs_json, indent=4)
            print(json_str)
            with open(os.path.join(outdir, f"sample-{i}.json"), "w") as fp:
                json.dump(attrs_json, fp, indent=4)
