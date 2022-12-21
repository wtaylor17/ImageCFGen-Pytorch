from argparse import ArgumentParser
import torch
import numpy as np
import os
from image_scms.audio_mnist import ATTRIBUTE_DIMS, AudioMNISTData, Generator
from scipy.io.wavfile import write as write_wav

parser = ArgumentParser()
parser.add_argument("-m", "--model", type=str)
parser.add_argument("-d", "--data", type=str, default="AudioMNIST-data.zip")
parser.add_argument("-n", "--num-samples", type=int, default=10)
parser.add_argument("-o", "--outdir", type=str, default=".")


if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.model
    num_samples = args.num_samples
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(model_path, map_location=device)
    G = Generator().to(device)
    G.load_state_dict(model_dict["G_state_dict"])
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
        spect_std = torch.sqrt(spect_ss - spect_mean.square())
        stds_kept = 3


        def spect_to_img(spect_):
            spect_ = (spect_ - spect_mean) / (spect_std + 1e-6)
            return torch.clip(spect_, -stds_kept, stds_kept) / float(stds_kept)


        def img_to_spect(img_):
            return img_ * stds_kept * (spect_std + 1e-6) + spect_mean

        z = torch.randn(size=(num_samples, 512, 1, 1))
        attrs = {
            k: torch.randint(0, v, size=(num_samples, 1))
            for k, v in ATTRIBUTE_DIMS.items()
        }

        spect = img_to_spect(G(z, attrs))

        wavs = data.inv_transforms["audio"](spect.cpu().numpy()).cpu().numpy()
        wavs = np.int16(wavs / np.max(np.abs(wavs)) * 32767)
        for i in range(num_samples):
            write_wav(os.path.join(outdir, f"sample-{i}.wav"), 8000, wavs[i])
