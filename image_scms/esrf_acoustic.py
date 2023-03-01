import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
import torchaudio
from scipy.io.wavfile import read as read_wav
from functools import partial


def continuous_feature_map(c: torch.Tensor, size: tuple = (512, 512)):
    return c.reshape((c.size(0), 1, 1, 1)).repeat(1, 1, *size)


ATTRIBUTE_DIMS = {
    "closest_boat": 1,
    "has_boat": 1
}
LATENT_DIM = 512


class EsrfStation:
    def __init__(self, station_wav_path: str, station_label_csv: str, device="cpu"):
        self.device = device
        self.wav_paths = list(map(str, Path(station_wav_path).rglob("*.wav")))
        self.audio_to_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=1023, win_length=256, hop_length=79, pad=200
        ).to(self.device)
        self.spectrogram_to_audio = torchaudio.transforms.GriffinLim(
            n_fft=1023, win_length=256, hop_length=79
        ).to(self.device)

        self.df = pd.read_csv(station_label_csv)
        self.df["filepath"] = self.df["filepath"].str.split('/').apply(lambda x: x[-1])
        bg_cols = [c for c in self.df.columns if c.startswith('BG')]

        def glb(e):
            if (e > 0).sum():
                return 100 - max(i for i in range(len(e)) if e[i] > 0)
            return -1
        self.distance_feature = np.array([glb(e) for e in np.asarray(self.df[bg_cols])])
        self.has_boat = (self.distance_feature > 0).astype('int32')
        self.distance_feature[~self.has_boat.astype(bool)] = 0

    def stream(self, transform=True, batch_size=64, shuffle=True):
        inds = np.array(list(range(len(self.wav_paths))))
        if shuffle:
            np.random.shuffle(inds)
        batch_len = 0
        batch = {
            "audio": [],
            "closest_boat": [],
            "has_boat": []
        }
        for p, i in enumerate(inds):
            audio_data = torch.from_numpy(read_wav(self.wav_paths[i])[1]).float().to(self.device)
            # choose a random 5s
            audio_start = np.random.randint(0, len(audio_data) - 5 * 8000)
            audio_data = audio_data[audio_start:audio_start + 5 * 8000]
            wav_fname = os.path.split(self.wav_paths[i])[-1]
            mask = np.asarray(self.df["filepath"] == wav_fname)
            batch["closest_boat"].append(self.distance_feature[mask][0])
            batch["has_boat"].append(self.has_boat[mask][0])

            batch["audio"].append(audio_data)
            batch_len += 1
            if batch_len == batch_size or p == len(inds) - 1:
                batch_out = dict(**batch)
                batch_out["audio"] = torch.stack(batch_out["audio"], dim=0)
                for k in ["closest_boat", "has_boat"]:
                    batch_out[k] = torch.from_numpy(np.asarray(batch_out[k])).float().to(self.device)
                if transform:
                    batch_out["audio"] = self.audio_to_spectrogram(batch_out["audio"])
                    batch_out["closest_boat"] = 2 * batch_out["closest_boat"] / 100 - 1
                batch = {
                    "audio": [],
                    "closest_boat": [],
                    "has_boat": []
                }
                batch_len = 0
                yield batch_out


class Encoder(nn.Module):
    def __init__(self, d=64):
        super(Encoder, self).__init__()
        c2d = partial(nn.Conv2d, stride=(2, 2), padding=1)
        self.has_boat_embedding = nn.Sequential(
                nn.Embedding(1, 256),
                nn.Unflatten(1, (1, 16, 16)),
                nn.Upsample(scale_factor=32),
                nn.Tanh()
        )
        self.layers = nn.Sequential(
            c2d(len(ATTRIBUTE_DIMS) + 1, d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(d, 2 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(2 * d, 4 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(4 * d, 8 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(8 * d, 16 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(16 * d, 32 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(32 * d, 64 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(64 * d, LATENT_DIM, (5, 5))
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X, a):
        has_boat = self.has_boat_embedding(a["has_boat"].reshape((-1,)).int())
        closest_boat = continuous_feature_map(a["has_boat"].reshape((-1, 1)))
        X = X.reshape((-1, 1, 512, 512))
        return self.layers(torch.concat([X, has_boat, closest_boat], dim=1))
