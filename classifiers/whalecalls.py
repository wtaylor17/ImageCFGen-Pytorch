import torch
import torch.nn as nn
import numpy as np
from .training_utils import batchify
import os
from pathlib import Path
import torchaudio
from scipy.io.wavfile import read as read_wav
from scipy.io import loadmat
from scipy import signal
from functools import partial
from tqdm import tqdm

ATTRIBUTE_DIMS = {
    "call_type": 3,
    "path": 1,
    "time": 2
}
IMAGE_SHAPE = (256, 256)
LATENT_DIM = 512


def init_weights(layer, std=0.001):
    name = layer.__class__.__name__
    if name.startswith('Conv'):
        torch.nn.init.normal_(layer.weight, mean=0, std=std)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


class WhaleCallData:
    def __init__(self,
                 nocall_directory: str,
                 shotgun_directory: str,
                 upcall_directory: str,
                 device="cpu",
                 validation_split=0.2, seed=42,
                 filter_length=None,
                 min_upcall_snr=-2.0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.filter_length = filter_length

        self.device = device
        self.audio_to_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=511, win_length=128, hop_length=24, pad=64
        ).to(self.device)
        self.audio_to_image = lambda x: (self.audio_to_spectrogram(x) + 1e-6).log()
        self.spectrogram_to_audio = torchaudio.transforms.GriffinLim(
            n_fft=511, win_length=128, hop_length=24,
        ).to(self.device)
        self.image_to_audio = lambda x: self.spectrogram_to_audio(x.exp())
        self.min_upcall_snr = min_upcall_snr

        self.shotgun_call_times = {}
        shotgun_log_paths = list(map(str, Path(shotgun_directory).rglob("*.mat")))
        for path in shotgun_log_paths:
            _, fname = os.path.split(path)
            date = fname.split('_')[1]
            event = loadmat(path)[f'Log_{fname[:-4]}']['event']
            times = event[0, 0]['time'][0].tolist()
            tags = event[0, 0]['tags'][0].tolist()
            self.shotgun_call_times[date] = np.asarray(
                [t for t, tag in zip(times, tags)
                 if len(tag) == 0]
            ).reshape((-1, 2))

        self.upcall_call_times = {}
        upcall_log_paths = list(map(str, Path(upcall_directory).rglob("*.mat")))
        for path in upcall_log_paths:
            _, fname = os.path.split(path)
            date = fname.split('_')[1]
            event = loadmat(path)[f'Log_{fname[:-4]}']['event']
            times = event[0, 0]['time'][0].tolist()
            tags = event[0, 0]['tags'][0].tolist()
            self.upcall_call_times[date] = np.asarray(
                [t for t, tag in zip(times, tags)
                 if len(tag) == 0]
            ).reshape((-1, 2))

        self.shotgun_wav_paths = list(map(str, Path(shotgun_directory).rglob("*.wav")))
        self.upcall_wav_paths = list(map(str, Path(upcall_directory).rglob("*.wav")))

        n_train_shotgun = int(len(self.shotgun_wav_paths) * (1 - validation_split))
        inds = np.random.permutation(len(self.shotgun_wav_paths))
        self.shotgun_train_paths = [self.shotgun_wav_paths[i]
                                    for i in inds[:n_train_shotgun]]
        self.shotgun_validation_paths = [self.shotgun_wav_paths[i]
                                         for i in inds[n_train_shotgun:]]

        n_train_upcall = int(len(self.upcall_wav_paths) * (1 - validation_split))
        inds = np.random.permutation(len(self.upcall_wav_paths))
        self.upcall_train_paths = [self.upcall_wav_paths[i]
                                   for i in inds[:n_train_upcall]]
        self.upcall_validation_paths = [self.upcall_wav_paths[i]
                                        for i in inds[n_train_upcall:]]

        self.nocall_wav_paths = list(map(str, Path(nocall_directory).rglob("*.wav")))
        n_train_nocall = int(len(self.nocall_wav_paths) * (1 - validation_split))
        inds = np.random.permutation(len(self.nocall_wav_paths))
        self.nocall_train_paths = [self.nocall_wav_paths[i]
                                   for i in inds[:n_train_nocall]]
        self.nocall_validation_paths = [self.nocall_wav_paths[i]
                                        for i in inds[n_train_nocall:]]

    def get_times_for_upcall(self, wav_path: str):
        parent_dir = os.path.dirname(wav_path)
        date = parent_dir.split('_')[-1]
        times_for_date = self.upcall_call_times[date]
        wav_start_time = wav_path.split('_')[-1][:-4]
        hrs, mins = int(wav_start_time[:2]), int(wav_start_time[2:4])
        lower_seconds = 3600 * hrs + 60 * mins
        upper_seconds = lower_seconds + 15 * 60

        return [(s - lower_seconds, e - lower_seconds)
                for (s, e) in times_for_date
                if lower_seconds <= s < upper_seconds]

    def get_times_for_shotgun(self, wav_path: str):
        parent_dir = os.path.dirname(wav_path)
        date = parent_dir.split('_')[-1]
        times_for_date = self.shotgun_call_times[date]
        wav_start_time = wav_path.split('_')[-1][:-4]
        hrs, mins = int(wav_start_time[:2]), int(wav_start_time[2:4])
        lower_seconds = 3600 * hrs + 60 * mins
        upper_seconds = lower_seconds + 15 * 60

        return [(s - lower_seconds, e - lower_seconds)
                for (s, e) in times_for_date
                if lower_seconds <= s < upper_seconds]

    def get_times_for_nocall(self, wav_path: str):
        return [(i, i + 3)
                for i in range(1, 11)]

    def stream(self, transform=True, batch_size=64, shuffle=True, mode='train'):
        paths = (self.nocall_train_paths,
                 self.shotgun_train_paths,
                 self.upcall_train_paths) \
            if mode == 'train' \
            else (self.nocall_validation_paths,
                  self.shotgun_validation_paths,
                  self.upcall_validation_paths)

        time_getters = [self.get_times_for_nocall,
                        self.get_times_for_shotgun,
                        self.get_times_for_upcall]

        batch_len = 0
        batch = {
            k: []
            for k in ATTRIBUTE_DIMS
        }
        batch["audio"] = []

        times = [[getter(e) for e in p]
                 for p, getter in zip(paths, time_getters)]
        call_type = [np.zeros((len(p), len(paths)))
                     for p in paths]
        for i, b in enumerate(call_type):
            b[:, i] = 1

        paths = sum(paths, [])
        times = sum(times, [])
        call_type = np.concatenate(call_type, axis=0)

        inds = np.asarray(range(len(times)))
        if shuffle:
            np.random.shuffle(inds)

        for p, i in enumerate(inds):
            sr, audio_data = read_wav(paths[i])

            for t0, t1 in times[i]:
                pad = max(0.0, (3 - (t1 - t0)) / 2)
                start, end = t0 - pad, t1 + pad
                start = max(0, int(sr * start))
                end = min(len(audio_data), int(sr * end))
                a = audio_data[start:end]
                snr = signaltonoise(a).max()
                if call_type[i].argmax() == 2 and snr < self.min_upcall_snr:
                    continue
                if self.filter_length:
                    a = signal.lfilter([1.0 / self.filter_length] * self.filter_length, 1.0, a)

                batch["audio"].append(a)
                batch["path"].append(paths[i])
                batch["time"].append([t0, t1])
                if len(batch["audio"][-1]) < 3 * sr:
                    batch["audio"][-1] = np.concatenate([
                        batch["audio"][-1],
                        np.zeros((3 * sr - len(batch["audio"][-1]),))
                    ], axis=0)
                elif len(batch["audio"][-1]) > 3 * sr:
                    batch["audio"][-1] = batch["audio"][-1][:3 * sr]

                assert len(batch["audio"][-1]) == 3 * sr, (t0, t1 - t0, paths[i])
                batch["call_type"].append(call_type[i])
                batch_len += 1

            if batch_len >= batch_size or (p == len(inds) - 1 and batch_len > 0):
                batch_out = dict(**batch)
                batch_out["audio"] = torch.stack([torch.from_numpy(v)
                                                  for v in batch_out["audio"]],
                                                 dim=0).float().to(self.device)
                for k in ATTRIBUTE_DIMS:
                    if k not in ["path", "time"]:
                        batch_out[k] = torch.from_numpy(np.asarray(batch_out[k])).float().to(self.device)
                if transform:
                    batch_out["audio"] = self.audio_to_image(batch_out["audio"])
                yield batch_out
                del batch_out
                del batch
                torch.cuda.empty_cache()
                batch = {
                    k: []
                    for k in ATTRIBUTE_DIMS
                }
                batch["audio"] = []
                batch_len = 0


class NARWClassifier(nn.Sequential):
    def __init__(self, num_classes: int = 3):
        super().__init__(
            nn.Conv2d(1, 32, (3, 3)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 3)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_classes)
        )


def train(nocall_directory,
          gunshot_directory,
          upcall_directory,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          batch_size=32,
          filter_length=None):
    print('Loading dataset...')
    data = WhaleCallData(nocall_directory,
                         gunshot_directory,
                         upcall_directory,
                         device=device,
                         filter_length=filter_length)
    print('Done')

    spect_mean, spect_ss, n_batches = 0, 0, 0

    print('Computing spectrogram statistics...')
    for batch in data.stream(batch_size=batch_size):
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

    clf = NARWClassifier().to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=l_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        tq = tqdm(data.stream(batch_size=batch_size), total=n_batches)
        for batch in tq:
            optimizer.zero_grad()
            img = spect_to_img(batch["audio"])
            pred = clf(img.reshape((-1, 1, 256, 256)))
            loss = criterion(pred, batch["call_type"].argmax(1))
            loss.backward()
            optimizer.step()

            pred = pred.argmax(dim=1)
            y = batch["call_type"].argmax(dim=1)
            acc = torch.eq(pred, y).float().mean()
            tq.set_postfix(dict(loss=loss.item(), acc=acc.item()))
        valid_correct = 0
        n_valid = 0
        with torch.no_grad():
            for batch in data.stream(batch_size=batch_size):
                pred = clf(spect_to_img(batch["audio"].reshape((-1, 1, 256, 256))))
                pred = pred.argmax(dim=1)
                y = batch["call_type"].argmax(dim=1)
                valid_correct += torch.eq(pred, y).float().sum().item()
                n_valid += len(y)
        print(f"Epoch {epoch + 1}/{n_epochs} complete. Validation accuracy = {round(valid_correct / n_valid, 4)}")

    return clf
