import os

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from functools import partial
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
import librosa
from scipy.io.wavfile import read as read_wav
from io import BytesIO
from .training_utils import batchify
import torchaudio
from zipfile import ZipFile
import json

np.random.seed(42)
VALIDATION_RUNS = [38, 7, 42, 10, 14, 18, 20, 22, 28]


class AudioMNISTClassifier(nn.Sequential):
    def __init__(self, num_classes: int = 10):
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
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_classes)
        )


class AudioMNISTData:
    def __init__(self, path_to_zip: str, device="cpu"):
        self.path_to_zip = path_to_zip
        self.device = device
        self.data = {
            "audio": [],
            "country_of_origin": [],
            "native_speaker": [],
            "accent": [],
            "digit": [],
            "age": [],
            "gender": [],
            "subject": [],
            "run": []
        }
        self.transforms = {k: lambda x: x for k in self.data}
        self.inv_transforms = {k: lambda x: x for k in self.data}

        self.audio_to_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=255, win_length=128, pad=96
        ).to(self.device)
        self.spectrogram_to_audio = torchaudio.transforms.GriffinLim(
            n_fft=255, win_length=128
        ).to(self.device)

        with ZipFile(self.path_to_zip, "r") as zf:
            json_str = zf.read("data/audioMNIST_meta.txt").decode("utf-8")
            meta_data = json.loads(json_str)
            for subject_num in range(1, 61):
                subject_name = f"0{subject_num}"[-2:]
                subject_meta = meta_data[subject_name]

                for dig in range(0, 10):
                    for run in range(0, 50):
                        wav_path = f"data/{subject_name}/{dig}_{subject_name}_{run}.wav"
                        sr, wav_arr = read_wav(BytesIO(zf.read(wav_path)))
                        wav_arr = librosa.core.resample(y=wav_arr.astype(np.float32),
                                                        orig_sr=sr, target_sr=8000,
                                                        res_type="scipy")
                        # zero padding
                        if len(wav_arr) > 8000:
                            raise ValueError("data length cannot exceed padding length.")
                        elif len(wav_arr) < 8000:
                            embedded_data = np.zeros(8000)
                            embedded_data[:len(wav_arr)] = wav_arr
                        elif len(wav_arr) == 8000:
                            # nothing to do here
                            embedded_data = wav_arr

                        self.data["audio"].append(embedded_data)

                        country = subject_meta["origin"].split(", ")[1].lower()
                        if country == "spanien":
                            country = "spain"

                        native_speaker = subject_meta["native speaker"]
                        accent = subject_meta["accent"].lower()
                        if accent == "german/spanish":
                            accent = "german"

                        age = int(subject_meta["age"])
                        if age > 100:  # error in data
                            age = 28
                        gender = subject_meta["gender"]

                        self.data["country_of_origin"].append(country)
                        self.data["native_speaker"].append(native_speaker)
                        self.data["accent"].append(accent)
                        self.data["digit"].append(dig)
                        self.data["age"].append(age)
                        self.data["gender"].append(gender)
                        self.data["subject"].append(subject_num)
                        self.data["run"].append(run)

            self.data["audio"] = np.stack(self.data["audio"], axis=0)
            self.transforms["audio"] = lambda x: (
                    self.audio_to_spectrogram(torch.from_numpy(x).float().to(self.device)) + 1e-6).log()
            self.inv_transforms["audio"] = lambda x: self.spectrogram_to_audio(
                torch.from_numpy(x).to(self.device).exp())

            for k in self.data:
                self.data[k] = np.asarray(self.data[k])
                if self.data[k].ndim == 1:
                    self.data[k] = self.data[k].reshape((-1, 1))

            for feature in ["country_of_origin",
                            "accent", "digit",
                            "native_speaker", "gender"]:
                one_hot = OneHotEncoder(sparse=False).fit(self.data[feature])

                def transform(x, oh=None):
                    return torch.from_numpy(oh.transform(x)).to(self.device)

                def inv_transform(x, oh=None):
                    return oh.inverse_transform(x)

                self.transforms[feature] = partial(transform, oh=one_hot)
                self.inv_transforms[feature] = partial(inv_transform, oh=one_hot)

            discretizer = KBinsDiscretizer(encode="onehot-dense",
                                           strategy="uniform")
            discretizer.fit(self.data["age"])
            self.transforms["age"] = lambda x: torch.from_numpy(discretizer.transform(x)).to(self.device)
            self.inv_transforms["age"] = lambda x: discretizer.inverse_transform(x)

    def stream(self,
               batch_size: int = 128,
               transform: bool = True,
               shuffle: bool = True,
               excluded_runs=None,
               excluded_subjects=None):
        excluded_runs = np.array(excluded_runs or [])
        excluded_subjects = np.array(excluded_subjects or [])
        data_to_use = {
            k: v[~np.isin(self.data["run"].flatten(), excluded_runs) &
                 ~np.isin(self.data["subject"].flatten(), excluded_subjects)]
            for k, v in self.data.items()
        }
        N = len(data_to_use["audio"])
        i = 0
        inds = np.random.permutation(N) if shuffle else np.array(list(range(N)))
        while i < N:
            batch_dict = {
                k: data_to_use[k][inds[i:min(N, i + batch_size)]]
                for k in data_to_use
            }
            if transform:
                batch_dict = {
                    k: self.transforms[k](v)
                    for k, v in batch_dict.items()
                }
            yield batch_dict
            i += batch_size


ATTRIBUTE_DIMS = {
    "country_of_origin": 13,
    "native_speaker": 2,
    "accent": 15,
    "digit": 10,
    "age": 5,
    "gender": 2
}


def evaluate(zip_path: str,
             model_path: str,
             attribute: str = "digit",
             stats_prefix: str = None,
             batch_size: int = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = AudioMNISTData(zip_path, device=device)
    model = torch.load(model_path, map_location=device)['model']

    if stats_prefix is not None:
        print('loading')
        spect_mean = torch.from_numpy(
            np.load(stats_prefix + '_mean.npy')
        ).float().to(device)
        spect_std = torch.from_numpy(
            np.load(stats_prefix + '_std.npy')
        ).float().to(device)
    else:
        spect_mean, spect_ss, n_batches = 0, 0, 0
        print('Computing spectrogram statistics...')
        for batch in data.stream(batch_size=batch_size,
                                 excluded_runs=VALIDATION_RUNS):
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

    n_correct = 0
    n_total = 0
    print('running clf')
    with torch.no_grad():
        for batch in tqdm(data.stream(batch_size=batch_size,
                                      excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS)))):
            labels = batch[attribute]
            x = spect_to_img(batch["audio"]).reshape((-1, 1, 128, 128))
            n_total += len(x)
            n_correct += (labels.argmax(1) == model(x).argmax(1)).sum().cpu().item()

    return n_correct / n_total


def train(zip_path: str,
          epochs: int = 100,
          batch_size: int = 100,
          attribute: str = "digit"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading data...")
    data = AudioMNISTData(zip_path, device=device)
    if attribute == "subject":
        model = AudioMNISTClassifier(60).to(device)
    else:
        model = AudioMNISTClassifier(ATTRIBUTE_DIMS[attribute]).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    spect_mean, spect_ss, n_batches = 0, 0, 0
    print('Computing spectrogram statistics...')
    for batch in data.stream(batch_size=batch_size,
                             excluded_runs=VALIDATION_RUNS):
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

    for e in range(epochs):
        tq = tqdm(data.stream(batch_size=batch_size,
                              excluded_runs=VALIDATION_RUNS), total=n_batches)
        for batch in tq:
            if attribute == "subject":
                batch[attribute] = torch.eye(60)[batch[attribute] - 1].to(device).reshape((-1, 60))
            opt.zero_grad()
            pred = model(spect_to_img(batch["audio"].reshape((-1, 1, 128, 128))))
            loss = criterion(pred, batch[attribute])
            loss.backward()
            opt.step()

            pred = pred.argmax(dim=1)
            y = batch[attribute].argmax(dim=1)
            acc = torch.eq(pred, y).float().mean()
            tq.set_postfix(dict(loss=loss.item(), acc=acc.item()))
        valid_correct = 0
        n_valid = 0
        with torch.no_grad():
            for batch in data.stream(batch_size=batch_size,
                                     excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS))):
                if attribute == "subject":
                    batch[attribute] = torch.eye(60)[batch[attribute] - 1].to(device).reshape((-1, 60))
                pred = model(spect_to_img(batch["audio"].reshape((-1, 1, 128, 128))))
                pred = pred.argmax(dim=1)
                y = batch[attribute].argmax(dim=1)
                valid_correct += torch.eq(pred, y).float().sum().item()
                n_valid += len(y)

        print(f"Epoch {e + 1}/{epochs} complete. Validation accuracy = {round(valid_correct / n_valid, 4)}")
    return model
