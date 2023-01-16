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
VALIDATION_RUNS = np.random.randint(0, 50, size=(10,)).tolist()


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
            "country_of_origin": [],
            "native_speaker": [],
            "accent": [],
            "digit": [],
            "age": [],
            "gender": []
        }
        self.transforms = {k: lambda x: x for k in self.data}
        self.inv_transforms = {k: lambda x: x for k in self.data}

        with ZipFile(self.path_to_zip, "r") as zf:
            json_str = zf.read("data/audioMNIST_meta.txt").decode("utf-8")
            meta_data = json.loads(json_str)
            for subject_num in range(1, 61):
                subject_name = f"0{subject_num}"[-2:]
                subject_meta = meta_data[subject_name]

                for dig in range(0, 10):
                    for run in range(0, 50):
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
                    return torch.from_numpy(oh.inverse_transform(x)).to(self.device)
                self.transforms[feature] = partial(transform, oh=one_hot)
                self.inv_transforms[feature] = partial(inv_transform, oh=one_hot)

            discretizer = KBinsDiscretizer(encode="onehot-dense",
                                           strategy="uniform")
            discretizer.fit(self.data["age"])
            print("number of age bins: ", discretizer.n_bins)
            self.transforms["age"] = lambda x: torch.from_numpy(discretizer.transform(x)).to(self.device)
            self.inv_transforms["age"] = lambda x: torch.from_numpy(discretizer.inverse_transform(x)).to(self.device)

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


def train(zip_path: str,
          epochs: int = 100,
          batch_size: int = 100,
          attribute: str = "digit"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading data...")
    data = AudioMNISTData(zip_path, device=device)
    model = AudioMNISTClassifier(ATTRIBUTE_DIMS[attribute]).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_batches = 0
    for _ in data.stream(batch_size=batch_size):
        n_batches += 1

    for e in range(epochs):
        tq = tqdm(data.stream(batch_size=batch_size), total=n_batches)
        for batch in tq:
            opt.zero_grad()
            pred = model(batch["audio"].reshape((-1, 1, 128, 128)))
            loss = criterion(pred, batch[attribute])
            loss.backward()
            opt.step()

            pred = pred.argmax(dim=1)
            y = batch[attribute].argmax(dim=1)
            acc = torch.eq(pred, y).float().mean()
            tq.set_postfix(dict(loss=loss.item(), acc=acc.item()))
        print(f"Epoch {e+1}/{epochs} complete.")
    return model
