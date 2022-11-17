import torch
import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
import json
from io import BytesIO
from zipfile import ZipFile
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from tqdm import tqdm

from .training_utils import batchify, dist_parameters


def gumbel_distribution():
    transforms = [
        T.ExpTransform().inv,
        T.AffineTransform(0, -1),
        T.ExpTransform().inv,
        T.AffineTransform(0, -1)
    ]
    base = dist.Uniform(0, 1)
    return dist.TransformedDistribution(base, transforms)


class ConditionalCategorical(dist.ConditionalDistribution):
    def __init__(self, model: torch.nn.Module, n_categories: int):
        self.model = model
        self.gumbel = gumbel_distribution()
        self.n_categories = n_categories

    def condition(self, context):
        logits = self.model(context)
        return dist.Categorical(logits=logits)

    def forward(self, noise, context):
        # computes f(noise;context)
        logits = self.model(context)
        return torch.argmax(logits + noise, dim=-1)

    def noise_sample(self, y: torch.Tensor, context: torch.Tensor, device="cpu"):
        inds = list(range(y.size(0)))
        g = self.gumbel.sample(y.shape + (self.n_categories,)).to(device)
        gk = g[inds, y]
        logits = self.model(context)
        noise_k = gk + logits.exp().sum(dim=-1).log() - logits[inds, y]
        noise_l = -torch.log(torch.exp(-g - logits) +
                             torch.exp(-gk - logits[inds, y])) - logits
        noise_l[inds, y] = noise_k
        return y

    def counterfactual(self, y, original_context, cf_context, mc_rounds=1, device="cpu"):
        cf_logits = mc_rounds * self.model(cf_context)

        for _ in range(mc_rounds):
            cf_logits = cf_logits + self.noise_sample(y, original_context, device=device)
        cf_logits = cf_logits / mc_rounds

        return torch.argmax(cf_logits, dim=-1)


def dense_net(n_in: int,
              n_hidden: int,
              n_out: int,
              n_hidden_layers: int = 0,
              hidden_activation=torch.nn.ReLU()) -> torch.nn.Sequential:
    seq = torch.nn.Sequential()
    seq.append(torch.nn.Linear(n_in, n_hidden))
    seq.append(hidden_activation)
    for _ in range(n_hidden_layers):
        seq.append(torch.nn.Linear(n_hidden, n_hidden))
        seq.append(hidden_activation)
    seq.append(torch.nn.Linear(n_hidden, n_out))
    return seq


def categorical_mle(data_train: torch.Tensor, device="cpu"):
    data_train = data_train.to(device)
    values, counts = data_train.unique(return_counts=True)
    probs = counts / data_train.size(0)
    return dist.Categorical(probs=probs)


def conditional_categorical_mle(n_categories: int,
                                context_dim: int,
                                device="cpu"):
    nn = dense_net(context_dim, 128, n_categories,
                   n_hidden_layers=2).to(device)
    return ConditionalCategorical(nn, n_categories)


class AudioMNISTData:
    def __init__(self, path_to_zip: str):
        self.path_to_zip = path_to_zip
        self.data = {
            "spectrogram": [],
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
            json_str = zf.read("spectrograms/audioMNIST_meta.txt").decode("utf-8")
            meta_data = json.loads(json_str)
            for subject_num in range(1, 61):
                subject_name = f"0{subject_num}"[-2:]
                subject_meta = meta_data[subject_name]
                spect_name = f"spectrograms/{subject_name}.npy"
                spectrogram = np.load(BytesIO(zf.read(spect_name)))
                self.data["spectrogram"].append(spectrogram)

                country = subject_meta["origin"].split(", ")[1]
                native_speaker = subject_meta["native speaker"]
                accent = subject_meta["accent"]
                age = int(subject_meta["age"])
                if age > 100:  # error in data
                    age = 22
                gender = subject_meta["gender"]
                file_list = zf.read("spectrograms/01_files.txt").decode("utf-8").split("\n")
                digits = [file_name.split('/')[-1].split('_')[0]
                          for file_name in file_list]

                N = len(spectrogram)
                self.data["country_of_origin"].extend([country] * N)
                self.data["native_speaker"].extend([native_speaker] * N)
                self.data["accent"].extend([accent] * N)
                self.data["digit"].extend(digits)
                self.data["age"].extend([age] * N)
                self.data["gender"].extend([gender] * N)

            self.data["spectrogram"] = np.concatenate(self.data["spectrogram"], axis=0)
            mean = self.data["spectrogram"].mean(axis=(0, 1))
            std = self.data["spectrogram"].std(axis=(0, 1))
            self.transforms["spectrogram"] = lambda x: (x - mean) / std
            self.inv_transforms["spectrogram"] = lambda x: x * std + mean

            for k in self.data:
                self.data[k] = np.asarray(self.data[k])
                if self.data[k].ndim == 1:
                    self.data[k] = self.data[k].reshape((-1, 1))

            for feature in ["country_of_origin",
                            "accent", "digit"]:
                one_hot = OneHotEncoder(sparse=False).fit(self.data[feature])
                self.transforms[feature] = one_hot.transform
                self.inv_transforms[feature] = one_hot.inverse_transform

            def binary_transforms(v1, v2):
                def transform(x):
                    y = np.zeros_like(x, dtype=float)
                    y[x == v2] = 1
                    return y

                def inv_transform(y):
                    x = np.empty_like(y, dtype=object)
                    x[y == 0] = v1
                    x[y == 1] = v2
                    return x

                return transform, inv_transform

            self.transforms["gender"], \
                self.inv_transforms["gender"] = binary_transforms("female", "male")
            self.transforms["native_speaker"], \
                self.inv_transforms["native_speaker"] = binary_transforms("no", "yes")

            discretizer = KBinsDiscretizer()
            discretizer.fit(self.data["age"])
            self.transforms["age"] = discretizer.transform
            self.inv_transforms["age"] = discretizer.inverse_transform

    def stream(self, batch_size: int = 128, transform: bool = True, shuffle: bool = True):
        N = len(self.data["spectrogram"])
        i = 0
        inds = np.random.permutation(N) if shuffle else np.array(list(range(N)))
        while i < N:
            batch_dict = {
                k: self.data[k][inds[i:min(N, i + batch_size)]]
                for k in self.data
            }
            if transform:
                batch_dict = {
                    k: self.transforms[k](v)
                    for k, v in batch_dict.items()
                }
            yield batch_dict
            i += batch_size


def train(path_to_zip: str,
          steps=2000,
          device='cpu',
          lr=1e-2):
    data = AudioMNISTData(path_to_zip)
    ds = list(data.stream(batch_size=30_000))[0]
    ds = {
        k: torch.from_numpy(v).float().to(device)
        for k, v in ds.items()
    }
    country_dist = categorical_mle(ds["country_of_origin"].argmax(dim=1), device=device)
    native_speaker_dist = conditional_categorical_mle(
        ds["native_speaker"].size(1),
        ds["country_of_origin"].size(1),
        device=device
    )
    accent_dist = conditional_categorical_mle(
        ds["accent"].size(1),
        ds["country_of_origin"].size(1),
        device=device
    )
    digit_dist = categorical_mle(ds["digit"].argmax(dim=1), device=device)
    age_dist = categorical_mle(ds["age"].argmax(dim=1), device=device)
    gender_dist = categorical_mle(ds["gender"], device=device)

    optimizer = torch.optim.Adam(list(native_speaker_dist.model.parameters()) +
                                 list(accent_dist.model.parameters()) +
                                 dist_parameters(age_dist),
                                 lr=lr)

    country = ds["country_of_origin"]
    native_speaker = ds["native_speaker"]
    accent = ds["accent"]
    age = ds["age"]

    tq = tqdm(range(steps))
    for _ in tq:
        idx = np.random.permutation(country.size(0))
        batches = list(batchify(country[idx],
                                native_speaker[idx],
                                accent[idx],
                                age[idx],
                                batch_size=10_000))
        epoch_loss = 0
        for c, n, a, ag in batches:
            optimizer.zero_grad()
            loss = -(native_speaker_dist.condition(c).log_prob(n.argmax(dim=1)) +
                     accent_dist.condition(c).log_prob(a.argmax(dim=1)) +
                     age_dist.log_prob(ag)).mean()
            loss.backward()
            optimizer.step()
            age_dist.clear_cache()
            epoch_loss += loss.item()
        tq.set_description(f'loss = {round(epoch_loss / len(batches), 4)}')

    return {
        "country": country_dist,
        "native_speaker": native_speaker_dist,
        "accent": accent_dist,
        "digit": digit_dist,
        "age": age_dist,
        "gender": gender_dist,
        "optimizer": optimizer
    }
