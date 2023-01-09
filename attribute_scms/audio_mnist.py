import torch
import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
import json
from io import BytesIO
from zipfile import ZipFile
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from tqdm import tqdm
from functools import partial

from .training_utils import batchify
from .causal_module import *
from .graph import CausalModuleGraph


def gumbel_distribution():
    transforms = [
        T.ExpTransform().inv,
        T.AffineTransform(0, -1),
        T.ExpTransform().inv,
        T.AffineTransform(0, -1)
    ]
    base = dist.Uniform(0, 1)
    return dist.TransformedDistribution(base, transforms)


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


class ComboNet(torch.nn.Module):
    def __init__(self, downstream_model, *models):
        super().__init__()
        self.models = models
        self.downstream = downstream_model

    def forward(self, *inputs):
        features = torch.concat(
            [f(i) for f, i in zip(self.models, inputs)],
            dim=-1
        )
        out = self.downstream(features)
        return out


def categorical_mle(data_train: torch.Tensor, device="cpu"):
    data_train = data_train.to(device)
    values, counts = np.unique(data_train.detach().numpy(), return_counts=True)
    probs = np.zeros((data_train.max().item() + 1,))
    for v in values:
        probs[v] = counts[values == v]
    probs = torch.from_numpy(probs / data_train.size(0)).float().to(device)
    return CategoricalCM(dist.Categorical(probs=probs))


def conditional_categorical_mle(n_categories: int,
                                context_dim: int,
                                n_hidden_layers: int = 2,
                                device: str = "cpu"):
    nn = dense_net(context_dim, 128, n_categories,
                   n_hidden_layers=n_hidden_layers).to(device)
    return ConditionalCategoricalCM(nn, n_categories)


def conditional_double_categorical_mle(n_categories: int,
                                       context_dim1: int,
                                       context_dim2: int,
                                       device="cpu"):
    nn1 = dense_net(context_dim1, 64, 64,
                    n_hidden_layers=2).to(device)
    nn2 = dense_net(context_dim2, 64, 64,
                    n_hidden_layers=2).to(device)
    downstream = dense_net(128, 64, n_categories,
                           n_hidden_layers=0).to(device)
    combo = ComboNet(downstream, nn1, nn2)
    return ConditionalCategoricalCM(combo, n_categories)


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

    def stream(self, batch_size: int = 128, transform: bool = True, shuffle: bool = True):
        N = len(self.data["age"])
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


class AudioMNISTCausalGraph(CausalModuleGraph):
    def __init__(self, ds: dict, device: str = "cpu"):
        super().__init__()
        country_dist = categorical_mle(ds["country_of_origin"].argmax(dim=1), device=device)
        print("native speaker has", ds["native_speaker"].size(1), "categories")
        native_speaker_dist = conditional_categorical_mle(
            ds["native_speaker"].size(1),
            ds["country_of_origin"].size(1),
            device=device
        )
        for m in native_speaker_dist.model.modules():
            print(m)
        accent_dist = conditional_double_categorical_mle(
            ds["accent"].size(1),
            ds["country_of_origin"].size(1),
            ds["native_speaker"].size(1),
            device=device
        )
        digit_dist = categorical_mle(ds["digit"].argmax(dim=1), device=device)
        age_dist = categorical_mle(ds["age"].argmax(dim=1), device=device)
        gender_dist = categorical_mle(ds["gender"].argmax(dim=1), device=device)

        self.add_module("country_of_origin", country_dist)
        self.add_module("native_speaker", native_speaker_dist)
        self.add_module("accent", accent_dist)
        self.add_module("digit", digit_dist)
        self.add_module("age", age_dist)
        self.add_module("gender", gender_dist)
        self.add_edge("country_of_origin", "native_speaker")
        self.add_edge("country_of_origin", "accent")
        self.add_edge("native_speaker", "accent")


def train(path_to_zip: str,
          steps=2000,
          device='cpu',
          lr=1e-2):
    data = AudioMNISTData(path_to_zip)
    ds = list(data.stream(batch_size=30_000))[0]
    ds = {
        k: v.float().to(device)
        for k, v in ds.items()
        if k != "audio"
    }
    graph = AudioMNISTCausalGraph(ds, device=device)

    params = sum((list(graph.get_module(k).parameters())
                  for k in ["native_speaker", "accent"]), [])
    optimizer = torch.optim.Adam(params, lr=lr)

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
            lp = graph.log_prob({
                "country_of_origin": c,
                "native_speaker": n,
                "accent": a,
                "age": ag
            })
            loss = -(lp["native_speaker"] + lp["accent"]).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        tq.set_description(f'loss = {round(epoch_loss / len(batches), 4)}')

    return graph
