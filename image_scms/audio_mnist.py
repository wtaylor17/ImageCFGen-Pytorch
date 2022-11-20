import torch
import torch.nn as nn
import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
import json
from io import BytesIO
from zipfile import ZipFile
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from tqdm import tqdm
import torchaudio
import librosa
from scipy.io.wavfile import read as read_wav
from functools import partial

from .training_utils import batchify, attributes_image, LambdaLayer, init_weights, AdversariallyLearnedInference


class AudioMNISTData:
    def __init__(self, path_to_zip: str, device="cpu"):
        self.path_to_zip = path_to_zip
        self.data = {
            "audio": [],
            "country_of_origin": [],
            "native_speaker": [],
            "accent": [],
            "digit": [],
            "age": [],
            "gender": []
        }
        self.transforms = {k: lambda x: x for k in self.data}
        self.inv_transforms = {k: lambda x: x for k in self.data}

        self.audio_to_spectrogram = torchaudio.transforms.Spectrogram(win_length=80).to(device)
        self.spectrogram_to_audio = torchaudio.transforms.GriffinLim(win_length=80).to(device)
        self.resample = torchaudio.transforms.Resample()
        self.device = device

        with ZipFile(self.path_to_zip, "r") as zf:
            json_str = zf.read("data/audioMNIST_meta.txt").decode("utf-8")
            meta_data = json.loads(json_str)
            for subject_num in range(1, 61):
                subject_name = f"0{subject_num}"[-2:]
                subject_meta = meta_data[subject_name]

                for dig in range(1, 10):
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
                            offset = np.random.randint(low=0, high=8000 - len(wav_arr))
                            embedded_data[offset:offset+len(wav_arr)] = wav_arr
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

            self.data["audio"] = np.stack(self.data["audio"], axis=0)
            self.transforms["audio"] = lambda x: self.audio_to_spectrogram(torch.from_numpy(x).to(self.device))
            self.inv_transforms["audio"] = lambda x: self.spectrogram_to_audio(torch.from_numpy(x).to(self.device))

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
            self.transforms["age"] = lambda x: torch.from_numpy(discretizer.transform(x)).to(self.device)
            self.inv_transforms["age"] = lambda x: torch.from_numpy(discretizer.inverse_transform(x)).to(self.device)

    def stream(self, batch_size: int = 128, transform: bool = True, shuffle: bool = True):
        N = len(self.data["audio"])
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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 16, (5, 5), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (2, 2), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (1, 1), (1, 1))
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X, a):
        return self.layers(attributes_image(X, a, device=self.device))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(512 + 47),
            nn.ConvTranspose2d(512 + 47, 512, (5, 5), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (3, 3), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, (3, 3), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, (3, 3), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, (3, 3), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, (1, 1), (1, 1)),
            LambdaLayer(lambda x: x[:, :, :201, :201]),
            nn.Sigmoid()
        )

    def forward(self, z, a):
        a = a.reshape((-1, 47, 1, 1))
        return self.layers(torch.concat([z, a], dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(512, 512, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 512, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dx = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(2, 16, (5, 5), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.Conv2d(16, 32, (5, 5), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 256, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 256, (3, 3), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 512, (3, 3), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 512, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dxz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1, (1, 1), (1, 1)),
            nn.Sigmoid()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X, z, a):
        dx = self.dx(attributes_image(X, a, device=self.device))
        dz = self.dz(z)
        return self.dxz(torch.concat([dx, dz], dim=1)).reshape((-1, 1))


def train(path_to_zip: str,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          save_images_every=2,
          image_output_path=''):
    E = Encoder().to(device)
    G = Generator().to(device)
    D = Discriminator().to(device)

    E.apply(init_weights)
    G.apply(init_weights)
    D.apply(init_weights)

    optimizer_E = torch.optim.Adam(list(E.parameters()) + list(G.parameters()),
                                   lr=l_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(),
                                   lr=l_rate, betas=(0.5, 0.999))

    loss_calc = AdversariallyLearnedInference(E, G, D)

    data = AudioMNISTData(path_to_zip, device=device)

    spect_mean, spect_ss, n_batches = 0, 0, 0

    for batch in data.stream():
        n_batches += 1
        spect_mean = spect_mean + batch["audio"].mean(dim=(0, 2), keepdim=True)
        spect_ss = spect_ss + batch["audio"].square().mean(dim=(0, 2), keepdim=True)

    spect_mean = spect_mean / n_batches
    spect_ss = spect_ss / n_batches

    spect_std = spect_ss - spect_mean

    attr_cols = [k for k in data.data if k != "audio"]

    for epoch in range(n_epochs):
        D_score = 0.
        EG_score = 0.
        D.train()
        E.train()
        G.train()

        vmin, vmax = float('inf'), -float('inf')
        for i, batch in tqdm(data.stream(), total=n_batches):
            images = batch["audio"].reshape((-1, 1, 201, 201)).float().to(device)
            attrs = torch.concat([batch[k] for k in attr_cols], dim=1)
            c = torch.clone(attrs.reshape((-1, 47))).float().to(device)
            images = (images - spect_mean) / spect_std
            vmin = min(vmin, images.min().item())
            vmax = max(vmax, images.max().item())

            z_mean = torch.zeros((len(images), 512, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(device)

            # Discriminator training
            optimizer_D.zero_grad()
            loss_D = loss_calc.discriminator_loss(images, z, c)
            loss_D.backward()
            optimizer_D.step()

            # Encoder & Generator training
            optimizer_E.zero_grad()
            loss_EG = loss_calc.generator_loss(images, z, c)
            loss_EG.backward()
            optimizer_E.step()

            Gz = G(z, c).detach()
            EX = E(images, c).detach()
            DG = D(Gz, z, c)
            DE = D(images, EX, c)
            D_score += DG.mean().item()
            EG_score += DE.mean().item()

        print(D_score / n_batches, EG_score / n_batches)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            n_show = 4
            D.eval()
            E.eval()
            G.eval()

            with torch.no_grad():
                # generate images from same class as real ones
                demo_batch = next(data.stream(batch_size=n_show))
                images = demo_batch["audio"].reshape((-1, 1, 201, 201)).float().to(device)
                attrs = torch.concat([demo_batch[k] for k in attr_cols], dim=1)
                c = torch.clone(attrs.reshape((-1, 47))).float().to(device)
                x = (images - spect_mean) / spect_std

                z_mean = torch.zeros((len(x), 512, 1, 1)).float()
                z = torch.normal(z_mean, z_mean + 1)
                z = z.to(device)

                gener = G(z, c).reshape(n_show, 28, 28).cpu().numpy()
                recon = G(E(x, c), c).reshape(n_show, 28, 28).cpu().numpy()
                real = x.cpu().numpy()

                if save_images_every is not None:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
                    fig.subplots_adjust(wspace=0.05, hspace=0)
                    plt.rcParams.update({'font.size': 20})
                    fig.suptitle('Epoch {}'.format(epoch + 1))
                    fig.text(0.01, 0.75, 'G(z, c)', ha='left')
                    fig.text(0.01, 0.5, 'x', ha='left')
                    fig.text(0.01, 0.25, 'G(E(x, c), c)', ha='left')

                    for i in range(n_show):
                        ax[0, i].imshow(gener[i], vmin=vmin, vmax=vmax)
                        ax[0, i].axis('off')
                        ax[1, i].imshow(real[i], vmin=vmin, vmax=vmax)
                        ax[1, i].axis('off')
                        ax[2, i].imshow(recon[i], vmin=vmin, vmax=vmax)
                        ax[2, i].axis('off')
                    plt.savefig(f'{image_output_path}/epoch-{epoch + 1}.png')
                    plt.close()

    return E, G, D, optimizer_D, optimizer_E
