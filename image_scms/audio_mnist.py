import torch
import torch.nn as nn
import numpy as np
import json
from io import BytesIO
from zipfile import ZipFile
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from tqdm import tqdm
import torchaudio
import librosa
from scipy.io.wavfile import read as read_wav, write as write_wav
from functools import partial

from .training_utils import (attributes_image,
                             init_weights,
                             AdversariallyLearnedInference,
                             binarized_attribute_channel)


LATENT_DIM = 1024
IMAGE_SHAPE = (128, 128)


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
            "gender": []
        }
        self.transforms = {k: lambda x: x for k in self.data}
        self.inv_transforms = {k: lambda x: x for k in self.data}

        self.audio_to_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=255, win_length=255, hop_length=63, pad=32
        ).to(self.device)
        self.spectrogram_to_audio = torchaudio.transforms.GriffinLim(
            n_fft=255, win_length=255, hop_length=63
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
            self.transforms["audio"] = lambda x: (torch.transpose(self.audio_to_spectrogram(torch.from_numpy(x).float()
                                                                                                 .to(self.device)),
                                                                  dim0=1, dim1=2) + 1e-6).log()
            self.inv_transforms["audio"] = lambda x: self.spectrogram_to_audio(
                torch.from_numpy(np.transpose(x, axes=(0, 2, 1))).float().to(self.device).exp()
            )

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
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 16, (5, 5), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, LATENT_DIM, (1, 1), (1, 1))
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X, a):
        inp = torch.concat([X] + [
            binarized_attribute_channel(X, ai, device=self.device)
            for ai in a
        ], dim=1)
        return self.layers(inp)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(LATENT_DIM + 47),
            nn.ConvTranspose2d(LATENT_DIM + 47, 256, (5, 5), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, (5, 5), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, (3, 3), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, (2, 2), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, (1, 1), (1, 1)),
            nn.Tanh()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, z, a):
        return self.layers(torch.concat([z] + [
            binarized_attribute_channel(z, ai, device=self.device)
            for ai in a
        ], dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.5),
            nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dx = nn.Sequential(
            nn.Conv2d(47, 32, (5, 5), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (5, 5), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, LATENT_DIM, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1)
        )
        self.dxz = nn.Sequential(
            nn.Conv2d(2 * LATENT_DIM, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1, (1, 1), (1, 1)),
            nn.Sigmoid()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X, z, a):
        inp = torch.concat([X] + [
            binarized_attribute_channel(X, ai, device=self.device)
            for ai in a
        ], dim=1)
        dx = self.dx(inp)
        dz = self.dz(z)
        return self.dxz(torch.concat([dx, dz], dim=1)).reshape((-1, 1))


def train(path_to_zip: str,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          save_images_every=2,
          batch_size=128,
          image_output_path=''):
    stds_kept = 10
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

    print('Loading dataset...')
    data = AudioMNISTData(path_to_zip, device=device)
    print('Done')

    spect_mean, spect_ss, n_batches = 0, 0, 0

    print('Computing spectrogram statistics...')
    for batch in data.stream(batch_size=batch_size):
        n_batches += 1
        spect_mean = spect_mean + batch["audio"].mean(dim=(0, 1), keepdim=True)
        spect_ss = spect_ss + batch["audio"].square().mean(dim=(0, 1), keepdim=True)
    print('Done')

    spect_mean = (spect_mean / n_batches).float().to(device)
    spect_ss = (spect_ss / n_batches).float().to(device)

    spect_std = torch.sqrt(spect_ss - spect_mean)

    attr_cols = [k for k in data.data if k != "audio"]
    print('Beginning training')
    for epoch in range(n_epochs):
        D_score = 0.
        EG_score = 0.
        D.train()
        E.train()
        G.train()
        for i, batch in enumerate(tqdm(data.stream(batch_size=batch_size), total=n_batches)):
            images = batch["audio"].reshape((-1, 1, *IMAGE_SHAPE)).float().to(device)
            attrs = [batch[k] for k in attr_cols]
            c = [torch.clone(attr).float().to(device)
                 for attr in attrs]
            images = (images - spect_mean) / spect_std
            images = (torch.clip(images, -stds_kept, stds_kept) + stds_kept) / float(2*stds_kept)

            z_mean = torch.zeros((len(images), LATENT_DIM, 1, 1)).float()
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
                images = demo_batch["audio"].reshape((-1, 1, *IMAGE_SHAPE)).float().to(device)
                attrs = [demo_batch[k] for k in attr_cols]
                c = [torch.clone(attr).float().to(device)
                     for attr in attrs]
                x = (images - spect_mean) / spect_std
                x = (torch.clip(x, -stds_kept, stds_kept) + stds_kept) / float(2*stds_kept)

                z_mean = torch.zeros((len(x), LATENT_DIM, 1, 1)).float()
                z = torch.normal(z_mean, z_mean + 1)
                z = z.to(device)

                gener = G(z, c).reshape(n_show, *IMAGE_SHAPE)
                gener = ((gener * 2 * stds_kept - stds_kept) * spect_std + spect_mean).cpu().numpy()
                recon = G(E(x, c), c).reshape(n_show, *IMAGE_SHAPE)
                recon = ((recon * 2 * stds_kept - stds_kept) * spect_std + spect_mean).cpu().numpy()
                real = x.reshape((n_show, *IMAGE_SHAPE))
                real = ((real * 2 * stds_kept - stds_kept) * spect_std + spect_mean).cpu().numpy()
                vmin, vmax = real.min(), real.max()

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

                gener_wav = data.inv_transforms["audio"](gener[0:1]).cpu().numpy()[0]
                rec_wav = data.inv_transforms["audio"](recon[0:1]).cpu().numpy()[0]
                real_wav = data.inv_transforms["audio"](real[0:1]).cpu().numpy()[0]

                write_wav(f"{image_output_path}/epoch-{epoch + 1}-generated.wav", 8000,
                          np.int16(gener_wav / np.max(np.abs(gener_wav)) * 32767))
                write_wav(f"{image_output_path}/epoch-{epoch + 1}-real.wav", 8000,
                          np.int16(real_wav / np.max(np.abs(real_wav)) * 32767))
                write_wav(f"{image_output_path}/epoch-{epoch + 1}-reconstructed.wav", 8000,
                          np.int16(rec_wav / np.max(np.abs(rec_wav)) * 32767))

                print('Image and audio saved to', image_output_path)

    return E, G, D, optimizer_D, optimizer_E
