import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
import torchaudio
from scipy.io.wavfile import read as read_wav, write as write_wav
from functools import partial
from tqdm import tqdm


def continuous_feature_map(c: torch.Tensor, size: tuple = (512, 512)):
    return c.reshape((c.size(0), 1, 1, 1)).repeat(1, 1, *size)


ATTRIBUTE_DIMS = {
    "closest_boat": 1,
    "has_boat": 2
}
LATENT_DIM = 512


def init_weights(layer, std=0.001):
    name = layer.__class__.__name__
    if name.startswith('Conv'):
        torch.nn.init.normal_(layer.weight, mean=0, std=std)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)


class EsrfStation:
    def __init__(self, station_wav_path: str, station_label_csv: str, device="cpu",
                 validation_split=0.2, seed=42):
        self.device = device
        self.wav_paths = list(map(str, Path(station_wav_path).rglob("*.wav")))
        self.wav_paths = list(filter(lambda x: '8000' in x, self.wav_paths))
        np.random.seed(seed)
        inds = np.random.permutation(len(self.wav_paths))
        n_train = int(len(self.wav_paths) * (1 - validation_split))
        self.train_paths = [self.wav_paths[i]
                            for i in inds[:n_train]]
        self.validation_paths = [self.wav_paths[i]
                                 for i in inds[n_train:]]
        self.audio_to_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=1023, win_length=256, hop_length=79, pad=200
        ).to(self.device)
        self.audio_to_image = lambda x: (self.audio_to_spectrogram(x) + 1e-6).log()
        self.spectrogram_to_audio = torchaudio.transforms.GriffinLim(
            n_fft=1023, win_length=256, hop_length=79
        ).to(self.device)
        self.image_to_audio = lambda x: self.spectrogram_to_audio(x.exp())

        self.df = pd.read_csv(station_label_csv)
        self.df["filepath"] = self.df["filepath"].str.split('/').apply(lambda x: x[-1])
        bg_cols = [c for c in self.df.columns if c.startswith('BG')]

        def glb(e):
            if (e > 0).sum():
                return 100 - max(i for i in range(len(e)) if e[i] > 0)
            return -1
        self.distance_feature = np.array([glb(e) for e in np.asarray(self.df[bg_cols])])
        self.has_boat = (self.distance_feature > 0).astype(float)
        self.distance_feature[~self.has_boat.astype(bool)] = 0
        has_boat_2d = np.zeros((self.has_boat.shape[0], 2))
        has_boat_2d[self.has_boat == 0, 0] = 1
        has_boat_2d[self.has_boat == 1, 1] = 1
        self.has_boat = has_boat_2d

    def stream(self, transform=True, batch_size=64, shuffle=True, mode='train'):
        paths = self.train_paths if mode == 'train' else self.validation_paths
        inds = np.array(list(range(len(paths))))
        if shuffle:
            np.random.shuffle(inds)
        batch_len = 0
        batch = {
            "audio": [],
            "closest_boat": [],
            "has_boat": [],
            "start_idx": []
        }
        for p, i in enumerate(inds):
            wav_fname = os.path.split(paths[i])[-1]
            mask = np.asarray(self.df["filepath"] == wav_fname)
            closest_boat = self.distance_feature[mask][0]
            has_boat = self.has_boat[mask].reshape((2,))
            audio_data = read_wav(paths[i])[1][5*8000:]

            if np.argmax(has_boat) == 1:
                audio_start = np.random.randint(0, len(audio_data) - 5 * 8000, size=(4,))
            else:
                audio_start = np.random.randint(0, len(audio_data) - 5 * 8000, size=(1,))

            for idx in audio_start:
                batch["start_idx"].append(idx)
                batch["audio"].append(audio_data[idx: idx + 5 * 8000])
                batch["has_boat"].append(has_boat)
                batch["closest_boat"].append(closest_boat)
            batch_len += len(audio_start)

            if batch_len >= batch_size or p == len(inds) - 1:
                batch_out = dict(**batch)
                batch_out["audio"] = torch.stack([torch.from_numpy(v) for v in batch_out["audio"]],
                                                 dim=0).float().to(self.device)
                for k in ["closest_boat", "has_boat", "start_idx"]:
                    batch_out[k] = torch.from_numpy(np.asarray(batch_out[k])).float().to(self.device)
                if transform:
                    batch_out["audio"] = self.audio_to_image(batch_out["audio"])
                    batch_out["closest_boat"] = 2 * batch_out["closest_boat"] / 100 - 1
                yield batch_out
                del batch_out
                del batch
                torch.cuda.empty_cache()
                batch = {
                    "audio": [],
                    "closest_boat": [],
                    "has_boat": [],
                    "start_idx": []
                }
                batch_len = 0


class Encoder(nn.Module):
    def __init__(self, d=64):
        super(Encoder, self).__init__()
        c2d = partial(nn.Conv2d, stride=(2, 2), padding=1)
        self.has_boat_embedding = nn.Sequential(
                nn.Embedding(2, 256),
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
        has_boat = self.has_boat_embedding(a["has_boat"].argmax(1))
        closest_boat = continuous_feature_map(a["closest_boat"].reshape((-1, 1)))
        X = X.reshape((-1, 1, 512, 512))
        return self.layers(torch.concat([X, has_boat, closest_boat], dim=1))


class Generator(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        ct2d = partial(nn.ConvTranspose2d,
                       stride=2,
                       padding=2,
                       output_padding=1)
        self.has_boat_embedding = nn.Embedding(2, 256)
        self.layers = nn.Sequential(
            nn.Linear(LATENT_DIM + 257, 256 * d),
            nn.Unflatten(1, (16 * d, 4, 4)),
            nn.LeakyReLU(0.2),
            ct2d(16 * d, 16 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(16 * d, 8 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(8 * d, 4 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(4 * d, 2 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(2 * d, d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(d, d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(d, 1, (5, 5)),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, a):
        z = z.reshape((-1, LATENT_DIM))
        has_boat = a["has_boat"].matmul(self.has_boat_embedding.weight)
        closest_boat = a["closest_boat"].reshape((-1, 1))
        return self.layers(torch.concat([z, has_boat, closest_boat], dim=1))


class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        c2d = partial(nn.Conv2d, stride=(2, 2), padding=1)
        self.has_boat_embedding = nn.Sequential(
                nn.Embedding(2, 256),
                nn.Unflatten(1, (1, 16, 16)),
                nn.Upsample(scale_factor=32),
                nn.Tanh()
        )
        self.dx = nn.Sequential(
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
        self.dz = nn.Sequential(
            nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1), (1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.dxz = nn.Sequential(
            nn.Conv2d(2 * LATENT_DIM, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, (1, 1), (1, 1))
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X, z, a):
        X = X.reshape((-1, 1, 512, 512))
        z = z.reshape((-1, LATENT_DIM, 1, 1))
        has_boat = self.has_boat_embedding(a["has_boat"].argmax(1))
        closest_boat = continuous_feature_map(a["closest_boat"].reshape((-1, 1)))
        dx = self.dx(torch.concat([X, has_boat, closest_boat], dim=1))
        dz = self.dz(z)
        return self.dxz(torch.concat([dx, dz], dim=1)).reshape((-1, 1))


def train(path_to_wavs: str,
          path_to_labels: str,
          n_epochs: int = 200,
          l_rate: float = 1e-4,
          device: str = 'cpu',
          save_images_every: int = 2,
          batch_size: int = 64,
          image_output_path: str = '',
          validation_split=0.2,
          start_model_path=None):
    E = Encoder().to(device)
    G = Generator().to(device)
    D = Discriminator().to(device)
    E.apply(init_weights)
    G.apply(init_weights)
    D.apply(init_weights)

    if start_model_path is not None:
        model_dict = torch.load(start_model_path, map_location=device)
        E = model_dict['E']
        G = model_dict['G']
        D = model_dict['D']

    optimizer_E = torch.optim.Adam(list(E.parameters()) + list(G.parameters()),
                                   lr=l_rate, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(D.parameters(),
                                   lr=l_rate, betas=(0.5, 0.9))

    gan_loss = nn.BCEWithLogitsLoss()

    print('Loading dataset...')
    data = EsrfStation(path_to_wavs, path_to_labels,
                       device=device, validation_split=validation_split)
    print('Done')

    spect_mean, spect_ss, n_batches = 0, 0, 0

    print('Computing spectrogram statistics...')
    for batch in data.stream(batch_size=batch_size,
                             mode='train'):
        n_batches += 1
        spect_mean = spect_mean + batch["audio"].mean(dim=(0, 1)).reshape((1, 1, -1)).cpu().numpy()
        spect_ss = spect_ss + batch["audio"].square().mean(dim=(0, 1)).reshape((1, 1, -1)).cpu().numpy()

    spect_mean = spect_mean / n_batches  # E[X]
    spect_ss = spect_ss / n_batches  # E[X^2]
    spect_std = np.sqrt(spect_ss - np.square(spect_mean))

    spect_mean = torch.from_numpy(spect_mean).float().to(device)
    spect_std = torch.from_numpy(spect_std).float().to(device)

    print('Done.')

    stds_kept = 3

    def spect_to_img(spect_):
        spect_ = (spect_ - spect_mean) / (spect_std + 1e-6)
        return torch.clip(spect_, -stds_kept, stds_kept) / float(stds_kept)

    def img_to_spect(img_):
        return img_ * stds_kept * (spect_std + 1e-6) + spect_mean

    attr_cols = list(ATTRIBUTE_DIMS.keys())
    print('Beginning training')
    for epoch in range(n_epochs):
        D_score = 0.
        EG_score = 0.
        D.train()
        E.train()
        G.train()
        for i, batch in enumerate(tqdm(data.stream(batch_size=batch_size,
                                                   mode='train'),
                                       total=n_batches)):
            images = batch["audio"].reshape((-1, 1, 512, 512)).float().to(device)
            c = {k: torch.clone(batch[k]).to(device)
                 for k in attr_cols if k in ATTRIBUTE_DIMS}
            images = spect_to_img(images)

            z_mean = torch.zeros((len(images), LATENT_DIM, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(device)

            valid = torch.autograd.Variable(
                torch.Tensor(images.size(0), 1).fill_(1.0).to(device),
                requires_grad=False
            )
            fake = torch.autograd.Variable(
                torch.Tensor(images.size(0), 1).fill_(0.0).to(device),
                requires_grad=False
            )

            # Encoder & Generator training
            optimizer_E.zero_grad()
            D_valid = D(images, E(images, c), c)
            D_fake = D(G(z, c), z, c)
            loss_EG = (gan_loss(D_valid, fake) + gan_loss(D_fake, valid)) / 2
            loss_EG.backward()
            optimizer_E.step()

            optimizer_D.zero_grad()
            D_valid = D(images, E(images, c), c)
            loss_D = gan_loss(D_valid, valid)
            loss_D.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()
            D_fake = D(G(z, c), z, c)
            loss_D = gan_loss(D_fake, fake)
            loss_D.backward()
            optimizer_D.step()

            Gz = G(z, c).detach()
            EX = E(images, c).detach()
            DG = D(Gz, z, c).sigmoid()
            DE = D(images, EX, c).sigmoid()
            D_score += DG.mean().item()
            EG_score += DE.mean().item()
            torch.cuda.empty_cache()
            del batch

        print(D_score / n_batches, EG_score / n_batches)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            n_show = 4
            D.eval()
            E.eval()
            G.eval()

            with torch.no_grad():
                # generate images from same class as real ones
                demo_batch = next(data.stream(batch_size=n_show,
                                              mode='validation'))
                images = demo_batch["audio"][:n_show].reshape((-1, 1, 512, 512)).float().to(device)
                c = {k: torch.clone(demo_batch[k][:n_show]).float().to(device)
                     for k in attr_cols if k in ATTRIBUTE_DIMS}
                x = spect_to_img(images)

                z_mean = torch.zeros((n_show, LATENT_DIM, 1, 1)).float()
                z = torch.normal(z_mean, z_mean + 1)
                z = z.to(device)

                gener = img_to_spect(G(z, c).reshape(n_show, 512, 512)).cpu().numpy()
                recon = img_to_spect(G(E(x, c), c).reshape(n_show, 512, 512)).cpu().numpy()
                real = img_to_spect(x.reshape((n_show, 512, 512))).cpu().numpy()
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

                gener_wav = data.image_to_audio(
                    torch.from_numpy(gener[0:1]).to(device)
                ).cpu().numpy()[0]
                rec_wav = data.image_to_audio(
                    torch.from_numpy(recon[0:1]).to(device)
                ).cpu().numpy()[0]
                real_wav = data.image_to_audio(
                    torch.from_numpy(real[0:1]).to(device)
                ).cpu().numpy()[0]

                write_wav(f"{image_output_path}/epoch-{epoch + 1}-generated.wav", 8000,
                          np.int16(gener_wav / np.max(np.abs(gener_wav)) * 32767))
                write_wav(f"{image_output_path}/epoch-{epoch + 1}-real.wav", 8000,
                          np.int16(real_wav / np.max(np.abs(real_wav)) * 32767))
                write_wav(f"{image_output_path}/epoch-{epoch + 1}-reconstructed.wav", 8000,
                          np.int16(rec_wav / np.max(np.abs(rec_wav)) * 32767))

                print('Image and audio saved to', image_output_path)

    return E, G, D, optimizer_D, optimizer_E
