import torch
import torch.nn as nn
import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions.conditional import ConditionalTransform
from tqdm import tqdm
from typing import Dict
from .training_utils import batchify, batchify_dict
import os
from pathlib import Path
import torchaudio
from scipy.io.wavfile import read as read_wav, write as write_wav
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


class VAEEncoder(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        c2d = partial(nn.Conv2d, stride=(2, 2), padding=1)
        self.embedding_dict = nn.ModuleDict({
            k: nn.Sequential(
                nn.Embedding(v, 256),
                nn.Unflatten(1, (1, 16, 16)),
                nn.Upsample(scale_factor=16),
                nn.Tanh()
            )
            for k, v in ATTRIBUTE_DIMS.items()
            if k not in ["time", "path"]
        })
        self.layers = nn.Sequential(
            c2d(2, d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(d, 2 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(2 * d, 4 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(4 * d, 8 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(8 * d, 16 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(16 * d, 16 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            c2d(16 * d, LATENT_DIM, (5, 5))
        )
        self.mean_linear = nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1))
        self.log_var_linear = nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1))

    def forward(self, X: torch.Tensor, a):
        embeddings = [
            self.embedding_dict[k](a[k].argmax(dim=1))
            for k in sorted(ATTRIBUTE_DIMS.keys())
            if k not in ["time", "path"]
        ]
        X = X.reshape((-1, 1, *IMAGE_SHAPE))
        upstream = self.layers(torch.concat([X, *embeddings], dim=1))

        return self.mean_linear(upstream), self.log_var_linear(upstream)

    def sample(self, x, c, device='cpu'):
        mean, log_var = self(x, c)
        var = torch.exp(log_var)
        return mean + torch.randn(mean.shape).to(device) * var


class VAEDecoder(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.digit_embedding = nn.Embedding(10, 256)
        ct2d = partial(nn.ConvTranspose2d,
                       stride=2,
                       padding=2,
                       output_padding=1)
        self.embedding_dict = nn.ModuleDict({
            k: nn.Embedding(v, 256)
            for k, v in ATTRIBUTE_DIMS.items()
            if k not in ["time", "path"]
        })
        self.layers = nn.Sequential(
            # nn.BatchNorm1d(LATENT_DIM + 256 * len(ATTRIBUTE_DIMS)),
            nn.Linear(LATENT_DIM + 256, 256 * d),
            nn.Unflatten(1, (16 * d, 4, 4)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(16 * d),
            ct2d(16 * d, 16 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(16 * d, 8 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(8 * d),
            ct2d(8 * d, 4 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(4 * d),
            ct2d(4 * d, 2 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(2 * d),
            ct2d(2 * d, d, (5, 5)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(d),
            ct2d(d, 1, (5, 5)),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, a):
        z = z.reshape((-1, LATENT_DIM))
        embeddings = [
            a[k].float().matmul(self.embedding_dict[k].weight)
            for k in sorted(ATTRIBUTE_DIMS.keys())
            if k not in ["time", "path"]
        ]
        return self.layers(torch.concat([z, *embeddings], dim=1))


class WhaleCallDecoderTransformation(ConditionalTransform):
    def __init__(self, decoder: nn.Module, log_var=-5,
                 device='cpu'):
        self.decoder = decoder
        self.scale = torch.exp(torch.ones((IMAGE_SHAPE[0] * IMAGE_SHAPE[1],)) * log_var / 2).to(device)

    def condition(self, context):
        bias = self.decoder(*context).reshape((-1, IMAGE_SHAPE[0] * IMAGE_SHAPE[1]))
        return T.AffineTransform(bias, self.scale)


class WhaleCallVAE(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.encoder = VAEEncoder().to(device)
        self.decoder = VAEDecoder().to(device)
        self.base = dist.Normal(torch.zeros((IMAGE_SHAPE[0]*IMAGE_SHAPE[1],)).to(device),
                                torch.ones((IMAGE_SHAPE[0]*IMAGE_SHAPE[1],)).to(device))
        self.dec_transform = WhaleCallDecoderTransformation(self.decoder,
                                                            device=device)

        self.dist = dist.ConditionalTransformedDistribution(self.base,
                                                            [self.dec_transform])

    def forward(self, x: torch.Tensor, c, num_samples=10):
        return self.elbo(x, c, num_samples=num_samples)

    def elbo(self, x: torch.Tensor, c, num_samples=4, device='cpu', kl_weight=1.0):
        z_mean, z_log_var = self.encoder(x, c)
        z_std = torch.exp(z_log_var * .5)
        lp = 0
        x_reshaped = x.reshape((-1, IMAGE_SHAPE[0] * IMAGE_SHAPE[1]))
        for _ in range(num_samples):
            z = z_mean + torch.randn(z_mean.shape).to(device) * z_std
            lp = lp + self.dist.condition((z, c)).log_prob(x_reshaped)
        lp = lp / num_samples
        dkl = .5 * (torch.square(z_std) +
                    torch.square(z_mean) -
                    1 - 2 * torch.log(z_std)).sum(dim=1)
        return (lp - kl_weight * dkl).mean()


def train(nocall_directory,
          gunshot_directory,
          upcall_directory,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          save_images_every=2,
          batch_size=32,
          image_output_path='',
          filter_length=None,
          num_samples_per_step=4,
          kl_weight=10):
    vae = WhaleCallVAE(device=device)
    vae.encoder.apply(init_weights)
    vae.decoder.apply(init_weights)
    optimizer = torch.optim.Adam(vae.parameters(),
                                 lr=l_rate)

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

    def img_to_spect(img_):
        return img_ * stds_kept * (spect_std + 1e-6) + spect_mean

    attr_cols = list(ATTRIBUTE_DIMS.keys())
    print('Beginning training')

    for epoch in range(n_epochs):
        epoch_elbo = 0
        vae.train()

        tq = tqdm(data.stream(batch_size=batch_size),
                  total=n_batches)
        for i, batch in enumerate(tq):
            images = batch["audio"].reshape((-1, 1, *IMAGE_SHAPE)).float().to(device)
            c = {k: torch.clone(batch[k]).int().to(device)
                 for k in attr_cols if k in ATTRIBUTE_DIMS
                 and k not in ["time", "path"]}
            images = spect_to_img(images)

            optimizer.zero_grad()
            elbo_loss = -vae.elbo(images, c,
                                  num_samples=num_samples_per_step,
                                  device=device,
                                  kl_weight=kl_weight)
            elbo_loss.backward()
            optimizer.step()
            epoch_elbo = epoch_elbo + elbo_loss.item()

        print(epoch_elbo / n_batches)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            n_show = 10
            vae.eval()

            with torch.no_grad():
                demo_batch = next(data.stream(batch_size=n_show,
                                              mode='validation'))
                images = demo_batch["audio"].reshape((-1, 1, *IMAGE_SHAPE)).float().to(device)
                c = {k: torch.clone(demo_batch[k]).int().to(device)
                     for k in attr_cols if k in ATTRIBUTE_DIMS
                     and k not in ["time", "path"]}
                x = spect_to_img(images)

                z_mean = torch.zeros((len(x), LATENT_DIM, 1, 1)).float()
                z = torch.normal(z_mean, z_mean + 1)
                z = z.to(device)

                gener = img_to_spect(vae.decoder(z, c).reshape(-1, *IMAGE_SHAPE)).cpu().numpy()
                recon = img_to_spect(vae.decoder(vae.encoder(x, c)[0], c).reshape(-1, *IMAGE_SHAPE))\
                                    .cpu().numpy()
                real = img_to_spect(x.reshape((-1, *IMAGE_SHAPE))).cpu().numpy()
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
                        ax[0, i].set_title(f'Call = {c["call_type"][i].argmax().item()}')
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

                    write_wav(f"{image_output_path}/epoch-{epoch + 1}-generated.wav", 2000,
                              np.int16(gener_wav / np.max(np.abs(gener_wav)) * 32767))
                    write_wav(f"{image_output_path}/epoch-{epoch + 1}-real.wav", 2000,
                              np.int16(real_wav / np.max(np.abs(real_wav)) * 32767))
                    write_wav(f"{image_output_path}/epoch-{epoch + 1}-reconstructed.wav", 2000,
                              np.int16(rec_wav / np.max(np.abs(rec_wav)) * 32767))

                print('Image and audio saved to', image_output_path)

    return vae, optimizer
