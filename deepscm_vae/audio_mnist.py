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
import pyro.distributions.transforms as T
from pyro.distributions.conditional import ConditionalTransform
import pyro.distributions as dist

np.random.seed(42)
VALIDATION_RUNS = np.random.randint(0, 50, size=(10,)).tolist()

LATENT_DIM = 512
ATTRIBUTE_COUNT = 47
IMAGE_SHAPE = (128, 128)
ATTRIBUTE_DIMS = {
    "country_of_origin": 13,
    "native_speaker": 2,
    "accent": 15,
    "digit": 10,
    "age": 5,
    "gender": 2
}


def init_weights(layer, std=0.001):
    name = layer.__class__.__name__
    if name.startswith('Conv'):
        torch.nn.init.normal_(layer.weight, mean=0, std=std)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)


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


class VAEEncoder(nn.Module):
    def __init__(self, d=64):
        super(VAEEncoder, self).__init__()
        c2d = partial(nn.Conv2d, stride=(2, 2), padding=1)
        self.embedding_dict = nn.ModuleDict({
            k: nn.Sequential(
                nn.Embedding(v, 256),
                nn.Unflatten(1, (1, 16, 16)),
                nn.Upsample(scale_factor=8),
                nn.Tanh()
            )
            for k, v in ATTRIBUTE_DIMS.items()
        })
        self.layers = nn.Sequential(
            # nn.BatchNorm2d(2),
            c2d(len(ATTRIBUTE_DIMS) + 1, d, (5, 5)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(d),
            c2d(d, 2 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(2 * d),
            c2d(2 * d, 4 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(4 * d),
            c2d(4 * d, 8 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(8 * d),
            c2d(8 * d, 16 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(16 * d),
            c2d(16 * d, LATENT_DIM, (5, 5)),
            nn.LeakyReLU(0.2)
        )
        self.mean = nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1),
                              stride=(1, 1),
                              padding="same")
        self.log_var = nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1),
                                 stride=(1, 1),
                                 padding="same")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X, a):
        embeddings = [
            self.embedding_dict[k](a[k].argmax(dim=1))
            for k in sorted(ATTRIBUTE_DIMS.keys())
        ]
        X = X.reshape((-1, 1, *IMAGE_SHAPE))
        feat = self.layers(torch.concat([X, *embeddings], dim=1))
        return self.mean(feat), self.log_var(feat)

    def sample(self, X, a):
        mean, log_var = self(X, a)
        var = torch.exp(log_var)
        return mean + torch.randn(mean.shape).to(self.device) * var


class VAEDecoder(nn.Module):
    def __init__(self, d=64):
        super(VAEDecoder, self).__init__()
        ct2d = partial(nn.ConvTranspose2d,
                       stride=2,
                       padding=2,
                       output_padding=1)
        self.embedding_dict = nn.ModuleDict({
            k: nn.Embedding(v, 256)
            for k, v in ATTRIBUTE_DIMS.items()
        })
        self.layers = nn.Sequential(
            # nn.BatchNorm1d(LATENT_DIM + 256 * len(ATTRIBUTE_DIMS)),
            nn.Linear(LATENT_DIM + 256 * len(ATTRIBUTE_DIMS), 256 * d),
            nn.Unflatten(1, (16 * d, 4, 4)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(16 * d),
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

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, z: torch.Tensor, a):
        z = z.reshape((-1, LATENT_DIM))
        embeddings = [
            a[k].matmul(self.embedding_dict[k].weight)
            for k in sorted(ATTRIBUTE_DIMS.keys())
        ]
        return self.layers(torch.concat([z, *embeddings], dim=1))


class DecoderTransformation(ConditionalTransform):
    def __init__(self, decoder: VAEDecoder, log_var=-5,
                 device='cpu'):
        self.decoder = decoder
        self.scale = torch.exp(torch.ones((128 * 128,)) * log_var / 2).to(device)

    def condition(self, context):
        bias = self.decoder(*context).reshape((-1, 128 * 128))
        return T.AffineTransform(bias, self.scale)


class VAE(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.encoder = VAEEncoder().to(device)
        self.decoder = VAEDecoder().to(device)
        self.base = dist.MultivariateNormal(torch.zeros((128 * 128,)).to(device),
                                            torch.eye(128 * 128).to(device))
        self.dec_transform = DecoderTransformation(self.decoder,
                                                   device=device)

        self.dist = dist.ConditionalTransformedDistribution(self.base,
                                                            [self.dec_transform])

    def forward(self, x, c, num_samples=4):
        return self.elbo(x, c, num_samples=num_samples)

    def elbo(self, x, c, num_samples=4, device='cpu', kl_weight=1.0):
        z_mean, z_log_var = self.encoder(x, c)
        z_std = torch.exp(z_log_var * .5)
        lp = 0
        x_reshaped = x.reshape((-1, 128 * 128))
        for _ in range(num_samples):
            z = z_mean + torch.randn(z_mean.shape).to(device) * z_std
            lp = lp + self.dist.condition((z, c)).log_prob(x_reshaped)
        lp = lp / num_samples
        dkl = .5 * (torch.square(z_std) +
                    torch.square(z_mean) -
                    1 - 2 * torch.log(z_std)).sum(dim=1)
        return (lp - kl_weight * dkl).mean()


def train(path_to_zip: str,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          save_images_every=2,
          batch_size=128,
          image_output_path='',
          num_samples_per_step=4,
          kl_weight=10):
    vae = VAE(device=device)
    vae.encoder.apply(init_weights)
    vae.decoder.apply(init_weights)

    optimizer = torch.optim.Adam(vae.parameters(),
                                 lr=l_rate)

    print('Loading dataset...')
    data = AudioMNISTData(path_to_zip, device=device)
    print('Done')

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

    attr_cols = [k for k in data.data if k in ATTRIBUTE_DIMS]
    print('Beginning training')
    for epoch in range(n_epochs):
        vae.train()
        epoch_elbo = 0
        for i, batch in enumerate(tqdm(data.stream(batch_size=batch_size,
                                                   excluded_runs=VALIDATION_RUNS),
                                       total=n_batches)):
            images = batch["audio"].reshape((-1, 1, *IMAGE_SHAPE)).float().to(device)
            c = {k: torch.clone(batch[k]).float().to(device)
                 for k in attr_cols if k in ATTRIBUTE_DIMS}
            images = spect_to_img(images)

            optimizer.zero_grad()
            elbo_loss = -vae.elbo(images, c,
                                  num_samples=num_samples_per_step,
                                  device=device,
                                  kl_weight=kl_weight)
            elbo_loss.backward()
            optimizer.step()
            epoch_elbo = epoch_elbo + elbo_loss.item()

        print(f'Epoch {epoch+1}/{n_epochs}:', epoch_elbo / n_batches)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            n_show = 4
            vae.eval()

            with torch.no_grad():
                # generate images from same class as real ones
                demo_batch = next(data.stream(batch_size=n_show,
                                              excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS))))
                images = demo_batch["audio"].reshape((-1, 1, *IMAGE_SHAPE)).float().to(device)
                c = {k: torch.clone(demo_batch[k]).int().to(device)
                     for k in attr_cols if k in ATTRIBUTE_DIMS}
                x = spect_to_img(images)

                z_mean = torch.zeros((len(x), LATENT_DIM, 1, 1)).float()
                gener = 0
                for i in range(32):
                    z = torch.normal(z_mean, z_mean + 1).to(device)
                    gener = gener + vae.decoder(z, c)
                gener = img_to_spect(gener / 32)
                gener = gener.cpu().detach().numpy().reshape((n_show, *IMAGE_SHAPE))

                recon = 0
                for i in range(32):
                    z = vae.encoder.sample(x, c)
                    recon = recon + vae.decoder(z, c)
                recon = img_to_spect(recon / 32)
                recon = recon.cpu().detach().numpy().reshape((n_show, *IMAGE_SHAPE))

                real = img_to_spect(x.reshape((n_show, *IMAGE_SHAPE))).cpu().numpy()
                vmin, vmax = real.min(), real.max()

            if save_images_every is not None:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
                fig.subplots_adjust(wspace=0.05, hspace=0)
                plt.rcParams.update({'font.size': 20})
                fig.suptitle('Epoch {}'.format(epoch + 1))
                fig.text(0.01, 0.75, 'Generated', ha='left')
                fig.text(0.01, 0.5, 'Original', ha='left')
                fig.text(0.01, 0.25, 'Reconstructed', ha='left')
                print(gener.shape, real.shape, recon.shape)
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

    return vae, optimizer
