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


np.random.seed(42)
VALIDATION_RUNS = np.random.randint(0, 50, size=(10,)).tolist()


def init_weights(layer, std=0.001):
    name = layer.__class__.__name__
    if name.startswith('Conv'):
        torch.nn.init.normal_(layer.weight, mean=0, std=std)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)


def compute_gradient_penalty(disc: nn.Module, interpolates: torch.Tensor):
    """Calculates the gradient penalty loss for WGAN GP
    source: https://github.com/eriklindernoren/PyTorch-GAN/blob/a163b82beff3d01688d8315a3fd39080400e7c01/implementations/wgan_gp/wgan_gp.py"""
    interpolates = interpolates.requires_grad_(True)
    d_interpolates = disc(interpolates)
    fake = torch.autograd.Variable(
        torch.ones_like(d_interpolates).to(interpolates.device),
        requires_grad=False
    )
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def wgan_loss_it(disc: nn.Module,
                 x_real: torch.Tensor,
                 x_fake: torch.Tensor,
                 penalty_weight=10.0) -> torch.Tensor:
    assert x_real.shape[0] == x_fake.shape[0], "batch size must be constant"

    loss_no_penalty = disc(x_fake) - disc(x_real)

    n = x_real.shape[0]
    eps = torch.rand((n, 1, 1, 1)).to(x_real.device)
    x_rand = eps * x_real + (1 - eps) * x_fake

    return loss_no_penalty + penalty_weight * compute_gradient_penalty(disc, x_rand)


LATENT_DIM = 100
IMAGE_SHAPE = (128, 128)


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


class Generator(nn.Module):
    def __init__(self, d=64):
        super(Generator, self).__init__()
        ct2d = partial(nn.ConvTranspose2d,
                       stride=2,
                       padding=2,
                       output_padding=1)
        self.layers = nn.Sequential(
            nn.Linear(LATENT_DIM, 256 * d),
            nn.Unflatten(1, (16 * d, 4, 4)),
            nn.LeakyReLU(0.2),
            ct2d(16 * d, 8 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(8 * d, 4 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(4 * d, 2 * d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(2 * d, d, (5, 5)),
            nn.LeakyReLU(0.2),
            ct2d(d, 1, (5, 5)),
            nn.Tanh()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, z: torch.Tensor):
        z = z.reshape((-1, LATENT_DIM))
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, d, (5, 5), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d, 2 * d, (5, 5), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * d, 4 * d, (5, 5), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * d, 8 * d, (5, 5), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8 * d, 16 * d, (5, 5), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(16 * d, 1)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X: torch.Tensor):
        X = X.reshape((-1, 1, *IMAGE_SHAPE))
        return self.layers(X)


def train(path_to_zip: str,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          save_images_every=1,
          batch_size=64,
          image_output_path='',
          generator_size=64,
          discriminator_size=64,
          d_updates_per_g_update=1,
          discriminator_weight_decay=0.0,
          loss_mode="gan"):
    G = Generator(generator_size).to(device)
    D = Discriminator(discriminator_size).to(device)

    G.apply(init_weights)
    D.apply(init_weights)

    optimizer_G = torch.optim.Adam(G.parameters(),
                                   lr=l_rate,
                                   betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(D.parameters(),
                                   lr=l_rate,
                                   betas=(0.5, 0.9),
                                   weight_decay=discriminator_weight_decay)

    gan_loss = nn.BCEWithLogitsLoss()

    print('Loading dataset...')
    data = AudioMNISTData(path_to_zip, device=device)
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

    print('Beginning training')
    ctr = 0
    for epoch in range(n_epochs):
        D_score = 0.
        EG_score = 0.
        D.train()
        G.train()
        tq = tqdm(data.stream(batch_size=batch_size), total=n_batches)
        for i, batch in enumerate(tq):
            images = batch["audio"].float().to(device)
            images = spect_to_img(images)

            valid = torch.autograd.Variable(
                torch.Tensor(images.size(0), 1).fill_(1.0).to(device),
                requires_grad=False
            )
            fake = torch.autograd.Variable(
                torch.Tensor(images.size(0), 1).fill_(0.0).to(device),
                requires_grad=False
            )

            # Generator training
            if ctr % d_updates_per_g_update == 0:
                z = torch.randn((len(images), LATENT_DIM)).to(device)
                optimizer_G.zero_grad()
                if loss_mode == "gan":
                    gen = G(z)
                    loss_G = gan_loss(D(gen), valid)
                elif loss_mode == "wgan":
                    loss_G = -D(G(z)).mean()
                else:
                    raise NotImplementedError(loss_mode)
                loss_G.backward()
                optimizer_G.step()
            ctr += 1

            # Discriminator training
            optimizer_D.zero_grad()
            z = torch.randn((len(images), LATENT_DIM)).to(device)
            if loss_mode == "gan":
                loss_D = gan_loss(D(images), valid)
                loss_D = (loss_D + gan_loss(D(G(z)), fake)) / 2
            elif loss_mode == "wgan":
                loss_D = wgan_loss_it(D, images, G(z)).mean()
            else:
                raise NotImplementedError(loss_mode)
            loss_D.backward()
            optimizer_D.step()

            z = torch.randn((len(images), LATENT_DIM)).to(device)
            Gz = G(z).detach()
            DG = D(Gz)
            DE = D(images)
            if loss_mode == "gan":
                DG = DG.sigmoid()
                DE = DE.sigmoid()
            D_score += DG.mean().item()
            EG_score += DE.mean().item()
            tq.set_postfix({"D(G(z))": round(DG.mean().item(), 4), "D(X)": round(DE.mean().item(), 4)})

        print(D_score / n_batches, EG_score / n_batches)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            n_show = 4
            D.eval()
            G.eval()

            with torch.no_grad():
                # generate images from same class as real ones
                demo_batch = next(data.stream(batch_size=n_show))
                images = demo_batch["audio"].float().to(device)
                x = spect_to_img(images)

                z = torch.randn((len(x), LATENT_DIM))
                z = z.to(device)

                gener = G(z).reshape(n_show, *IMAGE_SHAPE)
                real = x.reshape((n_show, *IMAGE_SHAPE))
                vmin, vmax = -1, 1

            if save_images_every is not None:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(2, n_show, figsize=(15, 5))
                fig.subplots_adjust(wspace=0.05, hspace=0)
                plt.rcParams.update({'font.size': 20})
                fig.suptitle('Epoch {}'.format(epoch + 1))
                fig.text(0.01, 0.75, 'G(z, c)', ha='left')
                fig.text(0.01, 0.25, 'x', ha='left')

                for i in range(n_show):
                    ax[0, i].imshow(gener[i].cpu(), vmin=vmin, vmax=vmax)
                    ax[0, i].axis('off')
                    ax[1, i].imshow(real[i].cpu(), vmin=vmin, vmax=vmax)
                    ax[1, i].axis('off')
                plt.savefig(f'{image_output_path}/epoch-{epoch + 1}.png')
                plt.close()

                gener_wav = data.inv_transforms["audio"](
                    img_to_spect(gener[0:1]).cpu().numpy()
                ).cpu().numpy()[0]
                real_wav = data.inv_transforms["audio"](
                    img_to_spect(real[0:1]).cpu().numpy()
                ).cpu().numpy()[0]

                write_wav(f"{image_output_path}/epoch-{epoch + 1}-generated.wav", 8000,
                          np.int16(gener_wav / np.max(np.abs(gener_wav)) * 32767))
                write_wav(f"{image_output_path}/epoch-{epoch + 1}-real.wav", 8000,
                          np.int16(real_wav / np.max(np.abs(real_wav)) * 32767))

                print('Image and audio saved to', image_output_path)

    return G, D, optimizer_D, optimizer_G
