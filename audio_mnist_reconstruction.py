from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt
from image_scms.audio_mnist import ATTRIBUTE_DIMS, AudioMNISTData, Generator, Encoder, VALIDATION_RUNS

parser = ArgumentParser()
parser.add_argument("-d", "--data", type=str, default="AudioMNIST-data.zip")


if __name__ == "__main__":
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load("audio-bigan.tar", map_location=device)
    E = Encoder().to(device)
    G = Generator().to(device)
    E.load_state_dict(model_dict["E_state_dict"])
    G.load_state_dict(model_dict["G_state_dict"])

    model_dict_ft = torch.load("audio-mnist-bigan-finetuned-mse.tar", map_location=device)
    E_ft = Encoder().to(device)
    G_ft = Generator().to(device)
    E_ft.load_state_dict(model_dict["E_state_dict"])
    G_ft.load_state_dict(model_dict["G_state_dict"])

    vae = torch.load("audiomnist-vae.tar", map_location=device)["vae"]

    with torch.no_grad():
        print("loading data...")
        data = AudioMNISTData(args.data, device=device)
        print("done.")

        spect_mean, spect_ss, n_batches = 0, 0, 0

        print('Computing spectrogram statistics...')
        for batch in data.stream(batch_size=128,
                                 excluded_runs=VALIDATION_RUNS):
            n_batches += 1
            spect_mean = spect_mean + batch["audio"].mean(dim=(0, 1)).reshape((1, 1, -1))
            spect_ss = spect_ss + batch["audio"].square().mean(dim=(0, 1)).reshape((1, 1, -1))

        spect_mean = (spect_mean / n_batches).float().to(device)  # E[X]
        spect_ss = (spect_ss / n_batches).float().to(device)  # E[X^2]
        spect_var = torch.relu(spect_ss - spect_mean.square())
        spect_std = torch.sqrt(spect_var)
        stds_kept = 3


        def spect_to_img(spect_):
            spect_ = (spect_ - spect_mean) / (spect_std + 1e-6)
            return torch.clip(spect_, -stds_kept, stds_kept) / float(stds_kept)


        def img_to_spect(img_):
            return img_ * stds_kept * (spect_std + 1e-6) + spect_mean

        batch = next(data.stream(excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS))))
        img = spect_to_img(batch["audio"]).float()
        batch = {k: v.float() for k, v in batch.items() if k in ATTRIBUTE_DIMS}
        bigan_rec = G(E(img, batch), batch)[0:1]
        bigan_ft_rec = G_ft(E_ft(img, batch), batch)[0:1]
        vae_rec = vae.decoder(vae.encoder(img, batch)[0], batch)[0:1]

        real_audio = data.spectrogram_to_audio(batch["audio"][0:1]).cpu().numpy()
        bigan_audio = data.spectrogram_to_audio(img_to_spect(bigan_rec)).cpu().numpy()
        bigan_ft_audio = data.spectrogram_to_audio(img_to_spect(bigan_ft_rec)).cpu().numpy()
        vae_audio = data.spectrogram_to_audio(img_to_spect(vae_rec)).cpu().numpy()

        fig, axs = plt.subplots(2, 4)

        axs[0, 0].imshow(img[0].reshape((128, 128)).numpy()[::-1], vmin=-1, vmax=1)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 0].set_title('Original')
        axs[1, 0].plot(real_audio.reshape((-1,)))

        axs[0, 1].imshow(bigan_rec[0].reshape((128, 128)).numpy()[::-1], vmin=-1, vmax=1)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 1].set_title('ImageCFGen')
        axs[1, 0].plot(bigan_audio.reshape((-1,)))

        axs[0, 1].imshow(bigan_ft_rec[0].reshape((128, 128)).numpy()[::-1], vmin=-1, vmax=1)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 1].set_title('ImageCFGen (fine-tuned)')
        axs[1, 0].plot(bigan_ft_audio.reshape((-1,)))

        axs[0, 1].imshow(vae_rec[0].reshape((128, 128)).numpy()[::-1], vmin=-1, vmax=1)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 1].set_title('DeepSCM')
        axs[1, 0].plot(vae_audio.reshape((-1,)))

        fig.suptitle('Audio-MNIST reconstructions')
        plt.savefig('audiomnist_reconstruction.png')
