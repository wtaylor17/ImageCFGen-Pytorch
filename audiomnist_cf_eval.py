import torch
from image_scms.audio_mnist import AudioMNISTData, ATTRIBUTE_DIMS, VALIDATION_RUNS, Encoder, Generator
from deepscm_vae.audio_mnist import VAE
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bigan = torch.load('audio-mnist-retrain.tar', map_location=device)
    E = Encoder().to(device)
    E.load_state_dict(bigan["E_state_dict"])
    G = Generator().to(device)
    G.load_state_dict(bigan["G_state_dict"])
    bigan_finetuned = torch.load('audio-mnist-bigan-finetuned-mse.tar', map_location=device)
    E_ft, G_ft = Encoder().to(device), Generator().to(device)
    E_ft.load_state_dict(bigan_finetuned["E_state_dict"])
    G_ft.load_state_dict(bigan_finetuned["G_state_dict"])
    vae = torch.load('audiomnist-vae.tar', map_location=device)['vae']

    print("loading data...")
    data = AudioMNISTData('AudioMNIST-data.zip', device=device)
    print("done.")

    spect_mean, spect_ss, n_batches = 0, 0, 0

    print('Computing spectrogram statistics...')
    for batch in data.stream(batch_size=128,
                             excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS))):
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

    bigan_mat = np.zeros((60, 10, 10))
    bigan_ft_mat = np.zeros((60, 10, 10))
    vae_mat = np.zeros((60, 10, 10))

    for subject in range(1, 61):
        print("computing cfs for subject", subject)
        subject_batches = list(data.stream(
            excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS)),
            excluded_subjects=list(set(range(1, 61)) - {subject})
        ))
        subject_audio = torch.concat([b["audio"] for b in subject_batches], dim=0)
        subject_attrs = {
            k: torch.concat([b[k] for b in subject_batches], dim=0).float()
            for k in subject_batches[0]
            if k in ATTRIBUTE_DIMS
        }
        for d in range(10):
            print('original digit', d)
            mask = subject_attrs["digit"].argmax(1) == d
            xd = spect_to_img(subject_audio[mask])
            ad = {
                k: v[mask].float()
                for k, v in subject_attrs.items()
                if k in ATTRIBUTE_DIMS
            }
            bigan_codes = E(xd, ad)
            bigan_ft_codes = E_ft(xd, ad)
            vae_codes = vae.encoder(xd, ad)[0]

            for cf_d in range(10):
                if cf_d == d:
                    continue
                print('cf digit', cf_d)
                acf = dict(**ad)
                acf["digit"] = torch.zeros((len(xd), 10)).float()
                acf["digit"][:, cf_d] = 1

                bigan_cf = G(bigan_codes, acf).flatten(start_dim=1)
                bigan_ft_cf = G_ft(bigan_ft_codes, acf).flatten(start_dim=1)
                vae_cf = vae.decoder(vae_codes, acf).flatten(start_dim=1)

                # loop through data from different subject but same label
                bigan_other_err = torch.zeros((len(bigan_cf),)).to(device)
                bigan_ft_other_err = torch.zeros((len(bigan_cf),)).to(device)
                vae_other_err = torch.zeros((len(bigan_cf),)).to(device)
                denom = 0
                for other_batch in data.stream(excluded_subjects=[subject],
                                               excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS))):
                    other_mask = other_batch["digit"].argmax(1) == cf_d
                    other_x = spect_to_img(other_batch["audio"][other_mask]).flatten(start_dim=1)

                    bigan_other_err += (bigan_cf.unsqueeze(1) - other_x).square().sum(dim=(1, 2))
                    bigan_ft_other_err += (bigan_ft_cf.unsqueeze(1) - other_x).square().sum(dim=(1, 2))
                    vae_other_err += (vae_cf.unsqueeze(1) - other_x).square().sum(dim=(1, 2))
                    denom += len(other_x)

                bigan_other_err /= denom
                bigan_ft_other_err /= denom
                vae_other_err /= denom

                same_mask = subject_attrs["digit"].argmax(1) == cf_d
                same_x = spect_to_img(subject_audio[same_mask]).flatten(start_dim=1)
                bigan_same_err = (bigan_cf.unsqueeze(1) - same_x).square().sum(dim=(1, 2)) / len(same_x)
                bigan_ft_same_err = (bigan_ft_cf.unsqueeze(1) - same_x).square().sum(dim=(1, 2)) / len(same_x)
                vae_same_err = (vae_cf.unsqueeze(1) - same_x).square().sum(dim=(1, 2)) / len(same_x)

                bigan_mat[subject - 1, d, cf_d] = bigan_same_err.mean().cpu().item() / bigan_other_err.mean().cpu().item()
                vae_mat[subject - 1, d, cf_d] = vae_same_err.mean().item() / vae_other_err.mean().cpu().item()
                bigan_ft_mat[subject - 1, d, cf_d] = bigan_ft_same_err.mean().cpu().item() / bigan_ft_other_err.mean().cpu().item()

    np.save('bigan_cf_metric_mat.npy', bigan_mat)
    np.save('bigan_ft_cf_metric_mat.npy', bigan_ft_mat)
    np.save('vae_cf_metric_mat.npy', vae_mat)
