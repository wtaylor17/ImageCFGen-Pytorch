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
    subject_clf = torch.load('AudioMNIST-subject-clf.tar', map_location=device)['model']

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


    bigan_mat = np.zeros((60, 10, 81))
    bigan_ft_mat = np.zeros((60, 10, 81))
    vae_mat = np.zeros((60, 10, 81))
    bigan_int_mat = np.zeros((60, 10, 81))
    vae_int_mat = np.zeros((60, 10, 81))

    with torch.no_grad():
        for subject in range(60):
            print("computing cfs for subject", subject)
            subject_batches = list(data.stream(
                excluded_runs=list(set(range(50)) - set(VALIDATION_RUNS)),
                excluded_subjects=list(set(range(1, 61)) - {subject + 1})
            ))
            subject_audio = torch.concat([b["audio"] for b in subject_batches], dim=0)
            subject_attrs = {
                k: torch.concat([b[k] for b in subject_batches], dim=0).float()
                for k in subject_batches[0]
                if k in ATTRIBUTE_DIMS
            }
            for d in range(10):
                print('CF digit', d)
                masknd = subject_attrs["digit"].argmax(1) != d
                xnd = spect_to_img(subject_audio[masknd])
                a_nd = {
                    k: v[masknd].float()
                    for k, v in subject_attrs.items()
                    if k in ATTRIBUTE_DIMS
                }
                bigan_codes = E(xnd, a_nd)
                bigan_ft_codes = E_ft(xnd, a_nd)
                vae_codes = vae.encoder(xnd, a_nd)[0]
                cf_a = dict(**a_nd)
                cf_a["digit"] = torch.zeros((len(xnd), 10)).float().to(device)
                cf_a["digit"][:, d] = 1

                # compute interventions/CFs
                bigan_cf = G(bigan_codes, cf_a)
                bigan_ft_cf = G_ft(bigan_ft_codes, cf_a)
                vae_cf = vae.decoder(vae_codes, cf_a)
                bigan_int = G(torch.randn_like(bigan_codes), cf_a)
                vae_int = vae.decoder(torch.randn_like(vae_codes), cf_a)

                bigan_mat[subject-1, d] = (
                            subject_clf(bigan_cf).argmax(1) == subject).int().cpu().numpy()
                bigan_ft_mat[subject-1, d] = (
                            subject_clf(bigan_ft_cf).argmax(1) == subject).int().cpu().numpy()
                vae_mat[subject-1, d] = (subject_clf(vae_cf).argmax(1) == subject-1).int().cpu().numpy()
                bigan_int_mat[subject-1, d] = (
                            subject_clf(bigan_int).argmax(1) == subject).int().cpu().numpy()
                vae_int_mat[subject-1, d] = (
                            subject_clf(vae_int).argmax(1) == subject).int().cpu().numpy()

    np.save('bigan_cf_agreement_mat.npy', bigan_mat)
    np.save('bigan_ft_cf_agreement_mat.npy', bigan_ft_mat)
    np.save('vae_cf_agreement_mat.npy', vae_mat)
    np.save('vae_int_agreement_mat.npy', vae_int_mat)
    np.save('bigan_int_agreement_mat.npy', bigan_int_mat)
