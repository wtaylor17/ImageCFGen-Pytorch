from image_scms import whalecalls
import torch
from tqdm import tqdm


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc_rounds = 4
    print('loading data')
    data = whalecalls.WhaleCallData("WhaleData/Nocall", "WhaleData/Gunshot", "WhaleData/Upcall")
    print('done')
    bigan_dict = torch.load('whale_bigan1.tar', map_location=device)
    E, G = bigan_dict['E'], bigan_dict['G']
    vae = torch.load('whale_vae1.tar', map_location=device)['vae']
    clf = torch.load('whalecall_clf.tar', map_location=device)['clf']

    with torch.no_grad():
        """spect_mean, spect_ss, n_batches = 0, 0, 0

        print('Computing spectrogram statistics...')
        for batch in data.stream():
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
            return img_ * stds_kept * (spect_std + 1e-6) + spect_mean"""
        n_batches = 0
        for batch in data.stream(mode='validation'):
            n_batches += 1

        n_bigan_correct, n_vae_correct, n = 0, 0, 0
        for batch in tqdm(data.stream(mode='validation'), total=n_batches):
            bigan_gen = 0
            vae_gen = 0
            batch["call_type"] = batch["call_type"].to(device)
            for _ in range(mc_rounds):
                z = torch.randn((len(batch["call_type"]), 512, 1, 1)).to(device)
                bigan_gen = bigan_gen + G(z, batch)
                vae_gen = vae_gen + vae.decoder(z, batch)

            bigan_gen = bigan_gen / mc_rounds
            vae_gen = vae_gen / mc_rounds
            bigan_ft_gen = bigan_ft_gen / mc_rounds
            tgt = batch["call_type"].argmax(1)

            n_bigan_correct += (clf(bigan_gen).argmax(1) == tgt).sum()
            n_vae_correct += (clf(vae_gen).argmax(1) == tgt).sum()
            n += len(batch["call_type"])

        print('Bigan:', n_bigan_correct / n)
        print('Vae:', n_vae_correct / n)
