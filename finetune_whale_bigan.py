from image_scms import whalecalls
import torch
from tqdm import tqdm

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--model-file', type=str)
parser.add_argument('--steps', type=int, default=20)

if __name__ == '__main__':
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    print('loading data')
    data = whalecalls.WhaleCallData("WhaleData/Nocall", "WhaleData/Gunshot", "WhaleData/Upcall")
    print('done')
    bigan_dict = torch.load(args.model_file, map_location=device)
    E, G = bigan_dict['E'], bigan_dict['G']

    with torch.no_grad():
        spect_mean, spect_ss, n_batches = 0, 0, 0

        print('Computing spectrogram statistics...')
        training_call_types = []
        for batch in data.stream():
            n_batches += 1
            spect_mean = spect_mean + batch["audio"].mean(dim=(0, 1)).reshape((1, 1, -1))
            spect_ss = spect_ss + batch["audio"].square().mean(dim=(0, 1)).reshape((1, 1, -1))
            training_call_types.extend(batch["call_type"].argmax(1).flatten().cpu().numpy().tolist())

        spect_mean = (spect_mean / n_batches).float().to(device)  # E[X]
        spect_ss = (spect_ss / n_batches).float().to(device)  # E[X^2]
        spect_std = torch.sqrt(spect_ss - spect_mean.square())
        stds_kept = 3


        def spect_to_img(spect_):
            spect_ = (spect_ - spect_mean) / (spect_std + 1e-6)
            return torch.clip(spect_, -stds_kept, stds_kept) / float(stds_kept)


        def img_to_spect(img_):
            return img_ * stds_kept * (spect_std + 1e-6) + spect_mean
        n_batches = 0
        for batch in data.stream(batch_size=batch_size):
            n_batches += 1

    E.train()
    G.eval()
    opt = torch.optim.Adam(E.parameters(), lr=1e-5)

    for i in range(args.steps):
        R, L = 0, 0
        n_batches = 0

        for batch in tqdm(data.stream(batch_size=batch_size)):
            x = spect_to_img(batch["audio"].to(device)).to(device)
            batch["call_type"] = batch["call_type"].to(device)

            opt.zero_grad()
            codes = E(x, batch)
            xr = G(codes, batch)
            rec_loss = torch.square(x - xr).mean()
            loss = rec_loss
            latent = torch.square(codes).mean()
            L += latent.item()
            loss = loss + latent
            R += rec_loss.item()
            loss.backward()
            opt.step()
            n_batches += 1
        print(f'Epoch {i + 1}/{args.steps}: mse={round(R / n_batches, 4)} ', end='')
        print(f'latent loss ={round(L / n_batches, 4)}')

    torch.save({
        'E': E,
        'G': G
    }, 'whale_bigan_finetuned.tar')
