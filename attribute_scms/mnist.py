import torch
import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
from tqdm import tqdm

from .training_utils import batchify, dist_parameters


def thickness_distribution(device='cpu'):
    t_base = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    transforms = [T.BatchNorm(1).to(device),
                  T.ExpTransform()]
    return dist.TransformedDistribution(t_base, transforms)


def intensity_distribution(min_, max_, device='cpu'):
    i_base = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    transforms = [T.conditional_affine_autoregressive(1, 1).to(device),
                  T.SigmoidTransform(),
                  T.AffineTransform(min_, max_ - min_)]
    return dist.ConditionalTransformedDistribution(i_base, transforms)


def slant_distribution(min_, max_, device='cpu'):
    s_base = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    transforms = [T.Spline(1).to(device),
                  T.AffineTransform(min_, max_ - min_)]
    return dist.TransformedDistribution(s_base, transforms)


def label_distribution(a_train: torch.Tensor,
                       device='cpu'):
    a_train = a_train.to(device)
    torch.unique()
    y_vals, y_counts = torch.unique(a_train[:, :10].argmax(dim=1), return_counts=True)
    # this doesn't need gradient training, get MLE like this
    return dist.Categorical(probs=torch.Tensor(y_counts / a_train.size(0)).to(device))


def init_all_distributions(a_train: torch.Tensor,
                           intensity_idx=11,
                           slant_idx=12,
                           device='cpu'):
    a_train = a_train.to(device)

    t_dist = thickness_distribution(device)

    intensity = a_train[:, intensity_idx]
    i_min, i_max = intensity.min(), intensity.max()
    i_dist = intensity_distribution(i_min, i_max, device)

    slant = a_train[:, slant_idx]
    s_min, s_max = slant.min(), slant.max()
    s_dist = slant_distribution(s_min, s_max, device)

    l_dist = label_distribution(a_train, device)

    return t_dist, i_dist, s_dist, l_dist


def train(a_train: torch.Tensor,
          steps=2000,
          thickness_idx=10,
          intensity_idx=11,
          slant_idx=12,
          device='cpu'):
    t_dist, i_given_t_dist, s_dist, l_dist = init_all_distributions(
        a_train,
        intensity_idx=intensity_idx,
        slant_idx=slant_idx,
        device=device
    )

    optimizer = torch.optim.Adam(dist_parameters(t_dist) +
                                 dist_parameters(i_given_t_dist) +
                                 dist_parameters(s_dist),
                                 lr=1e-2)

    thickness = a_train[:, thickness_idx]
    intensity = a_train[:, intensity_idx]
    slant = a_train[:, slant_idx]

    tq = tqdm(range(steps))
    for _ in tq:
        idx = np.random.permutation(thickness.size(0))
        batches = list(batchify(thickness[idx],
                                intensity[idx],
                                slant[idx], batch_size=10_000))
        epoch_loss = 0
        for t, i, s in batches:
            optimizer.zero_grad()
            loss = -(t_dist.log_prob(t) +
                     i_given_t_dist.condition(t).log_prob(i) +
                     s_dist.log_prob(s)).mean()
            loss.backward()
            optimizer.step()
            t_dist.clear_cache()
            i_given_t_dist.clear_cache()
            s_dist.clear_cache()
            epoch_loss += loss.item()
        tq.set_description(f'loss = {round(epoch_loss / len(batches), 4)}')

    return t_dist, i_given_t_dist, s_dist, l_dist, optimizer


def load_model(tar_path, device='cpu'):
    obj = torch.load(tar_path, map_location=device)

    t_dist = obj['t_dist']
    i_given_t_dist = obj['i_given_t_dist']
    s_dist = obj['s_dist']
    return t_dist, i_given_t_dist, s_dist
