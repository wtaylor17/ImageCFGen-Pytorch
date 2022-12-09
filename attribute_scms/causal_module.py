import torch
import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
from torch.distributions import TransformedDistribution as TorchTD
from abc import abstractmethod

from .training_utils import nf_inverse


class TransformedDistribution(dist.TorchDistributionMixin, TorchTD):
    pass


class CausalModuleBase(torch.nn.Module):
    # meant to compute p(U)
    @abstractmethod
    def noise_distribution(self, *args, **kwargs) -> dist.Distribution:
        raise NotImplementedError

    # meant to compute a sample from p(U|obs,context)
    @abstractmethod
    def recover_noise(self, obs, context, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    # meant to compute p(obs|context)
    @abstractmethod
    def forward(self, *args, **kwargs) -> dist.Distribution:
        raise NotImplementedError


class TransformedCM(CausalModuleBase):
    def __init__(self, td: TransformedDistribution):
        super().__init__()
        self.td = td

    def noise_distribution(self, *args, **kwargs) -> dist.Distribution:
        return self.td.base_dist

    def recover_noise(self, obs, context, **kwargs) -> torch.Tensor:
        noise_val = nf_inverse(self.td, obs)
        return noise_val

    def forward(self, *args, **kwargs) -> dist.Distribution:
        return self.td


class CategoricalCM(CausalModuleBase):
    def __init__(self, d: dist.Categorical):
        super().__init__()
        self.d = d

    def noise_distribution(self, *args, **kwargs) -> dist.Distribution:
        return self.d

    def recover_noise(self, obs, context, **kwargs) -> torch.Tensor:
        return x

    def forward(self, *args, **kwargs) -> dist.Distribution:
        return self.d


class ConditionalTransformedCM:
    def __init__(self, ctd: dist.ConditionalTransformedDistribution):
        self.ctd = ctd

    def condition(self, context) -> TransformedCM:
        return TransformedCM(self.ctd.condition(context))


def gumbel_distribution() -> TransformedDistribution:
    transforms = [
        T.ExpTransform().inv,
        T.AffineTransform(0, -1),
        T.ExpTransform().inv,
        T.AffineTransform(0, -1)
    ]
    base = dist.Uniform(0, 1)
    return TransformedDistribution(base, transforms)


class ConditionalCategoricalCM(CausalModuleBase):
    def __init__(self, model: torch.nn.Module, n_categories: int):
        super().__init__()
        self.model = model
        self.n_categories = n_categories

    def noise_distribution(self, *args, **kwargs) -> dist.Distribution:
        return self.gumbel

    def recover_noise(self, y, context, **kwargs) -> torch.Tensor:
        inds = list(range(y.size(0)))
        g = self.gumbel.sample(y.shape + (self.n_categories,)).to(y.device)
        gk = g[inds, y]
        logits = self.model(context)
        noise_k = gk + logits.exp().sum(dim=-1).log() - logits[inds, y]
        noise_l = -torch.log(torch.exp(-g - logits) +
                             torch.exp(-gk - logits[inds, y])) - logits
        noise_l[inds, y] = noise_k
        return noise_l

    def forward(self, context, **kwargs) -> dist.Distribution:
        logits = self.model(context)
        return dist.Categorical(logits=logits)
