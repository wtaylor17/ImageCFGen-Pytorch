import torch.nn as nn
import torch
from typing import List


def batchify(*tensors, batch_size=128, device='cpu'):
    i = 0
    N = min(map(len, tensors))
    while i < N:
        yield tuple(x[i:i+batch_size] for x in tensors)
        i += batch_size
    if i < N:
        yield tuple(x[i:N].to(device) for x in tensors)


class FixedFeaturesModule(nn.Module):
    def __init__(self,
                 wrapped_module: nn.Module,
                 attrs_init: torch.Tensor,
                 fixed_attrs: List[int]):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.attrs_init = torch.clone(attrs_init).detach()
        self.fixed_attrs = fixed_attrs
        self.unfixed_attrs = [i for i in range(self.attrs_init.size(1))
                              if i not in fixed_attrs]

    def forward(self, x: torch.Tensor):
        attrs = [torch.Tensor() for _ in range(self.attrs_init.size(1))]
        for i, attr_idx in enumerate(self.fixed_attrs):
            attrs[attr_idx] = self.attrs_init[:, attr_idx:attr_idx+1]
        for i, attr_idx in enumerate(self.unfixed_attrs):
            attrs[attr_idx] = x[:, i:i+1]
        inp = torch.concat(attrs, dim=1)
        return self.wrapped_module(inp)


class CFConditionalBiGANWrapper(nn.Module):
    """
    use this module to fix the first argument of a wrapped module
    """
    def __init__(self, G: nn.Module, codes: torch.Tensor):
        super().__init__()
        self.codes = codes
        self.G = G

    def forward(self, x: torch.Tensor):
        return self.G(self.codes, x)
