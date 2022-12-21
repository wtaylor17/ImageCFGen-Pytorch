import torch.nn as nn
import numpy as np
import torch
from nice import NICE
from typing import List
from pytorch_msssim import ssim


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
    def __init__(self, G: nn.Module, codes: torch.Tensor):
        super().__init__()
        self.codes = codes
        self.G = G

    def forward(self, x: torch.Tensor):
        return self.G(self.codes, x)


class GradientCFExplainer:
    def __init__(self,
                 E: nn.Module,
                 G: nn.Module,
                 clf: nn.Module,
                 fixed_feat: List[int] = None,
                 cat_feat: List[List[int]] = None):
        self.E = E
        self.G = G
        self.clf = clf
        self.fixed_feat = fixed_feat or []
        self.cat_feat = cat_feat or []

    def explain(self,
                x_obs: torch.Tensor,
                a_obs: torch.Tensor,  # all dims
                n_iters=100,
                lr=1e-4,
                pred_loss_weight=0.1,
                target_class="other"):
        a_unfixed = a_obs[:, [i for i in range(a_obs.size(1))
                              if i not in self.fixed_feat]]
        a_out = torch.nn.Parameter(data=torch.clone(a_unfixed) + (2*torch.rand(a_unfixed.shape)-1) * 0.1,
                                   requires_grad=True)

        opt = torch.optim.Adam([a_out], lr=lr)

        codes = self.E(x_obs, a_obs).detach()
        wrapper = CFConditionalBiGANWrapper(self.G, codes)
        wrapper = FixedFeaturesModule(wrapper, a_obs, self.fixed_feat)
        original_pred = self.clf(wrapper(a_unfixed)).detach()
        for _ in range(n_iters):
            opt.zero_grad()
            gen = wrapper(a_out)
            pred = self.clf(gen)
            feature_loss = 1 - ssim(gen, x_obs, data_range=1.0).mean()
            if target_class == "other":
                # maximize dist
                target = original_pred
                criterion = torch.nn.CrossEntropyLoss()
                pred_loss = -criterion(pred, target)
            else:
                target = torch.ones((x_obs.size(0),)).long() * target_class
                criterion = torch.nn.CrossEntropyLoss()
                pred_loss = criterion(pred, target)
            loss = feature_loss + pred_loss_weight * pred_loss
            loss.backward()
            opt.step()
            with torch.no_grad():
                uncat_feat = [i for i in range(a_out.size(1))
                              if not any(i in x for x in self.cat_feat)]
                a_out[:, uncat_feat].clip_(0, 1)
                for x in self.cat_feat:
                    if all(i not in self.fixed_feat for i in x):
                        a_out[:, x] = a_out[:, x].softmax(1)

            print(f"Feature loss = {feature_loss.item()}, pred loss = {pred_loss.item()}")
        final_instance = wrapper(a_out)
        final_pred = self.clf(final_instance)
        a_new = torch.clone(a_obs)
        a_new[:, [i for i in range(a_obs.size(1))
                  if i not in self.fixed_feat]] = a_out
        return final_instance, a_new, final_pred
