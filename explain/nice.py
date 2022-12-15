import torch.nn as nn
import numpy as np
import torch
from nice import NICE
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
        inp = torch.zeros((x.size(0), self.attrs_init.size(1)))
        inp[:, self.fixed_attrs] = self.attrs_init[:, self.fixed_attrs]
        inp[:, self.unfixed_attrs] = x
        return self.wrapped_module(inp)


class NICEConditionalBiGANWrapper(nn.Module):
    def __init__(self, G: nn.Module, codes: torch.Tensor):
        super().__init__()
        self.codes = codes
        self.G = G

    def forward(self, x: torch.Tensor):
        return self.G(self.codes, x)


class LocalCFNICE:
    def __init__(self,
                 E: nn.Module,
                 G: nn.Module,
                 clf: nn.Module,
                 a_train: torch.Tensor,  # all dims
                 y_train: torch.Tensor = None,
                 cat_feat: List[int] = None,
                 fixed_feat: List[int] = None):
        self.E = E
        self.G = G
        self.a_train = a_train[:, [i for i in range(a_train.size(1))
                                   if i not in fixed_feat]]
        self.clf = clf
        self.cat_feat = cat_feat or []
        self.y_train = y_train
        self.fixed_feat = fixed_feat or []

    def explain(self,
                x_obs: torch.Tensor,
                a_obs: torch.Tensor,  # all dims
                from_logits=False,
                batch_size=100,
                **kwargs):
        codes: torch.Tensor = self.E(x_obs, a_obs).detach()

        def predict_fn(a: np.array):
            preds = []
            for (a_batch,) in batchify(a):
                a_t = torch.from_numpy(a_batch).float().to(x_obs.device)
                codes_rpt = codes.expand(a_t.size(0), -1, -1, -1)
                a_obs_rpt = a_obs.expand(a_t.size(0), -1)

                wrapper = NICEConditionalBiGANWrapper(self.G, codes_rpt)
                wrapper = FixedFeaturesModule(wrapper, a_obs_rpt, self.fixed_feat)
                pred = self.clf(wrapper(a_t))
                if from_logits:
                    pred = pred.softmax(1)
                preds.append(pred.detach().cpu().numpy())
            return np.concatenate(preds, axis=0)

        nice = NICE(predict_fn,
                    self.a_train.detach().cpu().numpy(),
                    self.cat_feat,
                    y_train=self.y_train,
                    **kwargs)
        attr_explain = nice.explain(a_obs[:, [10, 11, 12]].detach().cpu().numpy())
        attr_explain = torch.from_numpy(attr_explain).to(a_obs.device).reshape((1, -1))
        return self.G(codes, attr_explain), attr_explain
