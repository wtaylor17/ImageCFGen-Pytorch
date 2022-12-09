from typing import Dict, List, Union, Optional

from collections import defaultdict
import torch
from .causal_module import CausalModuleBase, ConditionalTransformedCM, TransformedCM


class CausalModuleGraph:
    def __init__(self):
        self.modules: Dict[str, Union[CausalModuleBase, ConditionalTransformedCM]] = {}
        self.adj = defaultdict(set)
        self.adj_rev = defaultdict(set)

    def get_module(self, key: str) -> Union[CausalModuleBase, ConditionalTransformedCM]:
        return self.modules.get(key)

    def add_module(self, key: str, module: Union[CausalModuleBase, ConditionalTransformedCM]):
        self.modules[key] = module

    def assert_has_module(self, key):
        assert key in self.modules, "modules must be added with add_module first"

    def add_edge(self, u, v):
        self.assert_has_module(u)
        self.assert_has_module(v)
        self.adj[u].add(v)
        self.adj_rev[v].add(u)

    def remove_edge(self, u, v):
        self.assert_has_module(u)
        self.assert_has_module(v)
        self.adj[u].remove(v)
        self.adj_rev[v].remove(u)

    def parents(self, u):
        self.assert_has_module(u)
        return sorted(self.adj_rev[u])

    def children(self, u):
        self.assert_has_module(u)
        return sorted(self.adj[u])

    def top_sort(self):
        copy = CausalModuleGraph()
        copy.modules = dict(**self.modules)
        copy.adj = defaultdict(set)
        copy.adj_rev = defaultdict(set)
        for v in self.adj:
            for u in self.adj[v]:
                copy.add_edge(v, u)

        out = []
        sources = {
            v for v in copy.modules
            if len(copy.parents(v)) == 0
        }

        while sources:
            n = sources.pop()
            out.append(n)
            for m in copy.children(n):
                copy.remove_edge(n, m)
                if len(copy.parents(m)) == 0:
                    sources.add(m)

        return out

    def recover_noise(self, obs: Dict[str, torch.Tensor]):
        noise_out = {}
        for v in self.modules:
            if v in obs:
                v_parents = self.parents(v)
                if all(u in obs for u in v_parents):
                    self_val = obs[v]
                    parent_vals = [obs[u] for u in v_parents]
                    if len(parent_vals) > 0:
                        parent_vals = torch.concat(parent_vals, dim=-1)
                    if isinstance(self.modules[v], CausalModuleBase):
                        module: CausalModuleBase = self.modules[v]
                        noise_out[v] = module.recover_noise(self_val, parent_vals)
                    else:
                        module: ConditionalTransformedCM = self.modules[v]
                        noise_out[v] = module.condition(parent_vals)\
                                             .recover_noise(self_val, parent_vals)

        return noise_out

    def log_prob(self, obs: Dict[str, torch.Tensor]):
        lp_out = {}
        for v in self.modules:
            if v in obs:
                v_parents = self.parents(v)
                if all(u in obs for u in v_parents):
                    self_val = obs[v]
                    parent_vals = [obs[u] for u in v_parents]
                    if len(parent_vals) > 0:
                        parent_vals = torch.concat(parent_vals, dim=-1)
                    if isinstance(self.modules[v], CausalModuleBase):
                        module: CausalModuleBase = self.modules[v]
                        lp_out[v] = module.forward(parent_vals).log_prob(self_val)
                    else:
                        module: ConditionalTransformedCM = self.modules[v]
                        lp_out[v] = module.condition(parent_vals)\
                                          .forward(parent_vals).log_prob(self_val)
        return lp_out

    def sample(self,
               obs_in: Optional[Dict[str, torch.Tensor]] = None,
               n: int = 1) -> Dict[str, torch.Tensor]:
        obs_out = obs_in or {}
        for v in self.modules:
            if v in obs_out:  # this value is being held constant
                continue
            v_parents = self.parents(v)
            parent_vals = [obs_out[u] for u in v_parents]
            if len(parent_vals) > 0:
                parent_vals = torch.concat(parent_vals, dim=-1)
            if isinstance(self.modules[v], CausalModuleBase):
                module: CausalModuleBase = self.modules[v]
                obs_out[v] = module.condition(parent_vals).sample((n,))
            else:
                module: ConditionalTransformedCM = self.modules[v]
                obs_out[v] = module.condition(parent_vals) \
                                   .condition(parent_vals).sample((n,))
        return obs_out
