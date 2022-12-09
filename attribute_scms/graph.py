from typing import Dict, List

from collections import defaultdict
from .causal_module import CausalModuleBase, ConditionalTransformedCM, ConditionalCategoricalCM


class CausalModuleGraph:
    def __init__(self):
        self.modules: Dict[str, CausalModuleBase | ConditionalTransformedCM] = {}
        self.adj = defaultdict(set)
        self.adj_rev = defaultdict(set)

    def add_module(self, key: str, module: CausalModuleBase):
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
        copy.modules = self.modules.copy()
        copy.adj = self.adj.copy()
        copy.adj_rev = self.adj_rev.copy()

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

    def top_sort_modules(self):
        return [
            (v, self.modules[v],
             [self.modules[p] for p in self.parents(v)])
            for v in self.top_sort()
        ]

    def recover_noise(self, obs: dict):
        noise_out = {}
        for v in self.top_sort():
            if v in obs:
                v_parents = self.parents(v)
                if all(u in obs for u in v_parents):
                    self_val = obs[v]
                    parent_vals = [obs[u] for u in v_parents]
                    if type(self.modules[v]) is CausalModuleBase:
                        module: CausalModuleBase = self.modules[v]
                        noise_out[v] = module.recover_noise(self_val, parent_vals)
                    else:
                        module: ConditionalTransformedCM = self.modules[v]
                        noise_out[v] = module.condition(parent_vals)\
                                             .recover_noise(self_val, parent_vals)

        return noise_out
