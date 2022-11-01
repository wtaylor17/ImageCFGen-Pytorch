from collections import defaultdict


class Graph:
    def __init__(self):
        self.adj = defaultdict(set)
        self.adj_rev = defaultdict(set)
        self.vertices = set()

    def add_edge(self, u, v):
        self.adj[u].add(v)
        self.adj_rev[v].add(u)
        self.vertices.add(u)
        self.vertices.add(v)

    def remove_edge(self, u, v):
        self.adj[u].remove(v)
        self.adj_rev[v].remove(u)

    def parents(self, u):
        return self.adj_rev[u]

    def children(self, u):
        return self.adj[u]

    def top_sort(self):
        copy = Graph()
        copy.vertices = self.vertices
        copy.adj = self.adj
        copy.adj_rev = self.adj_rev

        out = []
        sources = {
            v for v in copy.vertices
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
