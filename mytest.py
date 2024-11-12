import networkx as nx
import numpy as np
import random

sizes = [10, 10]
p = [[0.5, 0.2], [0.2, 0.5]]

G = nx.stochastic_block_model(sizes, p, directed=True)
while not nx.is_weakly_connected(G):
    G = nx.stochastic_block_model(sizes, p, directed=True)

print(G.edges())

rnd = np.random.default_rng()

edges = list(G.edges())
rnd.shuffle(edges)

print(edges)