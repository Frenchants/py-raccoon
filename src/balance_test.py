import py_raccoon.sampling as smp
import py_raccoon.spanning_trees as st
import py_raccoon.balance_sampling as bal
import networkx as nx 
from matplotlib import pyplot as plt
import numpy as np
import random

G = nx.Graph()
G.add_edge(0, 1, weight=1)
G.add_edge(1, 3, weight=1)
G.add_edge(3, 2, weight=-1)
G.add_edge(2, 0, weight=-1)
G.add_edge(0, 4, weight=-1)

p = 0.2
G = nx.gnp_random_graph(7, p)
while not nx.is_connected(G):
    G  = nx.gnp_random_graph(7, p)

for u, v in G.edges():
        G[u][v]['weight'] = random.choice([-1, 1])



np_total_est_counts, np_positive_est_counts, np_negative_est_counts, total_zeros, positive_zeros, negative_zeros, np_total_occurred, np_positive_occurred, np_negative_occurred = bal.estimate_balance(G, 1000)
#np_est_counts, zeros, np_occured = smp.estimate_cycle_count(G, 1000)

print("-------------------------------------------")
print(f"np_total_est_counts: {np_total_est_counts}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_positive_est_counts: {np_positive_est_counts}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_negative_est_counts: {np_negative_est_counts}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"total_zeros: {total_zeros}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"positive_zeros: {positive_zeros}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"negative_zeros: {negative_zeros}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_total_occurred: {np_total_occurred}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_positive_occurred: {np_positive_occurred}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_negative_occurred: {np_negative_occurred}")
print("-------------------------------------------")

plt.figure(figsize=(5, 5))
pos = nx.spring_layout(G, k=30, seed=7)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(G, pos)

# edges
nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=2)

# node labels
nx.draw_networkx_labels(G, pos)
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.show()