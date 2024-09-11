# cython: gdb_debug=True
import py_raccoon.sampling as smp
import py_raccoon.spanning_trees as st
import py_raccoon.balance_sampling as bal
import networkx as nx 
from matplotlib import pyplot as plot
import numpy as np

G = nx.Graph()
G.add_edge(1, 2, weight=1)
G.add_edge(2, 3, weight=1)
G.add_edge(1, 3, weight=1)
G.add_edge(0, 3, weight=1)


np_est_counts, zeros, np_occured = bal.estimate_cycle_count(G, 1000)
#np_est_counts, zeros, np_occured = smp.estimate_cycle_count(G, 1000)


print(np_est_counts)

nx.draw(G)
plot.show()