import py_raccoon.sampling as smp
import py_raccoon.spanning_trees as st
import py_raccoon.balance
import networkx as nx 
from matplotlib import pyplot as plot
import numpy as np

p = 0.2
G = nx.gnp_random_graph(5, p)
while not nx.is_connected(G):
    G  = nx.gnp_random_graph(10, p)

nx.draw(G)
plot.show()


np_est_counts, zeros, np_occured = smp.estimate_cycle_count(G, 100000, p)

print(np_est_counts)

nx.draw(G)
plot.show()



"""
0 - 1 - 2
|   
3
-
4



print(smp.get_induced_cycle((0,2), np.array([0, 0, 1, 0, 3]), np.array([0, 1, 2, 1, 2])))
#def get_induced_cycle(edge: Tuple[int, int], parent: np.ndarray, depth: np.ndarray) -> tuple:



0, 0, 1, 2 """