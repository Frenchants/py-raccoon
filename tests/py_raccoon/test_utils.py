import py_raccoon as pr
import pandas as pd
import numpy as np
import networkx as nx

def test_estimate_er_params():
    G = nx.from_edgelist([(1,2), (1,3), (1,4)])
    n, p = pr.utils.estimate_er_params(G)
    assert n == 4, "Wrong number of nodes"
    assert p == 0.5, "Wrong ER probability p"