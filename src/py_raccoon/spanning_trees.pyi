import networkx as nx
import numpy as np

def uniform_spanning_tree(G: nx.Graph, rnd: np.random.Generator) -> np.ndarray[np.int32]:
    """
    Implements Wilson's Algorithm for uniform spanning trees [1].
    Assumes G to be undirected and connected.

    [1] David Bruce Wilson. 1996. Generating random spanning trees more quickly than the cover time. In Proceedings of the twenty-eighth annual ACM symposium on Theory of Computing (STOC '96). Association for Computing Machinery, New York, NY, USA, 296–303. https://doi.org/10.1145/237814.237880
    """