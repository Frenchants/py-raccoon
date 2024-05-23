import networkx as nx

def estimate_er_params(G: nx.Graph):
    """
    Estimates n and p for a s.t. G ~ G(n,p)

    Returns: Tuple (n,p)
    """
    n = len(G.nodes)
    return n, len(G.edges) * 2 / (n*n - n)