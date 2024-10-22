# distutils: language=c++
# cython: profile=True

from libc.stdlib cimport malloc, free
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from math import log2
from libc.math cimport log2 as clog2
from .utils import estimate_er_params

# Backport since exp2 was introduced in python 3.11
# Note: The improved accuracy of exp2 will not be noticeable since the 
#       accuracy of 2**x is greater than that of our approximation.
cdef inline double cexp2(double x) nogil:
    return 2**x

import cython

from .balance_spanning_trees cimport lowest_common_ancestor, Edge, LcaResult, Graph_c, calc_property_fast, to_Graph_c, free_neighbors_and_weights, uniform_spanning_tree_c, Edge
from .balance_spanning_trees import NP_EDGE, calc_depth, uniform_spanning_tree, get_induced_cycle

#from libcpp.vector cimport vector

cdef packed struct OccurenceProb:
    int u
    int v
    int lca
    double p_c
    signed char bal

NP_OCCURENCE_PROB = np.dtype([
    ('u', np.int32),
    ('v', np.int32),
    ('lca', np.int32),
    ('p_c', np.float64),
    ('bal', np.int8)
])

def estimate_balance(G: nx.Graph, samples: int, p: float | None = None, seed:int|np.random.Generator|None=None) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[bool], NDArray[bool], NDArray[bool], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """
    Estimates the number of simple cycles in G using spanning-tree-based sampling.

    Originally designed for Erdös-Rényi Graphs, but is still relatively accurate on many other graphs.

    Parameters:
    - G: Graph to estimate the number of simple cycles for
    - samples: Number of spanning trees to sample for the estimation. More samples lead to a more accurate estimation.
    - p: Edge Probability used for generating G (assuming Erdös-Rényi). Will be inferred if not specified.
    - seed: Random seed or generator to use. Will generate a new `numpy.random.default_rng` if number or no seed is specified.

    Returns: log_cycle_counts, is_zero, length_occurred

    - log_cycle_counts: np.ndarray of length n + 1. Position l contains the log of the estimated number of cycles of length l.
    - is_zero: np.ndarray of length n + 1. Position l is True if the estimated number of cycles of length l is 0.
    - length_occurred: np.ndarray of length n + 1. Position l contains the number of times a cycles of length l was encountered during the sampling process.
    """
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)

    if p is None:
        _, p = estimate_er_params(G)

    edges = np.array([(u,v,w) for (u,v,w) in G.edges.data('weight')], dtype=NP_EDGE)

    return estimate_len_count_fast(G, edges, p, samples, seed)

@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_len_count_fast(G: nx.Graph, edges: np.ndarray[NP_EDGE], p: float, samples: int, seed: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """
    Estimates the number of cycles in `G` for each length.

    For the estimation, it simulates a sampling of cells with probability $P = 1$.
    The number of cells for each length is then the sum of the resulting sampling probability $p'_c$ (which may be greater than 1).

    **Parameters** 
    G:          the graph
    p:          edge probability on G
    samples:    number of spanning trees to sample

    Returns: array of count, indexed by length. Length of the array is number of nodes in `G` plus one.
    """
    cdef int n = len(G.nodes)
    node_degree = np.array(G.degree, dtype=np.int32)
    degree_np = node_degree[:,1][node_degree[:,0].argsort()] # nx may not consider the nodes to be in ascending order by their id
    cdef int[:] degree = degree_np
    cdef Graph_c graph_c = to_Graph_c(n, degree, G)
    cdef int** neighbors = graph_c.neighbors
    cdef signed char** weights = graph_c.weights

    np_positive_expected_counts = np.zeros(n + 1, np.float64)
    cdef double[:] positive_expected_counts = np_positive_expected_counts
    np_positive_occurred = np.zeros(n + 1, np.int32)
    cdef int[:] positive_occurred = np_positive_occurred

    np_negative_expected_counts = np.zeros(n + 1, np.float64)
    cdef double[:] negative_expected_counts = np_negative_expected_counts
    np_negative_occurred = np.zeros(n + 1, np.int32)
    cdef int[:] negative_occurred = np_negative_occurred

    np_P = np.ndarray(n + 1, np.float64)
    cdef double[:] P = np_P
    
    np_P[[0,1,2]] = 0
    cdef int l
    for l in range(3,n + 1):
        # log to avoid floating point limitations
        P[l] = clog2(samples) + clog2(l) + clog2(n-2)*(l-2) - clog2(n)*(2*l - 4) - clog2(p)*(l-3)
    #P[:] = 0

    undersample = 0
    cdef int i, j
    cdef int[:] parent, depth
    cdef double p_c_prime
    cdef OccurenceProb[:] occ_probs
    cdef OccurenceProb op
    cdef int tree_root
    for i in range(samples):
        np_parent = np.ndarray(n, np.int32)
        parent = np_parent
        np_parent_weight = np.ndarray(n, np.int8)
        parent_weight = np_parent_weight
        tree_root = uniform_spanning_tree_c(n, degree, neighbors, weights, parent, parent_weight, seed)
        np_depth = calc_depth(parent)
        depth = np_depth
        
        occ_probs = occurence_probability_fast(G, edges, p, np_parent, np_parent_weight, tree_root, np_depth, degree_np)
        for j in range(occ_probs.shape[0]):
            op = occ_probs[j]
            l = depth[op.u] + depth[op.v] - 2*depth[op.lca] + 1
            
            p_c_prime = cexp2(P[l] - op.p_c) / samples
            # DAS HIER IST 2^(log(Pl) - log(op.p_c)) / samples = 2^(log(Pl/op.p_c)) / samples = Pl/op.p_c/samples = Pl / (op.p_c * samples) Also ist p_c_prime = p'_c

            #if not type(p_c_prime) == float:
            #    print(p_c_prime, l, P[l], samples, p_c)

            if p_c_prime > 1:
                undersample += 1
            
            if op.bal == 1: # positive balance
                positive_expected_counts[l] += p_c_prime
                positive_occurred[l] += 1
            else: # negative balance
                negative_expected_counts[l] += p_c_prime
                negative_occurred[l] += 1         
            
    
    free_neighbors_and_weights(n, neighbors, weights)

    np_total_occurred = np_positive_occurred + np_negative_occurred
    np_total_expected_counts = np_positive_expected_counts + np_negative_expected_counts

    total_zeros = np_total_expected_counts == 0
    positive_zeros = np_positive_expected_counts == 0
    negative_zeros = np_negative_expected_counts == 0

    with np.errstate(divide='ignore'):
        np_positive_est_counts = np.exp2(np.log2(np_positive_expected_counts) - np_P)
        np_negative_est_counts = np.exp2(np.log2(np_negative_expected_counts) - np_P)
        np_total_est_counts = np.exp2(np.log2(np_total_expected_counts) - np_P)
    
    return np_total_est_counts, np_positive_est_counts, np_negative_est_counts, np_total_occurred, np_positive_occurred, np_negative_occurred

@cython.boundscheck(False)
@cython.wraparound(False)
cdef OccurenceProb[:] occurence_probability_fast(G: nx.Graph, Edge[:] edges, double p, int[:] parent, signed char[:] parent_weight, int tree_root, int[:] depth, int[:] degree):
    """
    Estimates the probability with which each cycle induced by the given spanning tree appears in a uniformly sampled spanning tree.

    Due to limitations of floating point numbers, it returns the logarithm of the result.

    **Parameters** 
    G:      underlying ER-Graph
    p:      edge probability of G
    parent: parent relationship on the spanning tree T (root has parent -1)
    depth:  distance of node from the root in T

    Returns: List of tuples (u, v, lca(u,v), log_2(p_c)) for each $(u, v) \in G \setminus T$
    """
    cdef int n = len(G.nodes)
    cdef double c_prob = p

    # calculate cumulative properties $\pi(r,u)$ and $\sigma(r,u)$
    cum_prod_np = np.ndarray(len(parent), np.float64)
    root_prod = deg_prod_update(tree_root, -1, 0, degree)
    calc_property_fast(parent, cum_prod_np, root_prod, degree, deg_prod_update)
    cdef double[:] cum_prod = cum_prod_np
    cum_sum_np = np.ndarray(len(parent), np.float64)
    calc_property_fast(parent, cum_sum_np, 0, degree, deg_sum_update)
    cdef double[:] cum_sum = cum_sum_np

    cdef int i
    cdef int num_candidates = len(edges) - n + 1
    np_candidate_edges = np.empty(num_candidates, dtype=NP_EDGE)
    cdef Edge[:] candidate_edges = np_candidate_edges
    cdef Edge e
    i = 0

    for e in edges:
        if parent[e.a] != e.b and parent[e.b] != e.a:
            candidate_edges[i] = e
            i += 1

    cdef int u, v, lca, l
    cdef double deg_sum, deg_prod, p_c
    cdef LcaResult* lca_result = lowest_common_ancestor(parent, parent_weight, candidate_edges)
    cdef int res_count = candidate_edges.shape[0]
    cdef OccurenceProb[:] result
    try:
        np_result = np.ndarray(res_count, dtype=NP_OCCURENCE_PROB)
        result = np_result
        for i in range(res_count):
            lca_res = lca_result[i]
            u = lca_res.a
            v = lca_res.b
            lca = lca_res.lca
            bal = lca_res.bal

            l = depth[u] + depth[v] - 2*depth[lca] + 1

            # $\sigma(u,v) = \sigma(r,u) + \sigma(r,v) - 2 \sigma(r,lca(u,v)) + (d(v) - 1)(d(u) - 1)
            deg_sum = cum_sum[u] + cum_sum[v] - 2*cum_sum[lca]
            deg_sum = deg_sum + (degree[v]-1) * (degree[u]-1)

            # $p'_{l-1} = \frac{1}{1 + \frac{(n-1)p - 2}{n-3}\frac{n-l}{l}}$
            deg_sum = deg_sum / (1 + ((n-1) * c_prob - 2) * (n - l) / (n - 3) / l)

            # $\pi(u,v) = (d(lca(u,v)) - 1) * \pi(r,u) * \pi(r,v) / \pi(r,lca(u,v))^2
            deg_prod = cum_prod[u] + cum_prod[v] + clog2(degree[lca] - 1) - (2*cum_prod[lca])
            
            # mod is the remaining part of the equation
            # [...] ((n-2)/n)^(l-3) (n-1)/n * ((n-1)p-1)/(n-1)p
            mod = clog2(n-1) - clog2(n) + clog2((n-2)/n)*(l-3) + clog2((n-1)*c_prob-1) - clog2((n-1)*c_prob)
            p_c = clog2(deg_sum) - deg_prod + mod

            if p_c > 0: #logarithmic -> actual p_c > 1
                p_c = 0

            result[i] = OccurenceProb(u, v, lca, p_c, bal)
    finally:
        free(lca_result)

    return result

@cython.wraparound(False)
cdef double deg_prod_update(int node, int p, double parent_val, int[:] degree):
    """
    # root r, node u, parent v
    # Degree product: $\pi(r,u) = \pi(r,v) * (d(v) - 1)$
    deg_prod_update = lambda _, p, parent_val: parent_val + (log2(degree_np[p] - 1) if degree_np[p] > 1 else 0) # if the degree is 1 we can ignore the node -- should only happen if the root has degree 1
    """
    return parent_val + (clog2(degree[node] - 1) if degree[node] > 1 else 0)

@cython.wraparound(False)
cdef double deg_sum_update(int u, int v, double parent_val, int[:] degree):
    """
    # Degree sum: $\sigma(r,u) = \sigma(r,v) + (d(v) - 1)(d(u)-1)$
    deg_sum_update = lambda u, v, parent_val: parent_val + (degree_np[u]-1) * (degree_np[v]-1)
    """
    return parent_val + (degree[u]-1) * (degree[v]-1)