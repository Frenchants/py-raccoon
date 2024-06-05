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

from .spanning_trees cimport lowest_common_ancestor, Edge, LcaResult, calc_property_fast, graph_to_neighbors, free_graph_neighbors, uniform_spanning_tree_c, Edge
from .spanning_trees import NP_EDGE, calc_depth, uniform_spanning_tree, get_induced_cycle

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed
        unsigned int operator()()
    
    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"

#from libcpp.vector cimport vector

cdef packed struct OccurenceProb:
    int u
    int v
    int lca
    double p_c

NP_OCCURENCE_PROB = np.dtype([
    ('u', np.int32),
    ('v', np.int32),
    ('lca', np.int32),
    ('p_c', np.float64),
])

def uniform_cc(n: int, p: float, N: float | NDArray[np.float64] | None = None, P: NDArray | None = None, samples: int = 100, G:nx.Graph | None = None, seed:int|np.random.Generator|None=None, fast_sampling: bool = True) -> tuple[nx.Graph, set[tuple], dict[tuple,float], dict[tuple,float]]:
    """
    Generates a uniform cell complex adhering to the given parameters, as introduced in [1]

    Parameters:
    - n: Number of nodes.
    - p: Edge probability for G(n,p) graph.
    - N: Number of 2-cells (in expectation). Behavior depends on whether N is a single number or an array:
        - For a single number, the sampled CC contains – in expectation – N 2-cells sampled from all available lengths
        - For an array, the sampled CC contains – in expectation – N[l] 2-cells of length l if such cells are found. **The array must have length n + 1**
    - P: Logarithmic (base 2) sampling probability, based on length; must have length `n + 1`. $P_l$ in the paper is equivalent to `exp2(P[l])`.
        - Samples from the model according to its theoretical definition, leading to possibly large variations in the number of cells, even with the same configuration (see [1] for more information). To avoid this, use `N` instead. If `N` is used, `P` must be `None` and vice versa.
        - For $P_l = 0$, set `P[l] = -np.inf`; otherwise `P[l] = log2($P_l$)`.
    - samples: Random spanning trees to sample. Larger is more accurate. Should be greater than `N` (or `np.sum(N)`).
    - seed: Random seed or generator to use. Will generate a new `numpy.random.default_rng` if number or no seed is specified.
    - G: Underlying graph to generate 2-cells for. If `None`, a G(n,p) random graph will be sampled instead. If specified, `p` will still be used for the sampling process.
    - fast_sampling: Uniform CCs can be sampled using a more accurate (slow) or a more inaccurate (fast) algorithm. Roughly, the slow algorithm takes 10s to sample for n²p=500, the fast algorithm takes 10s for n²p=500,000. See the package README or [1] for more details

    Returns: G, cells, undersampled, overcorrelated

    - G: nx.Graph, 1-skeleton of the CC
    - cells: set[tuple[int]], representing 2-cells as normalized tuples
    - undersampled: int, number of 2-cells where $\\rho'_c > 1$
    - overcorrelated: int, number of 2-cells where $\\rho'_c$ is large enough that, in expectation, multiple cells would be sampled from the same ST.

    References
    [1] Hoppe, Josef and Schaub, Michael T. "Random Abstract Cell Complexes." arXiv preprint arXiv:0000.00000 (2024).
    """
    if (N is None) == (P is None):
        raise ValueError("P xor N must be None")
    if fast_sampling:
        return uniform_cc_fast(n, p, N, P, samples, seed, G)
    else:
        return uniform_cc_slow(n, p, N, P, samples, seed, G)

def estimate_cycle_count(G: nx.Graph, samples: int, p: float | None = None, seed:int|np.random.Generator|None=None) -> tuple[NDArray[np.float64], NDArray[bool], NDArray[np.int32]]:
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

    edges = np.array([(u,v) for (u,v) in G.edges], dtype=NP_EDGE)

    return estimate_len_count_fast(G, edges, p, samples, seed)

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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef OccurenceProb[:] occurence_probability_fast(G: nx.Graph, Edge[:] edges, double p, int[:] parent, int tree_root, int[:] depth, int[:] degree):
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
    cdef LcaResult* lca_result = lowest_common_ancestor(parent, candidate_edges)
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

            result[i] = OccurenceProb(u, v, lca, p_c)
    finally:
        free(lca_result)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def uniform_cc_fast(n: int, p: float, N: float | NDArray[np.float64] | None, np_P: NDArray[np.float64] | None = None, samples: int = 100, seed:int|np.random.Generator|None=None, G:nx.Graph | None = None) -> tuple[nx.Graph, set[tuple], int, int]:
    """
    Samples from the cellular ER according to parameters. More accurate, but slower than `cellular_er_estimate_approx`.

    **Parameters**  
    n:          number of nodes for the generated graph
    p:          edge probability for the generated graph
    N:          approximate number of cells to sample (not guaranteed)
    samples:    number of spanning trees to sample. More samples take (linearly) more time, but reduce undersampling and overcorrelation of cells.

    Returns: Graph, sampled cells, dictionary c -> p_c for undersampled and correlated cells
    """
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)
    cdef mt19937 c_rnd = mt19937(seed.integers(0, 1 << 32))
    cdef uniform_real_distribution[double] sampling_dist = uniform_real_distribution[double](0,1)
    if G is None:
        G = nx.gnp_random_graph(n, p, seed)
        while not nx.is_connected(G):
            G = nx.gnp_random_graph(n, p, seed)
    cdef int m = len(G.edges)
    if m - n + 1 == 0:
        print("WARN: Graph is a tree, returning empty result")
        return G, set(), 0, 0
    edges = np.array([(u,v) for (u,v) in G.edges], dtype=NP_EDGE)
    node_degree = np.array(G.degree, dtype=np.int32)
    degree_np = node_degree[:,1][node_degree[:,0].argsort()] # nx may not consider the nodes to be in ascending order by their id
    cdef int[:] degree = degree_np
    cdef int** neighbors = graph_to_neighbors(n, degree, G)

    cdef int c_samples = samples
    cdef char[:] len_count_is_zero
    cdef int est_samples, occuring_lengths
    cdef double log_occuring_lengths
    cdef double[:] len_counts
    if np_P is None and N is not None:
        # Use significantly fewer samples for estimation to save time
        est_samples = c_samples // 10 if c_samples > 100 else c_samples
        np_len_counts, np_len_count_is_zero, sample_counts = estimate_len_count_fast(G, edges, p, est_samples, seed)
        np_len_count_is_zero[np_len_counts < 1] = True #small values make calculation weird; log_2(2) = 1
        np_len_count_is_zero[sample_counts < min(est_samples, 10)] = True #not actually occuring cells are also weird
        occuring_lengths = np.size(np_len_count_is_zero) - np.count_nonzero(np_len_count_is_zero)
        log_occuring_lengths = clog2(occuring_lengths)
        len_counts = np_len_counts
        with np.errstate(divide='ignore'):
            log_N = np.log2(N) # works for both scalars and arrays
        if np.isscalar(N):
            # approx sample count is distributed evenly among all lengths that occur
            np_P = log_N - np_len_counts - log_occuring_lengths # numpy auto broadcasts to arrays where necessary.
        else:
            np_P = log_N - np_len_counts
        len_count_is_zero = np_len_count_is_zero
    else:
        # treat all lengths as existing, the np_P parameter takes precedence (and may be `log2(0)`).
        len_count_is_zero = np.zeros(n, dtype=np.int8)
    cdef double[:] P = np_P

    cells = set()
    undersample = 0
    overcorrelate = 0
    cdef int l
    cdef double p_c, p_c_prime
    cdef OccurenceProb[:] occ_probs
    cdef OccurenceProb op
    cdef int i
    cdef int[:] depth
    # overcorrelation <=> if all cells had this probability, we would sample > 1 from this spanning tree in expectation.
    cdef double overcorrelate_thresh = 1.0 / (m - n + 1)
    for i in range(c_samples):
        parent = np.ndarray(n, dtype=np.int32)
        tree_root = uniform_spanning_tree_c(n, degree, neighbors, parent, seed)
        np_depth = calc_depth(parent)
        depth = np_depth
        
        occ_probs = occurence_probability_fast(G, edges, p, parent, tree_root, np_depth, degree_np)
        for op in occ_probs:
            l = depth[op.u] + depth[op.v] - 2*depth[op.lca] + 1
            if not len_count_is_zero[l]:
                p_c_prime = calc_sampling_probability(P[l], c_samples, op.p_c, log_input=True)
                if p_c_prime > 1:
                    undersample += 1
                elif p_c_prime > overcorrelate_thresh:
                    overcorrelate += 1
                if sampling_dist(c_rnd) < p_c_prime:
                    cells.add(get_induced_cycle((op.u, op.v), parent, depth))

    free_graph_neighbors(n, neighbors)
    return G, cells, undersample, overcorrelate

@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_len_count_fast(G: nx.Graph, edges: np.ndarray[NP_EDGE], p: float, samples: int, seed: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[bool], NDArray[np.int32]]:
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
    cdef int** neighbors = graph_to_neighbors(n, degree, G)
    np_expected_counts = np.zeros(n + 1, np.float64)
    cdef double[:] expected_counts = np_expected_counts
    np_occured = np.zeros(n + 1, np.int32)
    cdef int[:] occured = np_occured
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
        tree_root = uniform_spanning_tree_c(n, degree, neighbors, parent, seed)
        np_depth = calc_depth(parent)
        depth = np_depth
        
        occ_probs = occurence_probability_fast(G, edges, p, np_parent, tree_root, np_depth, degree_np)
        for j in range(occ_probs.shape[0]):
            op = occ_probs[j]
            l = depth[op.u] + depth[op.v] - 2*depth[op.lca] + 1
            
            p_c_prime = cexp2(P[l] - op.p_c) / samples

            #if not type(p_c_prime) == float:
            #    print(p_c_prime, l, P[l], samples, p_c)

            if p_c_prime > 1:
                undersample += 1
            expected_counts[l] += p_c_prime
            occured[l] += 1
    
    free_graph_neighbors(n, neighbors)

    zeros = np_expected_counts == 0

    with np.errstate(divide='ignore'):
        np_est_counts = np.log2(np_expected_counts) - np_P
    return np_est_counts, zeros, np_occured

def last_step(d, n, l):
    return 1 / (1 + (d-2)*(n-l)/(n-3)/l)

@cython.cpow(True)
cdef inline double calc_sampling_probability(float P, int samples, float p_c, char log_input = False) nogil:
    """Calculates the probability with which we should sample the given cell obtained from a spanning tree.
    
    **Parameters**
    P:          Overall Probability with which we should choose the cell
    samples:    Number of sampled spanning trees
    p_c:        Probability that a uniformly sampled spanning tree induces the cell
    log_input:  Whether the input is the logarithm of P and p_c or not
    """
    if log_input:
        if p_c + clog2(samples) < clog2(.001):
            # very close to correct; exp2(p_c) may be 0 due to floating point limitations
            return cexp2(P - p_c - clog2(samples))
        else:
            # have to be more accurate; fortunately, exp2(p_c) will be > 0
            P = cexp2(P)
            p_c = cexp2(p_c)
    if p_c * samples < 0.001 or P >= 1:
        # floating point calculation errors; here, this is very close to correct
        return P / p_c / samples
    return (1-(1-P)**(1/samples)) / p_c

@cython.boundscheck(False)
@cython.wraparound(False)
def uniform_cc_slow(n: int, p: float, N: float | NDArray[float] | None, np_P: NDArray[np.float64] | None = None, samples: int = 100, seed:int|np.random.Generator|None=None, G:nx.Graph | None = None) -> tuple[nx.Graph, set[tuple], int, int]:
    """
    Samples from the cellular ER according to parameters. More accurate, but slower than `cellular_er_estimate_approx`.

    **Parameters**  
    n:          number of nodes for the generated graph
    p:          edge probability for the generated graph
    N:          approximate number of cells to sample (not guaranteed)
    samples:    number of spanning trees to sample. More samples take (linearly) more time, but reduce undersampling and overcorrelation of cells.

    Returns: Graph, sampled cells, dictionary c -> p_c for undersampled and correlated cells
    """
    if seed is None:
        seed = np.random.default_rng()
    elif isinstance(seed, int):
        seed = np.random.default_rng(seed)
    if G is None:
        G = nx.gnp_random_graph(n, p, seed)
        while not nx.is_connected(G):
            G = nx.gnp_random_graph(n, p, seed)
    m = len(G.edges)
    edges = np.array([(u,v) for (u,v) in G.edges], dtype=NP_EDGE)
    cdef Edge[:] c_edges = edges

    if np_P is None and N is not None:
        # Use significantly fewer samples for estimation to save time
        est_samples = samples // 10 if samples > 100 else samples
        log_len_counts, len_counts_is_zero, sample_counts = estimate_len_count_fast(G, edges, p, est_samples, seed)
        len_counts = np.exp2(log_len_counts)
        len_counts[len_counts_is_zero] = 0
        len_counts[len_counts < 2] = 0 #small values make calculation weird
        len_counts[sample_counts < est_samples] = 0 #not actually occuring cells are also weird
        occuring_lengths = np.count_nonzero(len_counts)

        np_P = np.zeros_like(log_len_counts)
        # divide by 0 will occur, but is fixed afterwards. invalid occurs on 0/0, which will occur in the second case.
        with np.errstate(divide='ignore', invalid='ignore'):
            if np.isscalar(N):
                # approx sample count is distributed evenly among all lengths that occur
                np_P = N / len_counts / occuring_lengths # numpy auto broadcasts to arrays where necessary.
                np_P[len_counts == 0] = 0
            else:
                np_P = N - len_counts
                np_P[len_counts == 0] = 0
    else:
        # np_P is logarithmic for compatibility
        np_P = np.exp2(np_P)
    cdef double[:] P = np_P

    cells = set()
    cdef int undersample = 0
    cdef int overcorrelate = 0
    cdef int i, u, v
    cdef Edge e
    cdef int[:] parent, depth
    cdef double p_c, p_c_prime
    for i in range(samples):
        np_parent = uniform_spanning_tree(G, seed)
        parent = np_parent
        np_depth = calc_depth(parent)
        depth = np_depth
        for e in c_edges:
            u = e.a
            v = e.b
            if parent[u] != v and parent[v] != u:
                cycle = get_induced_cycle((u,v), np_parent, np_depth)
                if cycle not in cells:
                    deg_sum = 0
                    deg_prod = 1
                    l = len(cycle)
                    for u in cycle:
                        deg_prod *= G.degree[u] - 1
                    for u,v,v_prime in zip(cycle, cycle[1:] + cycle[:1], cycle[2:] + cycle[:2]):
                        deg_sum += (G.degree[u]-1) * (G.degree[v]-1) / G.degree[u] * (G.degree[v_prime]-1) * last_step(G.degree[v_prime], n, l)
                    mod = (n-1)/n * ((n-2)/n)**(l-3)
                    p_c = deg_sum / deg_prod * mod
                    p_c_prime = calc_sampling_probability(P[l], samples, p_c)
                    if p_c_prime > 1:
                        undersample += 1
                    elif p_c_prime * (m-n) > 1:
                        overcorrelate += 1
                    if seed.random() < p_c_prime:
                        cells.add(cycle)
    return G, cells, undersample, overcorrelate