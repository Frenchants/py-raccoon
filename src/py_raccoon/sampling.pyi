
import numpy as np
from numpy.typing import NDArray
import networkx as nx

def uniform_cc(n: int, p: float, N: float | NDArray[np.float64] | None = None, P: NDArray | None = None, samples: int = 100, G:nx.Graph | None = None, seed:int|np.random.Generator|None=None, fast_sampling: bool = True) -> tuple[nx.Graph, set[tuple], dict[tuple,float], dict[tuple,float]]:
    """
    Generates a uniform cell complex adhering to the given parameters, as introduced in [1].

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
    [1] Hoppe, Josef and Schaub, Michael T. "Random Abstract Cell Complexes." arXiv preprint, arXiv:0000.00000 (2024).
    """
    ...


def estimate_cycle_count(G: nx.Graph, samples: int, p: float | None = None, seed:int|np.random.Generator|None=None) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.int32]]:
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
    ...