# 🦝 PyRaCCooN: Random Cell Complexes on Networks

<img align="right" width="200" style="margin-top:-5px" src="https://raw.githubusercontent.com/josefhoppe/py-raccoon/main/readme_src/LOGO_ERC-FLAG_FP.png">

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/josefhoppe/py-raccoon/blob/main/LICENSE)
[![arXiv:2406.01999](https://img.shields.io/badge/arXiv-2406.01999-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2406.01999)
[![Package version on PyPI](https://img.shields.io/pypi/v/py-raccoon?logo=pypi&logoColor=ffd242)](https://pypi.org/project/py-raccoon/)

PyRaCCooN (**Ra**ndom **C**ell **Co**mplexes **o**n **N**etworks) randomly generates cell complexes and and provides an approximation for the number of simple cycles (by length) on a graph.
To see how to use PyRaCCooN, check out the Jupyter [examples](https://github.com/josefhoppe/py-raccoon/tree/main/examples) or the short examples below.

For more information on the theory and algorithmics, see our paper [*Random Abstract Cell Complexes*](https://arxiv.org/abs/2406.01999) on arXiv.
The [Evaluation Code](https://github.com/josefhoppe/random-abstract-cell-complexes) is also available on Github.

More specifically, it

- generates random cell complexes by sampling an Erdös-Rényi Graph and random 2-cells, or
- samples random 2-cells on arbitrary graphs.

Note that the sampling algorithm is approximate and designed to work on ER graphs, so the distribution of cycles sampled on other graphs may be less accurate.
Our aforementioned paper contains some analysis of the accuracy on non-ER graphs.

If you use PyRaCCooN, please cite the following paper:

```
@misc{hoppe2024random,
      title={Random Abstract Cell Complexes}, 
      author={Josef Hoppe and Michael T. Schaub},
      year={2024},
      eprint={2406.01999},
      archivePrefix={arXiv},
      primaryClass={cs.DS}
}
```

## Installation

```bash
pip install py-raccoon
```

## Generating Random Cell Complexes (by expected number of 2-cells)

PyRaCCooN uses `NetworkX` to represent the underlying graph of the resulting cell complex.
The 2-cells are represented as a list of tuples of nodes representing the boundary.
Tuples are normalized to start with the smallest node (by integer label), continuing with its smallest neighbor.
For example, `(3,2,1,4)` would be normalized to `(1,2,3,4)`.

```py
import py_raccoon as pr

# Generates CC based on G(20,0.5) with (in expectation) 50 cells, sampled using 100 spanning trees.
G, cells, _, _ = pr.uniform_cc(20, 0.5, 50, samples=100)
```

You may also specify an expected number of 2-cells for each length. Note that longer cells may be impossible to sample. In that case, the supplied expected number is ignored.

```py
import py_raccoon as pr

n = 20
N = np.zeros(n + 1)
N[3] = 10
N[9] = 10
N[19] = 10 # Cells of length 19 will be impossible to sample, so the result will not contain any. See our paper for more information.

# Generates CC based on G(20,0.5) with (in expectation) N[l] cells of length l, sampled using 100 spanning trees.
G, cells, _, _ = pr.uniform_cc(n, 0.5, N, samples=100)
```

If you have a graph you'd like to add random 2-cells to, you can also supply the graph:

```py
import py_raccoon as pr

G = ... # nx.Graph
n, p = pr.utils.estimate_er_params(G)

_, cells, _, _ = pr.uniform_cc(n, p, 50, samples=100, G=G)
```

Since `cells` is a list of tuples, the result can easily be imported into any library of your choosing.

## Generating Random Cell Complexes with given probability $P_l$

The previous examples all used the integrated functionality to sample an expected number of cells.
While this is more useful in many practical contexts, PyRaCCooN also supports sampling from the 'vanilla' model with a fixed probability $P_l$ for all cells of length $l$:

```py
import py_raccoon as pr
import numpy as np

n = 20
P = np.zeros(n + 1)
P[3] = .5
P[4] = .1
P[19] = 1.0 # Cells of length 19 will be impossible to sample, so the result will not contain any. See our paper for more information.

with np.errstate(divide='ignore'):
    log_P = np.log2(P) # Practical probabilities for greater $l$ are very small, thus represented logarithmically.

# Generates CC based on G(20,0.5). Each possible cell of length l is selected with probability P[l], sampled using 100 spanning trees.
G, cells, _, _ = pr.uniform_cc(n, 0.5, P=log_P, samples=100)
```

## Estimating the number of simple cycles

```py
import py_raccoon as pr
import numpy as np

G = ... # nx.Graph
log_counts, is_zero, sampled = pr.estimate_cycle_count(G, samples=1000)

# Assuming all cycle counts are in the range of 64-bit floats
cycle_counts = np.exp2(log_counts)
cycle_counts[is_zero] = 0
for l in range(3, len(G.nodes) + 1):
    print(f'G has approx. {cycle_counts[l]} simple cycles of length {l} ({sampled[l]} samples).')
```

When using the estimation, you should also check the number of sampled cells for each length.
As a rule of thumb, if `sampled[l] < 100`, the estimation for length $l$ is inaccurate.
Also note that eventually, longer cycles won't occur at all, leading to an incorrect estimation of 0.

## Runtime behavior

PyRaCCooN is both algorithmically optimized and efficiently implemented using Cython.
The runtime figure below is from our paper; it shows the average runtime for one run of `pr.uniform_cc(n, p, N=10*n, samples=1000)` for both fast and slow sampling.

![Runtime Behavior](https://raw.githubusercontent.com/josefhoppe/py-raccoon/main/readme_src/runtime.svg)

## Acknowledgements

Funded by the European Union (ERC, HIGH-HOPeS, 101039827). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
