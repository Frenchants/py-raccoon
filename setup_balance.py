from setuptools import setup, Extension
import numpy as np

# path to np headers
np_include = np.get_include()

extensions = [
    Extension(
        "py_raccoon.balance_sampling",
        ["src/py_raccoon/balance_sampling.pyx"],
        include_dirs=['src/', np_include]),
    Extension(
        "py_raccoon.balance_spanning_trees",
        ["src/py_raccoon/balance_spanning_trees.pyx"],
        include_dirs=['src/', np_include]),
]

# This is the function that is executed
setup(
    ext_modules = extensions,
)