from setuptools import setup, Extension

extensions = [
    Extension(
        "py_raccoon.balance_sampling",
        ["src/py_raccoon/balance_sampling.pyx"],
        include_dirs=['src/']),
    Extension(
        "py_raccoon.balance_spanning_trees",
        ["src/py_raccoon/balance_spanning_trees.pyx"],
        include_dirs=['src/']),
]

# This is the function that is executed
setup(
    ext_modules = extensions,
)