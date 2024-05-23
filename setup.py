from setuptools import setup, Extension

extensions = [
    Extension(
        "py_raccoon.spanning_trees",
        ["src/py_raccoon/spanning_trees.pyx"],
    ),
    Extension(
        "py_raccoon.sampling",
        ["src/py_raccoon/sampling.pyx"],
        include_dirs=['src/']
    ),
]

# This is the function that is executed
setup(
    ext_modules = extensions,
)