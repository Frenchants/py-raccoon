import pandas as pd
import seaborn as sns
from snakemake.script import Snakemake
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import networkx as nx
from collections import defaultdict

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()



