import importer

from snakemake.script import Snakemake
import py_raccoon.balance_sampling as bal
import networkx as nx 
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()


p = 0.2
G = nx.gnp_random_graph(50, p)
while not nx.is_connected(G):
    G  = nx.gnp_random_graph(7, p)

for u, v in G.edges():
        G[u][v]['weight'] = random.choice([-1, 1])



np_total_est_counts, np_positive_est_counts, np_negative_est_counts, total_zeros, positive_zeros, negative_zeros, np_total_occurred, np_positive_occurred, np_negative_occurred = bal.estimate_balance(G, 1000)

print("-------------------------------------------")
print(f"np_total_est_counts: {np_total_est_counts}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_positive_est_counts: {np_positive_est_counts}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_negative_est_counts: {np_negative_est_counts}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"total_zeros: {total_zeros}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"positive_zeros: {positive_zeros}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"negative_zeros: {negative_zeros}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_total_occurred: {np_total_occurred}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_positive_occurred: {np_positive_occurred}")
print("-------------------------------------------")
print("-------------------------------------------")
print(f"np_negative_occurred: {np_negative_occurred}")
print("-------------------------------------------")

data = [] 



for l in range(len(np_total_est_counts)):
    data.append({'l': l, 'np_total_est_counts': np_total_est_counts[l], 'np_positive_est_counts': np_positive_est_counts[l], 'np_negative_est_counts': np_negative_est_counts[l], 'total_zeros': total_zeros[l], 'positive_zeros': positive_zeros[l], 'negative_zeros': negative_zeros[l], 'np_total_occurred': np_total_occurred[l], 'np_positive_occurred': np_positive_occurred[l], 'np_negative_occurred': np_negative_occurred[l]})

df_data = pd.DataFrame(data)
df_data.to_csv(snakemake.output[0])