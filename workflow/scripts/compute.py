import networkx as nx 
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd

import utils
from snakemake.script import Snakemake # type: ignore
from py_raccoon.balance_sampling import estimate_balance
from py_raccoon.sampling import estimate_cycle_count
from cycleindex import balance_ratio, clean_matrix, to_adj_matrix, vxsampling


def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

def process_cx_results(plus_plus, plus_minus):
   
    plus_plus = np.asarray(plus_plus)
    plus_minus = np.asarray(plus_minus)
    
    if plus_minus.ndim == 1:
        plus_minus = np.expand_dims(plus_minus, 0)
        plus_plus = np.expand_dims(plus_plus, 0)
    elif plus_minus.ndim == 3:
        plus_minus = np.concatenate(plus_minus)
        plus_plus = np.concatenate(plus_plus)

    pos = (plus_plus + plus_minus) / 2
    neg = plus_plus - pos

    std_total = np.empty(plus_plus.shape[1] + 1, dtype=plus_plus.dtype)
    avg_total = np.empty(plus_plus.shape[1] + 1, dtype=plus_plus.dtype)
    std_pos = np.empty(pos.shape[1] + 1, dtype=pos.dtype)
    avg_pos = np.empty(pos.shape[1] + 1, dtype=pos.dtype)
    std_neg = np.empty(neg.shape[1] + 1, dtype=neg.dtype)
    avg_neg = np.empty(neg.shape[1] + 1, dtype=neg.dtype)

    std_total[0] = avg_total[0] = std_pos[0] = avg_pos[0] = std_neg[0] = avg_neg[0] = 0

    std_total[1:] = np.std(plus_plus, axis=0)
    avg_total[1:] = np.mean(plus_plus, axis=0)
    std_pos[1:] = np.std(pos, axis=0)
    avg_pos[1:] = np.mean(pos, axis=0)
    std_neg[1:] = np.std(neg, axis=0)
    avg_neg[1:] = np.mean(neg, axis=0)

    zeros_total = avg_total == 0
    zeros_pos = avg_pos == 0
    zeros_neg = avg_neg == 0

    return avg_total, avg_pos, avg_neg, zeros_total, zeros_pos, zeros_neg, std_total, std_pos, std_neg


if __name__ == "__main__":

    snakemake = fix_smk()
        
    SEEDS = [130250193, 4259459283, 1535995910, 1779285243, 347749838, 2843683941, 1121168978, 1281646427, 2640136520, 1846011679, 2548891259, 889869862, 816974034, 898432604, 3953502902, 480242123, 1422106884, 2844494128, 2766920156, 3963647525, 2800650013, 3026969699, 617436447, 3125820586, 968979614, 3011594777, 2949848623, 2343270211, 1911319159, 678032221, 2644994550, 2694585517, 876264514, 1420930522, 1191847850, 3452672408, 694543404, 429751076, 1464333811, 2794718515, 3303745309, 2176095271, 1235875709, 2083610798, 2992731158, 1458240102, 3463342733, 2894203811, 1901449950, 807625046]

    """ run = int(snakemake.wildcards['run'])
    samples = int(snakemake.wildcards['samples'])
    kind = snakemake.wildcards['kind']
    n_nodes = int(snakemake.wildcards['n_nodes'])
    prob_p = int(snakemake.wildcards['prob_p'])
    prob_r = int(snakemake.wildcards['prob_r']) 
    alg = snakemake.wildcards['alg']
    directed = snakemake.wildcards['directed'] """

    run = 1
    samples = 100
    kind = 'er'
    n_nodes = 20
    prob_p = 0.5
    prob_r = 0.5
    #alg = "cx" # pyr for py_raccoon or cx for cycleindex
    alg = snakemake.wildcards['alg']
    directed = False

    seed = SEEDS[run]
    random.seed(seed)
    np.random.seed(seed)
    rnd = np.random.default_rng(seed)

    if not directed:
        if kind == 'er':
            G = nx.gnp_random_graph(n_nodes, prob_p, seed=rnd)
            while not nx.is_connected(G):
                G  = nx.gnp_random_graph(n_nodes, prob_p, seed=rnd)
            for u, v in G.edges():
                G[u][v]['weight'] = rnd.choice([-1, 1], p=[prob_r, 1-prob_r])
        


    data = []


    if alg == 'pyr':

        total, pos, neg, total_occurred, pos_occurred, neg_occurred = estimate_balance(G, samples=samples, p=prob_p, seed=rnd)

    elif alg == 'cx':
        
        adj_matrix = clean_matrix(to_adj_matrix(G))
        plus_plus, plus_minus = balance_ratio(adj_matrix, 10, exact=False, n_samples=samples, parallel=False)

        avg_total, avg_pos, avg_neg, zeros_total, zeros_pos, zeros_neg, std_total, std_pos, std_neg = process_cx_results(plus_plus, plus_minus)

        for l in range(len(avg_total)):
            data.append({'l': l, 'avg_total': avg_total[l], 'avg_pos': avg_pos[l], 'avg_neg': avg_neg[l], 'zeros_total': zeros_total[l], 'zeros_pos': zeros_pos[l], 'zeros_neg': zeros_neg[l], 'std_total': std_total[l], 'std_pos': std_pos[l], 'std_neg': std_neg[l]})


    
    df_data = pd.DataFrame(data)
    df_data.to_csv(snakemake.output[0])