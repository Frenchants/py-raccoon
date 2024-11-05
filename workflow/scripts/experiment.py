import networkx as nx 
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import time
import datetime
import tracemalloc
from pympler import asizeof 
import sys

import utils
from snakemake.script import Snakemake # type: ignore
import py_raccoon.balance_sampling as pyr
import cycleindex as cx

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

def save_pyr_results(total_est, pos_est, neg_est, total_occurred, pos_occurred, neg_occurred):
    
    pos_degree_of_bal = np.nan_to_num(pos_est / total_est, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    neg_degree_of_bal = np.nan_to_num(neg_est / total_est, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    rel_signed_clust_coeff = np.nan_to_num((pos_est - neg_est) / total_est, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    pos_to_neg_ratio = np.nan_to_num(pos_est / neg_est, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    neg_to_pos_ratio = np.nan_to_num(neg_est / pos_est, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    total_zeros = total_est == 0
    pos_zeros = pos_est == 0
    neg_zeros = neg_est == 0

    data = []
    for l in range(len(total_est)):
            data.append({'l': l, 'pos_degree_of_bal': pos_degree_of_bal[l], 'neg_degree_of_bal': neg_degree_of_bal[l], 'rel_signed_clust_coeff': rel_signed_clust_coeff[l], 'pos_to_neg_ratio': pos_to_neg_ratio[l], 'neg_to_pos_ratio': neg_to_pos_ratio[l], 'total_est': total_est[l], 'pos_est': pos_est[l], 'neg_est': neg_est[l], 'total_zeros': total_zeros[l], 'pos_zeros': pos_zeros[l], 'neg_zeros': neg_zeros[l], 'total_occurred': total_occurred[l], 'pos_occurred': pos_occurred[l], 'neg_occurred': neg_occurred[l]})
    
    total_sum_cycles = np.sum(total_est)
    pos_sum_cycles = np.sum(pos_est)
    neg_sum_cycles = np.sum(neg_est)

    pos_degree_of_bal_graph = pos_sum_cycles / total_sum_cycles
    neg_degree_of_bal_graph = neg_sum_cycles / total_sum_cycles

    data[0]['pos_degree_of_bal_graph'] = pos_degree_of_bal_graph
    data[0]['neg_degree_of_bal_graph'] = neg_degree_of_bal_graph

    return data

def save_cx_results(plus_minus, plus_plus):
   
    plus_plus = np.asarray(plus_plus)
    plus_minus = np.asarray(plus_minus)
    
    if plus_minus.ndim == 1:
        plus_minus = np.expand_dims(plus_minus, 0)
        plus_plus = np.expand_dims(plus_plus, 0)
    elif plus_minus.ndim == 3:
        plus_minus = np.concatenate(plus_minus)
        plus_plus = np.concatenate(plus_plus)

    if not directed_graph:
         plus_plus /= 2
         plus_minus /= 2
    
    # insert zeros for cycle length 0 because these are included in the 'pyr' alg as well
    plus_plus = np.array([np.insert(arr, 0, 0) for arr in plus_plus])
    plus_minus = np.array([np.insert(arr, 0, 0) for arr in plus_minus])

    pos = (plus_plus + plus_minus) / 2
    neg = plus_plus - pos
    neg_degree_of_bal = np.nan_to_num(neg / plus_plus, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    pos_degree_of_bal = np.nan_to_num(pos / plus_plus, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    neg_to_pos_ratio = np.nan_to_num(neg / pos, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    pos_to_neg_ratio = np.nan_to_num(pos / neg, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    rel_signed_clust_coeff = np.nan_to_num(plus_minus / plus_plus, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    std_total = np.zeros(plus_plus.shape[1], dtype=plus_plus.dtype)
    avg_total = np.zeros(plus_plus.shape[1], dtype=plus_plus.dtype)
    std_pos = np.zeros(pos.shape[1], dtype=pos.dtype)
    avg_pos = np.zeros(pos.shape[1], dtype=pos.dtype)
    std_neg = np.zeros(neg.shape[1], dtype=neg.dtype)
    avg_neg = np.zeros(neg.shape[1], dtype=neg.dtype)
    std_pos_degree_of_bal = np.zeros(pos_degree_of_bal.shape[1], dtype=pos_degree_of_bal.dtype)
    avg_pos_degree_of_bal = np.zeros(pos_degree_of_bal.shape[1], dtype=pos_degree_of_bal.dtype)
    std_neg_degree_of_bal = np.zeros(neg_degree_of_bal.shape[1], dtype=neg_degree_of_bal.dtype)
    avg_neg_degree_of_bal = np.zeros(neg_degree_of_bal.shape[1], dtype=neg_degree_of_bal.dtype)
    std_pos_to_neg_ratio = np.zeros(pos_to_neg_ratio.shape[1], dtype=pos_to_neg_ratio.dtype)
    avg_pos_to_neg_ratio = np.zeros(pos_to_neg_ratio.shape[1], dtype=pos_to_neg_ratio.dtype)
    std_neg_to_pos_ratio = np.zeros(neg_to_pos_ratio.shape[1], dtype=neg_to_pos_ratio.dtype)
    avg_neg_to_pos_ratio = np.zeros(neg_to_pos_ratio.shape[1], dtype=neg_to_pos_ratio.dtype)
    std_rel_signed_clust_coeff = np.zeros(rel_signed_clust_coeff.shape[1], dtype=rel_signed_clust_coeff.dtype)
    avg_rel_signed_clust_coeff = np.zeros(rel_signed_clust_coeff.shape[1], dtype=rel_signed_clust_coeff.dtype)
    
    std_total = np.std(plus_plus, axis=0)
    avg_total = np.mean(plus_plus, axis=0)
    std_pos = np.std(pos, axis=0)
    avg_pos = np.mean(pos, axis=0)
    std_neg = np.std(neg, axis=0)
    avg_neg = np.mean(neg, axis=0)
    std_pos_degree_of_bal = np.std(pos_degree_of_bal, axis=0)
    avg_pos_degree_of_bal = np.mean(pos_degree_of_bal, axis=0)
    std_neg_degree_of_bal = np.std(neg_degree_of_bal, axis=0)
    avg_neg_degree_of_bal = np.mean(neg_degree_of_bal, axis=0)
    std_pos_to_neg_ratio = np.std(pos_to_neg_ratio, axis=0)
    avg_pos_to_neg_ratio = np.mean(pos_to_neg_ratio, axis=0)
    std_neg_to_pos_ratio = np.std(neg_to_pos_ratio, axis=0)
    avg_neg_to_pos_ratio = np.mean(neg_to_pos_ratio, axis=0)
    std_rel_signed_clust_coeff = np.std(rel_signed_clust_coeff, axis=0)
    avg_rel_signed_clust_coeff = np.mean(rel_signed_clust_coeff, axis=0)

    zeros_total = avg_total == 0
    zeros_pos = avg_pos == 0
    zeros_neg = avg_neg == 0

    data = []
    for l in range(len(avg_total)):
            data.append({'l': l, 'avg_pos_degree_of_bal': avg_pos_degree_of_bal[l], 'avg_neg_degree_of_bal': avg_neg_degree_of_bal[l], 'avg_rel_signed_clust_coeff': avg_rel_signed_clust_coeff[l], 'avg_pos_to_neg_ratio': avg_pos_to_neg_ratio[l], 'avg_neg_to_pos_ratio': avg_neg_to_pos_ratio[l], 'avg_total': avg_total[l], 'avg_pos': avg_pos[l], 'avg_neg': avg_neg[l], 'zeros_total': zeros_total[l], 'zeros_pos': zeros_pos[l], 'zeros_neg': zeros_neg[l], 'std_pos_degree_of_bal': std_pos_degree_of_bal[l], 'std_neg_degree_of_bal': std_neg_degree_of_bal[l], 'std_rel_signed_clust_coeff': std_rel_signed_clust_coeff[l], 'std_pos_to_neg_ratio': std_pos_to_neg_ratio[l], 'std_neg_to_pos_ratio': std_neg_to_pos_ratio[l], 'std_total': std_total[l], 'std_pos': std_pos[l], 'std_neg': std_neg[l]})
    
    total_sum_cycles = np.sum(plus_plus, axis=1)
    pos_sum_cycles = np.sum(pos, axis=1)
    neg_sum_cycles = np.sum(neg, axis=1)
    
    avg_pos_degree_of_bal_graph = np.mean(np.nan_to_num(pos_sum_cycles / total_sum_cycles, copy=False, nan=0.0, posinf=0.0, neginf=0.0), axis=0)
    avg_neg_degree_of_bal_graph = np.mean(np.nan_to_num(neg_sum_cycles / total_sum_cycles, copy=False, nan=0.0, posinf=0.0, neginf=0.0), axis=0)

    data[0]['avg_pos_degree_of_bal_graph'] = avg_pos_degree_of_bal_graph
    data[0]['avg_neg_degree_of_bal_graph'] = avg_neg_degree_of_bal_graph
    
    return data

def save_exp_params(data):
    data[0]['run'] = run
    data[0]['kind'] = kind
    if kind == 'er':
         data[0]['prob_p'] = prob_p
         data[0]['prob_r'] = prob_r
    data[0]['n_nodes'] = n_nodes
    data[0]['n_edges'] = n_edges
    data[0]['directed_graph'] = directed_graph
    data[0]['alg'] = alg
    data[0]['n_samples'] = n_samples
    if alg == 'cx': 
         data[0]['exact'] = exact
         data[0]['parallel'] = parallel
         data[0]['matrix_mem'] = matrix_mem
    if alg == 'pyr':
         data[0]['specified_edge_prob'] = specified_edge_prob
    data[0]['alg_run_time'] = alg_end_time - alg_start_time 
    data[0]['alg_peak_mem'] = alg_peak_mem
    data[0]['alg_only_mem'] = alg_peak_mem - alg_before_mem
    data[0]['before_alg_mem'] = alg_before_mem
    data[0]['seed'] = seed
    data[0]['script_run_time'] = script_end_time - script_start_time
    data[0]['exp_date'] = exp_date

    return data

if __name__ == "__main__":

    script_start_time = time.time()
    tracemalloc.start()
    exp_date = datetime.datetime.now()

    # Suppress invalid value warnings (e.g., divide by zero or NaN)
    np.seterr(divide='ignore', invalid='ignore')

    SEEDS = [130250193, 4259459283, 1535995910, 1779285243, 347749838, 2843683941, 1121168978, 1281646427, 2640136520, 1846011679, 2548891259, 889869862, 816974034, 898432604, 3953502902, 480242123, 1422106884, 2844494128, 2766920156, 3963647525, 2800650013, 3026969699, 617436447, 3125820586, 968979614, 3011594777, 2949848623, 2343270211, 1911319159, 678032221, 2644994550, 2694585517, 876264514, 1420930522, 1191847850, 3452672408, 694543404, 429751076, 1464333811, 2794718515, 3303745309, 2176095271, 1235875709, 2083610798, 2992731158, 1458240102, 3463342733, 2894203811, 1901449950, 807625046]

    snakemake = fix_smk()
    run = int(snakemake.wildcards['run'])
    kind = snakemake.wildcards['kind']
    n_nodes = int(snakemake.wildcards['n_nodes'])
    prob_p = float(snakemake.wildcards['prob_p'])
    prob_r = float(snakemake.wildcards['prob_r'])
    alg = snakemake.wildcards['alg'] # 'cx' or 'pyr'
    specified_edge_prob = float(snakemake.wildcards.get('specified_edge_prob', prob_p))
    exact = bool(snakemake.wildcards.get('exact', False))
    parallel = bool(snakemake.wildcards.get('parallel', False))
    n_samples = int(snakemake.wildcards['n_samples'])
    #directed_graph = bool(snakemake.wildcards.get('directed_graph', False))
    

    seed = SEEDS[run-1]
    random.seed(seed)
    np.random.seed(seed)
    rnd = np.random.default_rng(seed)

    if kind == 'er':
        directed_graph = False
        G = nx.gnp_random_graph(n_nodes, prob_p, seed=rnd)
        while not nx.is_connected(G):
            G  = nx.gnp_random_graph(n_nodes, prob_p, seed=rnd)
        for u, v in G.edges():
            G[u][v]['weight'] = rnd.choice([-1, 1], p=[prob_r, 1-prob_r])
    
    n_edges = G.number_of_edges()

    if alg == 'pyr':
         alg_before_mem, alg_before_peak_mem = tracemalloc.get_traced_memory()
         tracemalloc.reset_peak()
         alg_start_time = time.time()
         results = pyr.estimate_balance(G, samples=n_samples, p=prob_p, seed=rnd)
         alg_end_time = time.time()
         after_mem, alg_peak_mem = tracemalloc.get_traced_memory()
         data = save_pyr_results(*results)
    elif alg == 'cx':
         adj_matrix = cx.clean_matrix(cx.to_adj_matrix(G))
         matrix_mem = adj_matrix.nbytes
         alg_before_mem, _ = tracemalloc.get_traced_memory()
         tracemalloc.reset_peak()
         alg_start_time = time.time()
         results = cx.balance_ratio(adj_matrix, 10, exact=exact, n_samples=n_samples, parallel=parallel)
         alg_end_time = time.time()
         after_mem, alg_peak_mem = tracemalloc.get_traced_memory()
         data = save_cx_results(*results)
    else:
        raise ValueError("The only available algorithms are 'pyr' and 'cx'.")

    tracemalloc.stop()
    script_end_time = time.time()

    data = save_exp_params(data)
    df_data = pd.DataFrame(data)
    df_data.to_csv(snakemake.output[0])

    # Ordnerstruktur: 
    # results
    # synthetic_data / dataset1... /
    # dann die kind (also z.B. 'er' oder 'sbm')
    # dann die Parameter im Überordner
    # dann die einzelnen .csv Dateien mit dem 'run' als Name. Also einfach durchnummeriert 1, 2, 3 etc. 


    # ToDo für Dienstag: 
    # Überlegen, welche Graphen du testest