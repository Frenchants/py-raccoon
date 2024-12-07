import networkx as nx 
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import time
import datetime
import tracemalloc
import warnings
import math
import scipy

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
    
    pos_k_bal = np.nan_to_num(pos_est / total_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    neg_k_bal = np.nan_to_num(neg_est / total_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    rel_signed_clust_coeff = np.nan_to_num((pos_est - neg_est) / total_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pos_to_neg_ratio = np.nan_to_num(pos_est / neg_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    neg_to_pos_ratio = np.nan_to_num(neg_est / pos_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)

    total_zeros = total_est == 0
    pos_zeros = pos_est == 0
    neg_zeros = neg_est == 0

    data = []
    for l in range(len(total_est)):
            data.append({'l': l, 'pos_k_bal': pos_k_bal[l], 'neg_k_bal': neg_k_bal[l], 'rel_signed_clust_coeff': rel_signed_clust_coeff[l], 'pos_to_neg_ratio': pos_to_neg_ratio[l], 'neg_to_pos_ratio': neg_to_pos_ratio[l], 'total_est': total_est[l], 'pos_est': pos_est[l], 'neg_est': neg_est[l], 'total_zeros': total_zeros[l], 'pos_zeros': pos_zeros[l], 'neg_zeros': neg_zeros[l], 'total_occurred': total_occurred[l], 'pos_occurred': pos_occurred[l], 'neg_occurred': neg_occurred[l]})
    

    avg_pos_k_balance = np.nanmean(pos_k_bal)
    avg_neg_k_balance = np.nanmean(neg_k_bal)

    data[0]['avg_pos_k_balance'] = avg_pos_k_balance
    data[0]['avg_neg_k_balance'] = avg_neg_k_balance


    k = np.arange(1, len(total_est) + 1)

    weight_names = ["degree_of_bal", "weighted_1_k_bal", "weighted_1_k_2_bal", "weighted_1_k_3_bal", "weighted_1_k_4_bal", "weighted_1_k_fac_bal"]
    weight_functions = [1, 1 / k, 1 / k**2, 1 / k**3, 1 / k**4, 1 / scipy.special.factorial(k)]

    for i, weights in enumerate(weight_functions):
        weighted_total_sum_cycles = np.nansum(total_est * weights)
        weighted_pos_sum_cycles = np.nansum(pos_est * weights)
        weighted_neg_sum_cycles = np.nansum(neg_est * weights)

        pos_weighted_degree = np.nan_to_num(weighted_pos_sum_cycles / weighted_total_sum_cycles, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
        neg_weighted_degree = np.nan_to_num(weighted_neg_sum_cycles / weighted_total_sum_cycles, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
        
        measure_name_pos = f"pos_{weight_names[i]}"
        measure_name_neg = f"neg_{weight_names[i]}"

        data[0][measure_name_pos] = pos_weighted_degree
        data[0][measure_name_neg] = neg_weighted_degree

    return data

def save_cx_results(plus_minus, plus_plus):
   
    plus_plus = np.asarray(plus_plus, dtype=np.float64)
    plus_minus = np.asarray(plus_minus, dtype=np.float64)
    
    if plus_minus.ndim == 1:
        plus_minus = np.expand_dims(plus_minus, 0)
        plus_plus = np.expand_dims(plus_plus, 0)
    elif plus_minus.ndim == 3:
        plus_minus = np.concatenate(plus_minus)
        plus_plus = np.concatenate(plus_plus)

    if not directed:
         # if the graph is not directed, there are no cycles with length 2 and every cycle was counted twice 
        if plus_plus.shape[1] >= 2:
            plus_plus[:, 1] = 0
            plus_minus[:, 1] = 0
        plus_plus /= 2
        plus_minus /= 2
    
    # insert zeros for cycle length 0 because these are included in the 'pyr' alg as well
    plus_plus = np.array([np.insert(arr, 0, 0) for arr in plus_plus])
    plus_minus = np.array([np.insert(arr, 0, 0) for arr in plus_minus])

    pos = (plus_plus + plus_minus) / 2
    neg = plus_plus - pos
    neg_k_bal = np.nan_to_num(neg / plus_plus, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pos_k_bal = np.nan_to_num(pos / plus_plus, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    neg_to_pos_ratio = np.nan_to_num(neg / pos, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pos_to_neg_ratio = np.nan_to_num(pos / neg, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    rel_signed_clust_coeff = np.nan_to_num(plus_minus / plus_plus, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    std_total = np.nanstd(plus_plus, axis=0)
    avg_total = np.nanmean(plus_plus, axis=0)
    std_total_perc = np.abs(np.nan_to_num(std_total / avg_total, nan=0.0))
    std_pos = np.nanstd(pos, axis=0)
    avg_pos = np.nanmean(pos, axis=0)
    std_pos_perc = np.abs(np.nan_to_num(std_pos / avg_pos, nan=0.0))
    std_neg = np.nanstd(neg, axis=0)
    avg_neg = np.nanmean(neg, axis=0)
    std_neg_perc = np.abs(np.nan_to_num(std_neg / avg_neg, nan=0.0))
    std_pos_k_bal = np.nanstd(pos_k_bal, axis=0)
    avg_pos_k_bal = np.nanmean(pos_k_bal, axis=0)
    std_pos_k_bal_perc = np.abs(np.nan_to_num(std_pos_k_bal / avg_pos_k_bal, nan=0.0))
    std_neg_k_bal = np.nanstd(neg_k_bal, axis=0)
    avg_neg_k_bal = np.nanmean(neg_k_bal, axis=0)
    std_neg_k_bal_perc = np.abs(np.nan_to_num(std_neg_k_bal / avg_neg_k_bal, nan=0.0))
    std_pos_to_neg_ratio = np.nanstd(pos_to_neg_ratio, axis=0)
    avg_pos_to_neg_ratio = np.nanmean(pos_to_neg_ratio, axis=0)
    std_pos_to_neg_ratio_perc = np.abs(np.nan_to_num(std_pos_to_neg_ratio / avg_pos_to_neg_ratio, nan=0.0))
    std_neg_to_pos_ratio = np.nanstd(neg_to_pos_ratio, axis=0)
    avg_neg_to_pos_ratio = np.nanmean(neg_to_pos_ratio, axis=0)
    std_neg_to_pos_ratio_perc = np.abs(np.nan_to_num(std_neg_to_pos_ratio / avg_neg_to_pos_ratio, nan=0.0))
    std_rel_signed_clust_coeff = np.nanstd(rel_signed_clust_coeff, axis=0)
    avg_rel_signed_clust_coeff = np.nanmean(rel_signed_clust_coeff, axis=0)
    std_rel_signed_clust_coeff_perc = np.abs(np.nan_to_num(std_rel_signed_clust_coeff / avg_rel_signed_clust_coeff, nan=0.0))

    zeros_total = avg_total == 0
    zeros_pos = avg_pos == 0
    zeros_neg = avg_neg == 0

    data = []
    for l in range(len(avg_total)):
            data.append({'l': l, 'avg_pos_k_bal': avg_pos_k_bal[l], 'avg_neg_k_bal': avg_neg_k_bal[l], 'avg_rel_signed_clust_coeff': avg_rel_signed_clust_coeff[l], 'avg_pos_to_neg_ratio': avg_pos_to_neg_ratio[l], 'avg_neg_to_pos_ratio': avg_neg_to_pos_ratio[l], 'avg_total': avg_total[l], 'avg_pos': avg_pos[l], 'avg_neg': avg_neg[l], 'zeros_total': zeros_total[l], 'zeros_pos': zeros_pos[l], 'zeros_neg': zeros_neg[l], 'std_pos_k_bal': std_pos_k_bal[l], 'std_pos_k_bal_perc': std_pos_k_bal_perc[l], 'std_neg_k_bal': std_neg_k_bal[l], 'std_neg_k_bal_perc': std_neg_k_bal_perc[l], 'std_rel_signed_clust_coeff': std_rel_signed_clust_coeff[l], 'std_rel_signed_clust_coeff_perc': std_rel_signed_clust_coeff_perc[l], 'std_pos_to_neg_ratio': std_pos_to_neg_ratio[l], 'std_pos_to_neg_ratio_perc': std_pos_to_neg_ratio_perc[l], 'std_neg_to_pos_ratio': std_neg_to_pos_ratio[l], 'std_neg_to_pos_ratio_perc': std_neg_to_pos_ratio_perc[l], 'std_total': std_total[l], 'std_total_perc': std_total_perc[l], 'std_pos': std_pos[l], 'std_pos_perc': std_pos_perc[l], 'std_neg': std_neg[l], 'std_neg_perc': std_neg_perc[l]})
    
    avg_of_all_pos_k_balance = np.nanmean(avg_pos_k_bal)
    avg_of_all_neg_k_balance = np.nanmean(avg_neg_k_bal)

    data[0]['avg_of_all_pos_k_balance'] = avg_of_all_pos_k_balance
    data[0]['avg_of_all_neg_k_balance'] = avg_of_all_neg_k_balance

    k = np.arange(1, plus_plus.shape[1] + 1)

    weight_names = ["degree_of_bal", "weighted_1_k_bal", "weighted_1_k_2_bal", "weighted_1_k_3_bal", "weighted_1_k_4_bal", "weighted_1_k_fac_bal"]
    weight_functions = [1, 1 / k, 1 / k**2, 1 / k**3, 1 / k**4, 1 / scipy.special.factorial(k)]

    for i, weights in enumerate(weight_functions):
        weighted_total_sum_cycles = np.nansum(plus_plus * weights, axis=1)
        weighted_pos_sum_cycles = np.nansum(pos * weights, axis=1)
        weighted_neg_sum_cycles = np.nansum(neg * weights, axis=1)

        pos_weighted_degree = np.nanmean(np.nan_to_num(weighted_pos_sum_cycles / weighted_total_sum_cycles, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan), axis=0)
        neg_weighted_degree = np.nanmean(np.nan_to_num(weighted_neg_sum_cycles / weighted_total_sum_cycles, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan), axis=0)
        
        measure_name_pos = f"pos_{weight_names[i]}"
        measure_name_neg = f"neg_{weight_names[i]}"

        data[0][measure_name_pos] = pos_weighted_degree
        data[0][measure_name_neg] = neg_weighted_degree
    
    return data

def is_connected(G):
    if directed:
        return nx.is_weakly_connected(G)
    else:
        return nx.is_connected(G)

def get_group_sbm(u):
    group = 0
    upper = kind_params['com_sizes'][group]
    while u >= upper:
        group += 1
        upper += kind_params['com_sizes'][group]
    return group

if __name__ == "__main__":

    script_start_time = time.time()
    tracemalloc.start()
    exp_date = datetime.datetime.now()

    # Suppress invalid value warnings (e.g., divide by zero or NaN)
    np.seterr(divide='ignore', invalid='ignore', )
    warnings.filterwarnings("ignore", category=np.ComplexWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")


    SEEDS = [130250193, 4259459283, 1535995910, 1779285243, 347749838, 2843683941, 1121168978, 1281646427, 2640136520, 1846011679, 2548891259, 889869862, 816974034, 898432604, 3953502902, 480242123, 1422106884, 2844494128, 2766920156, 3963647525, 2800650013, 3026969699, 617436447, 3125820586, 968979614, 3011594777, 2949848623, 2343270211, 1911319159, 678032221, 2644994550, 2694585517, 876264514, 1420930522, 1191847850, 3452672408, 694543404, 429751076, 1464333811, 2794718515, 3303745309, 2176095271, 1235875709, 2083610798, 2992731158, 1458240102, 3463342733, 2894203811, 1901449950, 807625046]

    snakemake = fix_smk()

    run = int(snakemake.wildcards['run'])
    kind = snakemake.params['kind']
    alg = snakemake.params['alg']
    n_samples = int(snakemake.wildcards.get('n_samples', 1))
    n_nodes = int(snakemake.wildcards.get('n_nodes', -1))
    n_edges = int(snakemake.wildcards.get('n_edges', -1))
    neg_edge_prob = float(snakemake.wildcards.get('neg_edge_prob', -1))
    neg_edge_dist_exact = snakemake.wildcards.get('neg_edge_dist_exact', 'false').lower() == 'true'
    directed = snakemake.wildcards.get('directed', 'false').lower() == 'true'

    seed = SEEDS[run]
    random.seed(seed)
    np.random.seed(seed)
    rnd = np.random.default_rng(seed)

    kind_params = {}

    if kind == 'er':
        kind_params['prob_p'] = float(snakemake.wildcards['prob_p'])
        G = nx.gnp_random_graph(n_nodes, kind_params['prob_p'], directed=directed, seed=rnd)
        while not is_connected(G):
            G  = nx.gnp_random_graph(n_nodes, kind_params['prob_p'], directed=directed, seed=rnd)

    elif kind == 'complete':
        G = nx.complete_graph(n_nodes)

    elif kind == 'random':
        G = nx.gnm_random_graph(n_nodes, n_edges, directed=directed, seed=rnd)
        while not is_connected(G):
            G = nx.gnm_random_graph(n_nodes, n_edges, directed=directed, seed=rnd)
    
    elif kind == 'sbm':
        kind_params['com_sizes'] = snakemake.params['com_sizes']
        kind_params['edge_probs'] = snakemake.params['edge_probs']
        kind_params['neg_edge_probs'] = snakemake.params['neg_edge_probs']

        G = nx.stochastic_block_model(kind_params['com_sizes'], kind_params['edge_probs'], directed=directed, seed=rnd)
        while not is_connected(G):
            G = nx.stochastic_block_model(kind_params['com_sizes'], kind_params['edge_probs'], directed=directed, seed=rnd)

        if not directed:
            if not np.array_equal(kind_params['neg_edge_probs'], np.transpose(kind_params['neg_edge_probs'])):
                raise ValueError("If the graph is undirected, neg_edge_probs must be symmetric")


        n_pos_edges = n_neg_edges = 0

        if neg_edge_dist_exact:
            n_groups = len(kind_params['com_sizes'])
            edges_sorted_by_groups = [[[] for _ in range(n_groups)] for _ in range(n_groups)]

            for u, v in G.edges():
                group_u = get_group_sbm(u)
                group_v = get_group_sbm(v)
                edges_sorted_by_groups[group_u][group_v].append((u, v))
            
            for group_u in range(n_groups):
                for group_v in range(n_groups):
                    rnd.shuffle(edges_sorted_by_groups[group_u][group_v])
                    prob_r = kind_params['neg_edge_probs'][group_u][group_v]
                    n_edges_u_v = len(edges_sorted_by_groups[group_u][group_v])
                    n_neg_edges_u_v = round(n_edges_u_v * prob_r)
                    n_pos_edges_u_v = n_edges_u_v - n_neg_edges_u_v
                    for a, b in edges_sorted_by_groups[group_u][group_v][0:n_neg_edges_u_v]:
                        G[a][b]['weight'] = -1
                    for a, b in edges_sorted_by_groups[group_u][group_v][n_neg_edges_u_v:]:
                        G[a][b]['weight'] = 1
                    n_pos_edges += n_pos_edges_u_v
                    n_neg_edges += n_neg_edges_u_v
        else: 
            for u, v in G.edges():
                group_u = get_group_sbm(u)
                group_v = get_group_sbm(v)
                prob_r = kind_params['neg_edge_probs'][group_u][group_v]
                weight = rnd.choice([-1, 1], p=[prob_r, 1-prob_r])
                G[u][v]['weight'] = weight
                if weight > 0:
                    n_pos_edges += 1
                else: 
                    n_neg_edges += 1

    else:
        raise ValueError("Unknown kind: ", kind)
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if kind != 'sbm' and kind != 'dataset':
        n_pos_edges = n_neg_edges = 0

        if neg_edge_dist_exact:
            n_neg_edges = round(n_edges * neg_edge_prob)
            n_pos_edges = n_edges - n_neg_edges
            shuffled_edges = list(G.edges())
            rnd.shuffle(shuffled_edges)
            for u, v in shuffled_edges[0:n_neg_edges]:
                G[u][v]['weight'] = -1
            for u, v in shuffled_edges[n_neg_edges:]:
                G[u][v]['weight'] = 1
        else: 
            for u, v in G.edges():
                weight = rnd.choice([-1, 1], p=[neg_edge_prob, 1-neg_edge_prob])
                G[u][v]['weight'] = weight
                if weight > 0:
                    n_pos_edges += 1
                else: 
                    n_neg_edges += 1

    alg_params = {}

    if alg == 'pyr':
        if directed: 
            raise ValueError("The 'pyr' algorithm does not work on directed graphs.")
        if n_nodes <= 3:
            raise ValueError("The pyr algorithm doesn't work on less than 4 nodes")
        n_samples = int(snakemake.wildcards['n_samples'])
        if snakemake.wildcards['pyr_spec_edge_prob'].lower() == "none":
            alg_params['pyr_spec_edge_prob'] = None
        else:
            alg_params['pyr_spec_edge_prob'] = float(snakemake.wildcards['pyr_spec_edge_prob'])
        
        init_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        alg_start_time = time.time()
        results = pyr.estimate_balance(G, samples=n_samples, p=alg_params['pyr_spec_edge_prob'], seed=rnd)
        alg_end_time = time.time()
        _, alg_peak_mem = tracemalloc.get_traced_memory()
        data = save_pyr_results(*results)
    elif alg == 'cx':
        alg_params['max_length'] = int(snakemake.wildcards['max_length'])
        alg_params['exact'] = snakemake.wildcards['exact'].lower() == 'true'
        alg_params['parallel'] = snakemake.wildcards['parallel'].lower() == 'true'

        if alg_params['exact'] and n_samples != 1:
            raise ValueError("An exact solution does not use more than 1 sample.")
        
        full_matrix = cx.to_adj_matrix(G)
        alg_params['full_matrix_mem'] = full_matrix.nbytes
        clean_matrix = cx.clean_matrix(full_matrix)
        alg_params['clean_matrix_mem'] = clean_matrix.nbytes
        
        init_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        alg_start_time = time.time()
        results = cx.balance_ratio(clean_matrix, length=min(alg_params['max_length'], clean_matrix.shape[0]), exact=alg_params['exact'], n_samples=n_samples, parallel=alg_params['parallel'])
        alg_end_time = time.time()
        _, alg_peak_mem = tracemalloc.get_traced_memory()
        data = save_cx_results(*results)
    else:
        raise ValueError("Unknown algorithm: ", alg)

    tracemalloc.stop()
    script_end_time = time.time()

    params = {}
    params['run'] = run
    params['kind'] = kind
    params['directed'] = directed
    params.update(kind_params)
    params['neg_edge_dist_exact'] = neg_edge_dist_exact
    if kind != 'sbm': params['neg_edge_prob'] = neg_edge_prob
    params['n_nodes'] = n_nodes
    params['n_edges'] = n_edges
    params['n_pos_edges'] = n_pos_edges
    params['n_neg_edges'] = n_neg_edges
    params['alg'] = alg
    params['n_samples'] = n_samples
    params.update(alg_params)
    params['alg_run_time'] = alg_end_time - alg_start_time 
    params['alg_peak_mem'] = alg_peak_mem
    params['alg_only_mem'] = alg_peak_mem - init_mem
    params['init_mem'] = init_mem
    params['seed'] = seed
    params['script_run_time'] = script_end_time - script_start_time
    params['exp_date'] = exp_date
    data[0].update(params)

    df_data = pd.DataFrame(data)
    df_data.to_csv(snakemake.output[0])