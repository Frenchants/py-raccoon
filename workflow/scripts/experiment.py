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
import os

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

# function which accounts for the postprocessing part after the execution of the pyr (FBE) algorithm
def save_pyr_results(total_est, pos_est, neg_est, total_occurred, pos_occurred, neg_occurred):
    
    # compute measures
    pos_k_bal = np.nan_to_num(pos_est / total_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    neg_k_bal = np.nan_to_num(neg_est / total_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    rel_signed_clust_coeff = np.nan_to_num((pos_est - neg_est) / total_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pos_to_neg_ratio = np.nan_to_num(pos_est / neg_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    neg_to_pos_ratio = np.nan_to_num(neg_est / pos_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)

    total_zeros = total_est == 0
    pos_zeros = pos_est == 0
    neg_zeros = neg_est == 0

    # save estimated measures
    data = []
    for l in range(len(total_est)):
            data.append({'l': l, 'pos_k_bal': pos_k_bal[l], 'neg_k_bal': neg_k_bal[l], 'rel_signed_clust_coeff': rel_signed_clust_coeff[l], 'pos_to_neg_ratio': pos_to_neg_ratio[l], 'neg_to_pos_ratio': neg_to_pos_ratio[l], 'total_est': total_est[l], 'pos_est': pos_est[l], 'neg_est': neg_est[l], 'total_zeros': total_zeros[l], 'pos_zeros': pos_zeros[l], 'neg_zeros': neg_zeros[l], 'total_occurred': total_occurred[l], 'pos_occurred': pos_occurred[l], 'neg_occurred': neg_occurred[l]})
    

    # calculate the avgs of all obtained k_balance values
    avg_pos_k_balance = np.nanmean(pos_k_bal)
    avg_neg_k_balance = np.nanmean(neg_k_bal)

    data[0]['avg_pos_k_balance'] = avg_pos_k_balance
    data[0]['avg_neg_k_balance'] = avg_neg_k_balance


    # compute other cycle-based measures
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

# postprocessing of the CX algorithm
def save_cx_results(plus_minus, plus_plus):
   
   # the arrays returned by the algorithm are estimates for the total number of cycles, that is the sum of the positive and negative cylces (plus_plus), and estimates for the difference between the numbers of the positive and negative cycles (minus_minus)

    plus_plus = np.asarray(plus_plus, dtype=np.float64)
    plus_minus = np.asarray(plus_minus, dtype=np.float64)
    
    # this was originally part of the cycleindex code
    if plus_minus.ndim == 1:
        plus_minus = np.expand_dims(plus_minus, 0)
        plus_plus = np.expand_dims(plus_plus, 0)
    elif plus_minus.ndim == 3:
        plus_minus = np.concatenate(plus_minus)
        plus_plus = np.concatenate(plus_plus)


    # the CX algorithms always assumes that the input graph is directed, that is it treats undirected networks as directed networks as well. Therefore, the algorithm counts cycles twice, and additionally it may find cycles of length 2. Here, this undone by dividing the cycle counts by two and by setting the cycle counts for length 2 to 0. 
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

    # from the plus_plus and plus_minus arrays returned by the CX algorithm, the pos and neg cycle counts are derived
    pos = (plus_plus + plus_minus) / 2
    neg = plus_plus - pos

    # the values of the CX algorithm are sometimes distorted and adopt small values near 0. Here, this is undone by treating such values as 0. 
    threshold = 1e-5
    plus_plus[np.abs(plus_plus) < threshold] = 0
    plus_minus[np.abs(plus_minus) < threshold] = 0
    pos[np.abs(pos) < threshold] = 0
    neg[np.abs(neg) < threshold] = 0


    # compute measures for every sample
    neg_k_bal = np.nan_to_num(neg / plus_plus, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pos_k_bal = np.nan_to_num(pos / plus_plus, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    neg_to_pos_ratio = np.nan_to_num(neg / pos, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pos_to_neg_ratio = np.nan_to_num(pos / neg, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    rel_signed_clust_coeff = np.nan_to_num(plus_minus / plus_plus, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    # compute the averages of the measures across all samples 
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

    k_pos_bal = np.nan_to_num(avg_pos / avg_total, nan=0.0)
    k_neg_bal = np.nan_to_num(avg_neg / avg_total, nan=0.0)

    zeros_total = avg_total == 0
    zeros_pos = avg_pos == 0
    zeros_neg = avg_neg == 0

    # save data
    data = []
    for l in range(len(avg_total)):
            data.append({'l': l, 'k_pos_bal': k_pos_bal[l], 'k_neg_bal': k_neg_bal[l], 'avg_pos_k_bal': avg_pos_k_bal[l], 'avg_neg_k_bal': avg_neg_k_bal[l], 'avg_rel_signed_clust_coeff': avg_rel_signed_clust_coeff[l], 'avg_pos_to_neg_ratio': avg_pos_to_neg_ratio[l], 'avg_neg_to_pos_ratio': avg_neg_to_pos_ratio[l], 'avg_total': avg_total[l], 'avg_pos': avg_pos[l], 'avg_neg': avg_neg[l], 'zeros_total': zeros_total[l], 'zeros_pos': zeros_pos[l], 'zeros_neg': zeros_neg[l], 'std_pos_k_bal': std_pos_k_bal[l], 'std_pos_k_bal_perc': std_pos_k_bal_perc[l], 'std_neg_k_bal': std_neg_k_bal[l], 'std_neg_k_bal_perc': std_neg_k_bal_perc[l], 'std_rel_signed_clust_coeff': std_rel_signed_clust_coeff[l], 'std_rel_signed_clust_coeff_perc': std_rel_signed_clust_coeff_perc[l], 'std_pos_to_neg_ratio': std_pos_to_neg_ratio[l], 'std_pos_to_neg_ratio_perc': std_pos_to_neg_ratio_perc[l], 'std_neg_to_pos_ratio': std_neg_to_pos_ratio[l], 'std_neg_to_pos_ratio_perc': std_neg_to_pos_ratio_perc[l], 'std_total': std_total[l], 'std_total_perc': std_total_perc[l], 'std_pos': std_pos[l], 'std_pos_perc': std_pos_perc[l], 'std_neg': std_neg[l], 'std_neg_perc': std_neg_perc[l]})
    

    # obtain average of all the k_balance values
    avg_of_all_pos_k_balance = np.nanmean(avg_pos_k_bal)
    avg_of_all_neg_k_balance = np.nanmean(avg_neg_k_bal)

    data[0]['avg_of_all_pos_k_balance'] = avg_of_all_pos_k_balance
    data[0]['avg_of_all_neg_k_balance'] = avg_of_all_neg_k_balance

    k = np.arange(1, plus_plus.shape[1] + 1)


    # compute weighted degree of balance 

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

# check if a graph is either weakly connected if the graph is directed or if its connected if the graph is undirecte
def is_connected(G):
    if directed:
        return nx.is_weakly_connected(G)
    else:
        return nx.is_connected(G)

# helper function, not used in the experiments. 
def get_group_sbm(u):
    group = 0
    upper = kind_params['com_sizes'][group]
    while u >= upper:
        group += 1
        upper += kind_params['com_sizes'][group]
    return group


# main method of the script 
if __name__ == "__main__":

    # start measuring script runtime and memory consumption
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

    # obtain snakemake wildcards 
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


    # start preprocessing based on the experiment type
    if kind == 'dataset':
        kind_params['dataset'] = snakemake.wildcards['dataset']
        # determines whether the null_model of the graph should be calculated 
        kind_params['null_model'] = snakemake.wildcards.get('null_model', 'false').lower() == 'true'

        if directed: 
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        base_dir = "workflow/scripts/datasets/"
        n_pos_edges = n_neg_edges = 0
        if kind_params['dataset'] == 'epinions':
            file_name = "epinions.txt"
        elif kind_params['dataset'] == 'gahuku':
            file_name = "gahuku.txt"
        elif kind_params['dataset'] == 'slashdot':
            file_name = "slashdot.txt"
        elif kind_params['dataset'] == 'wikielections':
            file_name = "wikielections.txt"
        elif kind_params['dataset'] == 'cow':
            cow_year = snakemake.wildcards['year']
            file_name = f"correlates_of_war/combined_graphs/{cow_year}.txt"
        else:
            raise ValueError("There is no dataset of name " , kind_params['dataset'])
        
        print("Loading dataset...")
        with open(os.path.join(base_dir, file_name), "r") as file:
                for line in file:
                    x, y, z = line.split() 
                    G.add_edge(int(x), int(y), weight=int(z))
                    if int(z) > 0:
                        n_pos_edges += 1
                    else:
                        n_neg_edges += 1
        
        neg_edge_prob = n_neg_edges / (n_pos_edges + n_neg_edges)
        mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        print("Finished loading dataset.")
       
        # calculate null_model. Each edge in G is assigned a negative weight with probability r (neg_edge_prob)
        if kind_params['null_model']:
            print("Preparing null model...")
            n_pos_edges = n_neg_edges = 0
            for u, v in G.edges():
                weight = rnd.choice([-1, 1], p=[neg_edge_prob, 1-neg_edge_prob])
                G[u][v]['weight'] = weight
                if weight > 0:
                    n_pos_edges += 1
                else: 
                    n_neg_edges += 1
            print("Finished initialising null model.")

    elif kind == 'er':
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


    # assign random weights based on probability neg_edge_prob to graph
    if kind != 'sbm' and kind != 'dataset':
        n_pos_edges = n_neg_edges = 0

        # if neg_edge_dist_exact is set to true, an exact proportion of negative edges in the graph is tried to be achieved (the desired proportion then is indicated by neg_edge_prob) 
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

    # run algs
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
        
        if nx.is_connected(G):
            # start measuring algorithm runtime and memory usage
            init_mem, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            alg_start_time = time.time()
            results = pyr.estimate_balance(G, samples=n_samples, p=alg_params['pyr_spec_edge_prob'], seed=rnd)
            alg_end_time = time.time()
            _, alg_peak_mem = tracemalloc.get_traced_memory()
            data = save_pyr_results(*results)
            alg_total_time = alg_end_time - alg_start_time
        else:
            # if the graph is not connected, the algorithm is run on every connected component of the input graph. The values obtained in each run are added
            init_mem, _ = tracemalloc.get_traced_memory()
            alg_peak_mem = 0
            alg_total_time = 0
            total_est_sum = np.zeros(len(G.nodes) + 1, np.float64)
            pos_est_sum = np.zeros(len(G.nodes) + 1, np.float64)
            neg_est_sum = np.zeros(len(G.nodes) + 1, np.float64) 
            total_occurred_sum = np.zeros(len(G.nodes) + 1, np.float64)
            pos_occurred_sum = np.zeros(len(G.nodes) + 1, np.float64)
            neg_occurred_sum = np.zeros(len(G.nodes) + 1, np.float64)

            components = [G.subgraph(nodes).copy() for nodes in nx.connected_components(G)]
            for H in components:
                nodes_of_H = len(H.nodes)
                if nodes_of_H < 3:
                    continue
                
                # the nodes need to be renumbered so that they start counting from 0. Otherwise, the algorithm fails
                mapping = {old_label: new_label for new_label, old_label in enumerate(H.nodes())}
                H_relabel = nx.relabel_nodes(H, mapping)
                if nodes_of_H == 3: 
                    H_relabel.add_edge(0, 3, weight=1)
                tracemalloc.reset_peak()
                alg_start_time = time.time()
                total_est, pos_est, neg_est, total_occurred, pos_occurred, neg_occurred = pyr.estimate_balance(H_relabel, samples=n_samples, p=alg_params['pyr_spec_edge_prob'], seed=rnd)
                alg_end_time = time.time()
                _, connected_comp_peak_mem = tracemalloc.get_traced_memory()
                if connected_comp_peak_mem > alg_peak_mem:
                    alg_peak_mem = connected_comp_peak_mem
                alg_total_time += alg_end_time - alg_start_time 
                
                total_est_sum += np.pad(total_est, (0, max(0, len(G.nodes) - len(H_relabel.nodes))), constant_values=0)
                pos_est_sum += np.pad(pos_est, (0, max(0, len(G.nodes) - len(H_relabel.nodes))), constant_values=0)
                neg_est_sum += np.pad(neg_est, (0, max(0, len(G.nodes) - len(H_relabel.nodes))), constant_values=0)
                total_occurred_sum += np.pad(total_occurred, (0, max(0, len(G.nodes) - len(H_relabel.nodes))), constant_values=0)
                pos_occurred_sum += np.pad(pos_occurred, (0, max(0, len(G.nodes) - len(H_relabel.nodes))), constant_values=0)
                neg_occurred_sum += np.pad(neg_occurred, (0, max(0, len(G.nodes) - len(H_relabel.nodes))), constant_values=0)
            
            data = save_pyr_results(total_est_sum, pos_est_sum, neg_est_sum, total_occurred_sum, pos_occurred_sum, neg_occurred_sum)

    elif alg == 'cx':
        alg_params['max_length'] = int(snakemake.wildcards['max_length'])
        alg_params['exact'] = snakemake.wildcards['exact'].lower() == 'true'
        alg_params['parallel'] = snakemake.wildcards['parallel'].lower() == 'true'

        if alg_params['exact'] and n_samples != 1:
            raise ValueError("An exact solution does not use more than 1 sample.")
        
        # this is the computation intensive preprocessing part of the algorithm. The implementation is not efficient, the adj matrices are full n x n matrices. Depending on the dataset, this step might take hours
        full_matrix = cx.to_adj_matrix(G)
        alg_params['full_matrix_mem'] = full_matrix.nbytes
        # the clean_matrix function is described in the cycleindex package
        clean_matrix = cx.clean_matrix(full_matrix)
        alg_params['clean_matrix_mem'] = clean_matrix.nbytes

        # it may happen that the algorithm tries to sample induced subgraphs of a certain length. However, if the graph does not have any induced subgraphs of that length, the algorithm runs indefinitely. This is avoided here
        if (kind == "dataset" and kind_params['dataset'] == 'cow') or (kind=='er'):
            largest_cc_size = max(len(component) for component in nx.connected_components(G))
            if largest_cc_size < alg_params['max_length']:
                alg_params['max_length'] = largest_cc_size
        
        # measure time and memory
        init_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        alg_start_time = time.time()
        results = cx.balance_ratio(clean_matrix, length=min(alg_params['max_length'], clean_matrix.shape[0]), exact=alg_params['exact'], n_samples=n_samples, parallel=alg_params['parallel'])
        alg_end_time = time.time()
        _, alg_peak_mem = tracemalloc.get_traced_memory()
        alg_total_time = alg_end_time - alg_start_time
        data = save_cx_results(*results)
    else:
        raise ValueError("Unknown algorithm: ", alg)

    tracemalloc.stop()
    script_end_time = time.time()

    # save all the parameters
    params = {}
    params['run'] = run
    params['kind'] = kind
    if kind == 'dataset': params['null_model'] = kind_params['null_model']
    if kind == 'dataset' and kind_params['dataset'] == 'cow': params['cow_year'] = cow_year
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
    params['alg_run_time'] = alg_total_time
    params['alg_peak_mem'] = alg_peak_mem
    # the alg_only_mem is the peak memory observed during the execution of the algorithm, excluding any memory that is consumed in the preprocessing part (captured by the var init_mem here)
    params['alg_only_mem'] = alg_peak_mem - init_mem
    params['init_mem'] = init_mem
    params['seed'] = seed
    params['script_run_time'] = script_end_time - script_start_time
    params['exp_date'] = exp_date
    data[0].update(params)

    df_data = pd.DataFrame(data)
    df_data.to_csv(snakemake.output[0])


"""
    edge_colors = ['blue' if G[u][v]['weight'] == 1 else 'red' for u, v in G.edges]
    pos = nx.spring_layout(G, k=1.0)  # Layout for visualization
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, edge_color=edge_colors, width=2)
    plt.show()
"""