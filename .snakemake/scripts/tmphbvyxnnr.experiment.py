######## snakemake preamble start (automatically inserted, do not edit) ########
import sys;sys.path.extend(['/Users/fredericbusch/miniforge3/envs/py-raccoon/lib/python3.12/site-packages', '/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow', '/Users/fredericbusch/miniforge3/envs/py-raccoon/bin', '/Users/fredericbusch/miniforge3/envs/py-raccoon/lib/python3.12', '/Users/fredericbusch/miniforge3/envs/py-raccoon/lib/python3.12/lib-dynload', '/Users/fredericbusch/miniforge3/envs/py-raccoon/lib/python3.12/site-packages', '/Users/fredericbusch/Library/Caches/snakemake/snakemake/source-cache/runtime-cache/tmp_9v7srld/file/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts', '/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts']);import pickle;from snakemake import script;script.snakemake = pickle.loads(b'\x80\x04\x95`\x05\x00\x00\x00\x00\x00\x00\x8c\x10snakemake.script\x94\x8c\tSnakemake\x94\x93\x94)\x81\x94}\x94(\x8c\x05input\x94\x8c\x0csnakemake.io\x94\x8c\nInputFiles\x94\x93\x94)\x81\x94}\x94(\x8c\x06_names\x94}\x94\x8c\x12_allowed_overrides\x94]\x94(\x8c\x05index\x94\x8c\x04sort\x94eh\x0fh\x06\x8c\x0eAttributeGuard\x94\x93\x94)\x81\x94}\x94\x8c\x04name\x94h\x0fsbh\x10h\x12)\x81\x94}\x94h\x15h\x10sbub\x8c\x06output\x94h\x06\x8c\x0bOutputFiles\x94\x93\x94)\x81\x94\x8cUresults/synthetic/complete/n=25 r=0.5 d=False/cx/e=True p=False l=10 s=1/run/cx_0.csv\x94a}\x94(h\x0b}\x94h\r]\x94(h\x0fh\x10eh\x0fh\x12)\x81\x94}\x94h\x15h\x0fsbh\x10h\x12)\x81\x94}\x94h\x15h\x10sbub\x8c\x06params\x94h\x06\x8c\x06Params\x94\x93\x94)\x81\x94(\x8c\x08complete\x94\x8c\x02cx\x94e}\x94(h\x0b}\x94(\x8c\x04kind\x94K\x00N\x86\x94\x8c\x03alg\x94K\x01N\x86\x94uh\r]\x94(h\x0fh\x10eh\x0fh\x12)\x81\x94}\x94h\x15h\x0fsbh\x10h\x12)\x81\x94}\x94h\x15h\x10sbh,h(h.h)ub\x8c\twildcards\x94h\x06\x8c\tWildcards\x94\x93\x94)\x81\x94(\x8c\x0225\x94\x8c\x030.5\x94\x8c\x05False\x94\x8c\x04True\x94\x8c\x05False\x94\x8c\x0210\x94\x8c\x011\x94\x8c\x010\x94e}\x94(h\x0b}\x94(\x8c\x07n_nodes\x94K\x00N\x86\x94\x8c\rneg_edge_prob\x94K\x01N\x86\x94\x8c\x08directed\x94K\x02N\x86\x94\x8c\x05exact\x94K\x03N\x86\x94\x8c\x08parallel\x94K\x04N\x86\x94\x8c\nmax_length\x94K\x05N\x86\x94\x8c\tn_samples\x94K\x06N\x86\x94\x8c\x03run\x94K\x07N\x86\x94uh\r]\x94(h\x0fh\x10eh\x0fh\x12)\x81\x94}\x94h\x15h\x0fsbh\x10h\x12)\x81\x94}\x94h\x15h\x10sbhCh9hEh:hGh;\x8c\x05exact\x94h<\x8c\x08parallel\x94h=\x8c\nmax_length\x94h>hOh?\x8c\x03run\x94h@ub\x8c\x07threads\x94K\x01\x8c\tresources\x94h\x06\x8c\tResources\x94\x93\x94)\x81\x94(K\x01K\x01\x8c0/var/folders/dj/14c985vd1ng7gf0vvw_ytrpw0000gn/T\x94e}\x94(h\x0b}\x94(\x8c\x06_cores\x94K\x00N\x86\x94\x8c\x06_nodes\x94K\x01N\x86\x94\x8c\x06tmpdir\x94K\x02N\x86\x94uh\r]\x94(h\x0fh\x10eh\x0fh\x12)\x81\x94}\x94h\x15h\x0fsbh\x10h\x12)\x81\x94}\x94h\x15h\x10sbhdK\x01hfK\x01hhhaub\x8c\x03log\x94h\x06\x8c\x03Log\x94\x93\x94)\x81\x94}\x94(h\x0b}\x94h\r]\x94(h\x0fh\x10eh\x0fh\x12)\x81\x94}\x94h\x15h\x0fsbh\x10h\x12)\x81\x94}\x94h\x15h\x10sbub\x8c\x06config\x94}\x94\x8c\x08sbm-exps\x94}\x94(K\x00}\x94(\x8c\x05sizes\x94]\x94(K\nK\ne\x8c\x01p\x94]\x94(]\x94(G?\xe0\x00\x00\x00\x00\x00\x00G?\xc9\x99\x99\x99\x99\x99\x9ae]\x94(G?\xc9\x99\x99\x99\x99\x99\x9aG?\xe0\x00\x00\x00\x00\x00\x00eeuK\x01}\x94(\x8c\x05sizes\x94]\x94(K\nK\x14eh\x81]\x94(]\x94(G?\xd9\x99\x99\x99\x99\x99\x9aG?\xd3333333e]\x94(G?\xd3333333G?\xd9\x99\x99\x99\x99\x99\x9aeeuus\x8c\x04rule\x94\x8c\x0bcomplete_cx\x94\x8c\x0fbench_iteration\x94N\x8c\tscriptdir\x94\x8cE/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts\x94ub.');del script;from snakemake.logging import logger;from snakemake.script import snakemake; logger.printshellcmds = False;__real_file__ = __file__; __file__ = '/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/experiment.py';
######## snakemake preamble end #########
import networkx as nx 
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import time
import datetime
import tracemalloc
import warnings

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
    
    pos_degree_of_bal = np.nan_to_num(pos_est / total_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    neg_degree_of_bal = np.nan_to_num(neg_est / total_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    rel_signed_clust_coeff = np.nan_to_num((pos_est - neg_est) / total_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pos_to_neg_ratio = np.nan_to_num(pos_est / neg_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    neg_to_pos_ratio = np.nan_to_num(neg_est / pos_est, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)

    total_zeros = total_est == 0
    pos_zeros = pos_est == 0
    neg_zeros = neg_est == 0

    data = []
    for l in range(len(total_est)):
            data.append({'l': l, 'pos_degree_of_bal': pos_degree_of_bal[l], 'neg_degree_of_bal': neg_degree_of_bal[l], 'rel_signed_clust_coeff': rel_signed_clust_coeff[l], 'pos_to_neg_ratio': pos_to_neg_ratio[l], 'neg_to_pos_ratio': neg_to_pos_ratio[l], 'total_est': total_est[l], 'pos_est': pos_est[l], 'neg_est': neg_est[l], 'total_zeros': total_zeros[l], 'pos_zeros': pos_zeros[l], 'neg_zeros': neg_zeros[l], 'total_occurred': total_occurred[l], 'pos_occurred': pos_occurred[l], 'neg_occurred': neg_occurred[l]})
    
    total_sum_cycles = np.nansum(total_est)
    pos_sum_cycles = np.nansum(pos_est)
    neg_sum_cycles = np.nansum(neg_est)

    pos_degree_of_bal_graph = pos_sum_cycles / total_sum_cycles
    neg_degree_of_bal_graph = neg_sum_cycles / total_sum_cycles

    data[0]['pos_degree_of_bal_graph'] = pos_degree_of_bal_graph
    data[0]['neg_degree_of_bal_graph'] = neg_degree_of_bal_graph

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
         if len(plus_plus) >= 2:
            plus_plus[:, 1] = plus_minus[:, 1] = 0
         plus_plus /= 2
         plus_minus /= 2
    
    # insert zeros for cycle length 0 because these are included in the 'pyr' alg as well
    plus_plus = np.array([np.insert(arr, 0, 0) for arr in plus_plus])
    plus_minus = np.array([np.insert(arr, 0, 0) for arr in plus_minus])

    pos = (plus_plus + plus_minus) / 2
    neg = plus_plus - pos
    neg_degree_of_bal = np.nan_to_num(neg / plus_plus, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pos_degree_of_bal = np.nan_to_num(pos / plus_plus, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    neg_to_pos_ratio = np.nan_to_num(neg / pos, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pos_to_neg_ratio = np.nan_to_num(pos / neg, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    rel_signed_clust_coeff = np.nan_to_num(plus_minus / plus_plus, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    std_total = np.nanstd(plus_plus, axis=0)
    avg_total = np.nanmean(plus_plus, axis=0)
    std_pos = np.nanstd(pos, axis=0)
    avg_pos = np.nanmean(pos, axis=0)
    std_neg = np.nanstd(neg, axis=0)
    avg_neg = np.nanmean(neg, axis=0)
    std_pos_degree_of_bal = np.nanstd(pos_degree_of_bal, axis=0)
    avg_pos_degree_of_bal = np.nanmean(pos_degree_of_bal, axis=0)
    std_neg_degree_of_bal = np.nanstd(neg_degree_of_bal, axis=0)
    avg_neg_degree_of_bal = np.nanmean(neg_degree_of_bal, axis=0)
    std_pos_to_neg_ratio = np.nanstd(pos_to_neg_ratio, axis=0)
    avg_pos_to_neg_ratio = np.nanmean(pos_to_neg_ratio, axis=0)
    std_neg_to_pos_ratio = np.nanstd(neg_to_pos_ratio, axis=0)
    avg_neg_to_pos_ratio = np.nanmean(neg_to_pos_ratio, axis=0)
    std_rel_signed_clust_coeff = np.nanstd(rel_signed_clust_coeff, axis=0)
    avg_rel_signed_clust_coeff = np.nanmean(rel_signed_clust_coeff, axis=0)

    zeros_total = avg_total == 0
    zeros_pos = avg_pos == 0
    zeros_neg = avg_neg == 0

    data = []
    for l in range(len(avg_total)):
            data.append({'l': l, 'avg_pos_degree_of_bal': avg_pos_degree_of_bal[l], 'avg_neg_degree_of_bal': avg_neg_degree_of_bal[l], 'avg_rel_signed_clust_coeff': avg_rel_signed_clust_coeff[l], 'avg_pos_to_neg_ratio': avg_pos_to_neg_ratio[l], 'avg_neg_to_pos_ratio': avg_neg_to_pos_ratio[l], 'avg_total': avg_total[l], 'avg_pos': avg_pos[l], 'avg_neg': avg_neg[l], 'zeros_total': zeros_total[l], 'zeros_pos': zeros_pos[l], 'zeros_neg': zeros_neg[l], 'std_pos_degree_of_bal': std_pos_degree_of_bal[l], 'std_neg_degree_of_bal': std_neg_degree_of_bal[l], 'std_rel_signed_clust_coeff': std_rel_signed_clust_coeff[l], 'std_pos_to_neg_ratio': std_pos_to_neg_ratio[l], 'std_neg_to_pos_ratio': std_neg_to_pos_ratio[l], 'std_total': std_total[l], 'std_pos': std_pos[l], 'std_neg': std_neg[l]})
    
    total_sum_cycles = np.nansum(plus_plus, axis=1)
    pos_sum_cycles = np.nansum(pos, axis=1)
    neg_sum_cycles = np.nansum(neg, axis=1)
    
    avg_pos_degree_of_bal_graph = np.nanmean(np.nan_to_num(pos_sum_cycles / total_sum_cycles, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan), axis=0)
    avg_neg_degree_of_bal_graph = np.nanmean(np.nan_to_num(neg_sum_cycles / total_sum_cycles, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan), axis=0)

    data[0]['avg_pos_degree_of_bal_graph'] = avg_pos_degree_of_bal_graph
    data[0]['avg_neg_degree_of_bal_graph'] = avg_neg_degree_of_bal_graph
    
    return data

def is_connected(G):
    if directed:
        return nx.is_weakly_connected(G)
    else:
        return nx.is_connected(G)

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
    neg_edge_prob = float(snakemake.wildcards['neg_edge_prob'])
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
        G = nx.stochastic_block_model(kind_params['com_sizes'], kind_params['edge_probs'], directed=directed, seed=rnd)
        while not is_connected(G):
            G = nx.stochastic_block_model(kind_params['com_sizes'], kind_params['edge_probs'], directed=directed, seed=rnd)
    
    else:
        raise ValueError("Unknown kind: ", kind)
    
    n_pos_edges = n_neg_edges = 0
    for u, v in G.edges():
        weight = rnd.choice([-1, 1], p=[neg_edge_prob, 1-neg_edge_prob])
        G[u][v]['weight'] = weight
        if weight > 0:
            n_pos_edges += 1
        else: 
            n_neg_edges += 1
                 
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    alg_params = {}

    if alg == 'pyr':
        if directed: 
            raise ValueError("The 'pyr' algorithm does not work on directed graphs.")
        
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
    params['neg_edge_prob'] = neg_edge_prob
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