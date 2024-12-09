import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


for p in [0.25, 0.5]:

    cx_header = "/Users/fredericbusch/Desktop/Thesis/forks/collection_of_results/er_exp_cx_exact/synthetic/er"

    pyr_header = "/Users/fredericbusch/Desktop/Thesis/forks/collection_of_results/er_exp_pyr/synthetic/er"

    all_rel_errors = [[], [], [], [], [], [], [], [], [], [], [], []]
    all_k_values = [[], [], [], [], [], [], [], [], [], [], [], []]

    colors = ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026']



    for n in range(4, 21):
        for c, r in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]):
            folder_name = f"n={n} p={p} r={r} re=False d=False"

            cx_path = os.path.join(cx_header, folder_name, "cx", "e=True p=False l=20 s=1", "cx_0_combined.csv")
            pyr_path = os.path.join(pyr_header, folder_name, 'pyr', "s=1000 sp=none", "pyr_0_combined.csv")

            cx_df = pd.read_csv(cx_path)
            pyr_df = pd.read_csv(pyr_path)

            cx_values = cx_df["avg_pos_k_bal"].to_numpy()
            pyr_values = pyr_df["pos_k_bal"].to_numpy()

            rel_error_values = pyr_values / cx_values
            k_values = np.asarray(range(len(pyr_values)), dtype=int)

            nan_indices = np.isnan(rel_error_values)

            all_k_values[c].extend(k_values[~nan_indices])
            all_rel_errors[c].extend(rel_error_values[~nan_indices])




    plt.figure(figsize=(8, 5))

    for c, r in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]):

        plt.scatter(all_k_values[c], all_rel_errors[c], label=f"r = {r}", marker='o', color=colors[c])


    plt.xlabel('k-balance', fontsize=12)
    plt.ylabel('rel error', fontsize=12)
    plt.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
    #plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])
    plt.xticks(range(1, 25))
    #plt.grid(True, linestyle='--', alpha=0.6)
    #plt.xticks(pyr_real_k_values[:until_value])
    plt.legend()
    plt.tight_layout()
    plt.show()
