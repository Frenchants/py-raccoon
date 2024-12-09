import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


cx_header = "/Users/fredericbusch/Desktop/Thesis/forks/collection_of_results/complete_exp_cx_exact/synthetic/complete"

pyr_header = "/Users/fredericbusch/Desktop/Thesis/forks/collection_of_results/complete_exp_pyr/synthetic/complete"

all_rel_errors = [[], [], [], [], [], [], [], [], [], [], [], []]
all_k_values = [[], [], [], [], [], [], [], [], [], [], [], []]

colors = ['#FFFF00', '#FFEB00', '#FFD700', '#FFB300', '#FF9C00', 
          '#FF8500', '#FF6D00', '#FF5500', '#FF3E00', '#FF1F00', '#FF0000']



for n in range(4, 21):
    for i in range(0, 11): # range(0, 11)
        r = i / 10
        if r == 0:
            r = int(r)
        folder_name = f"n={n} r={r} re=False d=False"

        cx_path = os.path.join(cx_header, folder_name, "cx", "e=True p=False l=20 s=1", "cx_0_combined.csv")
        pyr_path = os.path.join(pyr_header, folder_name, 'pyr', "s=1000 sp=none", "pyr_0_combined.csv")

        cx_df = pd.read_csv(cx_path)
        pyr_df = pd.read_csv(pyr_path)

        cx_values = cx_df["avg_pos_k_bal"].to_numpy()
        pyr_values = pyr_df["pos_k_bal"].to_numpy()

        rel_error_values = pyr_values / cx_values
        k_values = np.asarray(range(len(pyr_values)), dtype=int)

        nan_indices = np.isnan(rel_error_values)

        all_k_values[i].extend(k_values[~nan_indices])
        all_rel_errors[i].extend(rel_error_values[~nan_indices])




plt.figure(figsize=(8, 5))

for i in range(11):
    plt.scatter(all_k_values[i], all_rel_errors[i], label=f"r = {i / 10}", marker='o', color=colors[i])


plt.xlabel('k-balance', fontsize=12)
plt.ylabel('rel error', fontsize=12)
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])
plt.xticks(range(1, 25))
#plt.grid(True, linestyle='--', alpha=0.6)
#plt.xticks(pyr_real_k_values[:until_value])
plt.legend()
plt.tight_layout()
plt.show()
