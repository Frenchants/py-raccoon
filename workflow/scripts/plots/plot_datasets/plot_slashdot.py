import pandas as pd
import matplotlib.pyplot as plt
import os

until_value = 30

dataset_folder = "/Users/fredericbusch/Desktop/Thesis/forks/collection_of_results/slashdot_combined/real/slashdot"

real_folder = os.path.join(dataset_folder, "null_model=False d=False")
null_folder = os.path.join(dataset_folder, "null_model=True d=False")

cx_path_real = os.path.join(real_folder, "cx", "e=False p=False l=10 s=10", "cx_0_combined.csv")
cx_path_null = os.path.join(null_folder, "cx", "e=False p=False l=10 s=10", "cx_0_combined.csv")

pyr_path_real = os.path.join(real_folder, "pyr", "s=100 sp=None", "pyr_0_combined.csv")
pyr_path_null = os.path.join(null_folder, "pyr", "s=100 sp=None", "pyr_0_combined.csv")



pyr_df_real = pd.read_csv(pyr_path_real)
pyr_df_null = pd.read_csv(pyr_path_null)

pyr_real_values = pyr_df_real["pos_k_bal"]
pyr_null_values = pyr_df_null["pos_k_bal"]
pyr_real_k_values = range(len(pyr_real_values))
pyr_null_k_values = range(len(pyr_null_values))


cx_df_real = pd.read_csv(cx_path_real)
cx_df_null = pd.read_csv(cx_path_null)

cx_real_values = cx_df_real["avg_pos_k_bal"]
cx_null_values = cx_df_null["avg_pos_k_bal"]
cx_real_k_values = range(len(cx_real_values))
cx_null_k_values = range(len(cx_null_values))



plt.figure(figsize=(8, 5))
plt.plot(cx_real_k_values[:until_value], cx_real_values[:until_value], marker='o', label='Monte Carlo CX', color='red')
plt.plot(cx_null_k_values[:until_value], cx_null_values[:until_value], marker='o', label='Monte Carlo CX null model', color='grey')

plt.plot(pyr_real_k_values[:until_value], pyr_real_values[:until_value], marker='o', label='FBE', color='blue')
plt.plot(pyr_null_k_values[:until_value], pyr_null_values[:until_value], marker='o', label='FBE null model', color='black')

plt.xlabel('k', fontsize=12)
plt.ylabel('k-balance', fontsize=12)
plt.xticks(pyr_real_k_values[:until_value])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
