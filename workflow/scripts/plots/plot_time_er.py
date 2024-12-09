import os
import pandas as pd
import matplotlib.pyplot as plt

n_values = range(5, 101, 5)
pyr_times = []
cx_times = []

base_dir = "/Users/fredericbusch/Desktop/Thesis/forks/collection_of_results/er_time_exp/synthetic/er"

p = 0.25

for i in range(5, 101, 5):
    n_folder = os.path.join(base_dir, f"n={i} p={p} r=0.5 re=False d=False")
    
    pyr_path = os.path.join(n_folder, "pyr", "s=1000 sp=None", "pyr_0_combined.csv")
    cx_path = os.path.join(n_folder, "cx", "e=False p=False l=10 s=1000", "cx_0_combined.csv")

    print(" ")
    print(" ")
    print(" ")
    print(pyr_path)
    print(" ")
    print(" ")
    print(" ")



    pyr_df = pd.read_csv(pyr_path)
    cx_df = pd.read_csv(cx_path)
    
    pyr_times.append(pyr_df["alg_run_time"].iloc[0])
    cx_times.append(cx_df["alg_run_time"].iloc[0])



# Plotting

plt.figure(figsize=(10, 6))

plt.plot(n_values, pyr_times, label="FBE s=1000", marker='o', linestyle='--', color='blue')

plt.plot(n_values, cx_times, label="Approx. CX l=10 s=1000", marker='o', linestyle='--', color='red')

plt.xlabel("Number of vertices n")
plt.ylabel("Runtime in seconds")
#plt.title("")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
