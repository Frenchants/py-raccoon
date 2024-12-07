import os
import pandas as pd
import matplotlib.pyplot as plt

# Initialize data holders
n_values = []
pyr_times = []
cx_times = []

# Base folder containing the "n=i" directories
base_dir = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/plots/complete/complete_time_exp_cx_approx_and_pyr"

# Iterate over the "n=i" directories
for i in range(4, 401):
    n_folder = os.path.join(base_dir, f"n={i} r=0.5 re=True d=False")
    print(n_folder)
    
    # Read "pyr" data
    pyr_path = os.path.join(n_folder, "pyr", "s=1000 sp=None", "pyr_0_combined.csv")
    if os.path.exists(pyr_path):
        pyr_df = pd.read_csv(pyr_path)
        if "alg_run_time" in pyr_df.columns and not pyr_df.empty:
            pyr_times.append(pyr_df["alg_only_mem"].iloc[0])
        else:
            pyr_times.append(None)  # Handle missing data
    else:
        pyr_times.append(None)  # Handle missing files

    # Read "cx" data
    cx_path = os.path.join(n_folder, "cx", "e=False p=False l=10 s=1000", "cx_0_combined.csv")
    if os.path.exists(cx_path):
        cx_df = pd.read_csv(cx_path)
        if "alg_run_time" in cx_df.columns and not cx_df.empty:
            cx_times.append(cx_df["alg_only_mem"].iloc[0])
        else:
            cx_times.append(None)  # Handle missing data
    else:
        cx_times.append(None)  # Handle missing files
    
    # Record the current "i" value
    n_values.append(i)

# Plotting

plt.figure(figsize=(10, 6))

plt.plot(n_values, pyr_times, label="FBE algorithm", marker='o', linestyle='', color='blue')

plt.plot(n_values, cx_times, label="CX algorithm", marker='o', linestyle='', color='red')

# Customize the plot
plt.xlabel("n")
plt.ylabel("Memory usage in bytes")
#plt.title("")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
