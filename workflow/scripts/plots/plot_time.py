import os
import pandas as pd
import matplotlib.pyplot as plt

n_values = []
pyr_times = []
cx_times = []
cx_l11_times = []
cx_l12_times = []
n_l_values = []

# base folder
base_dir = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/plots/complete/complete_time_exp_cx_approx_and_pyr"

for i in range(4, 401):
    n_folder = os.path.join(base_dir, f"n={i} r=0.5 re=True d=False")
    
    # Read "pyr" data
    pyr_path = os.path.join(n_folder, "pyr", "s=1000 sp=None", "pyr_0_combined.csv")
    if os.path.exists(pyr_path):
        pyr_df = pd.read_csv(pyr_path)
        if "alg_run_time" in pyr_df.columns and not pyr_df.empty:
            pyr_times.append(pyr_df["alg_run_time"].iloc[0])
        else:
            pyr_times.append(None)  # Handle missing data
    else:
        pyr_times.append(None)  # Handle missing files

    # Read "cx" data
    cx_path = os.path.join(n_folder, "cx", "e=False p=False l=10 s=1000", "cx_0_combined.csv")
    if os.path.exists(cx_path):
        cx_df = pd.read_csv(cx_path)
        if "alg_run_time" in cx_df.columns and not cx_df.empty:
            cx_times.append(cx_df["alg_run_time"].iloc[0])
        else:
            cx_times.append(None)  # Handle missing data
    else:
        cx_times.append(None)  # Handle missing files
    
    # Record the current "i" value
    n_values.append(i)



base_dir = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/plots/complete/complete_time_exp_cx_approx"
for i in range(5, 401, 5):
    n_folder = os.path.join(base_dir, f"n={i} r=0.5 re=False d=False")

    cx_path = os.path.join(n_folder, "cx", "e=False p=False l=11 s=100", "cx_0_combined.csv")
    if os.path.exists(cx_path):
        cx_df = pd.read_csv(cx_path)
        if "alg_run_time" in cx_df.columns and not cx_df.empty:
            cx_l11_times.append(cx_df["alg_run_time"].iloc[0])
        else:
            cx_l11_times.append(None) 
    else:
        cx_l11_times.append(None)  

    n_l_values.append(i)

for i in range(5, 401, 5):
    n_folder = os.path.join(base_dir, f"n={i} r=0.5 re=False d=False")

    cx_path = os.path.join(n_folder, "cx", "e=False p=False l=12 s=10", "cx_0_combined.csv")
    if os.path.exists(cx_path):
        cx_df = pd.read_csv(cx_path)
        if "alg_run_time" in cx_df.columns and not cx_df.empty:
            cx_l12_times.append(cx_df["alg_run_time"].iloc[0])
        else:
            cx_l12_times.append(None) 
    else:
        cx_l12_times.append(None) 


# Plotting

plt.figure(figsize=(10, 6))

plt.plot(n_values, pyr_times, label="FBE s=1000", marker='o', linestyle='', color='blue')

plt.plot(n_values, cx_times, label="Approx. CX l=10 s=1000", marker='o', linestyle='', color='red')

plt.plot(n_l_values, cx_l11_times, label="Approx. CX l=11 s=100", marker='o', linestyle='', color='yellow')

print(cx_l11_times)

plt.plot(n_l_values, cx_l12_times, label="Approx. CX l=12 s=10", marker='o', linestyle='', color='orange')

# Customize the plot
plt.xlabel("Number of vertices n")
plt.ylabel("Runtime in seconds")
#plt.title("")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
