import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

main_folder = "/Users/fredericbusch/Desktop/Thesis/forks/collection_of_results/cow_results/real/cow"

output_folder = "/Users/fredericbusch/Desktop/Figures/cow"

years = range(1816, 2013)

for k in range(1, 218):
    cx_values = np.full(len(years), np.nan)
    pyr_values = np.full(len(years), np.nan)
    null_model_values = np.full(len(years), np.nan)
    pyr_null_model_values = np.full(len(years), np.nan)

    #total_cycle_counts = np.full(len(years), np.nan)

    for year in range(1816, 2013): 
        i = year - 1816
        cx_folder = os.path.join(main_folder, f"year={year}", "null_model_cow=False d=False", "cx", "e=False p=False l=10 s=20")
        cx_file_path = os.path.join(cx_folder, "cx_0_combined.csv")

        if os.path.exists(cx_file_path):
            df = pd.read_csv(cx_file_path)
            

            if "avg_pos_k_bal" in df.columns and len(df) - 1 >= k:
                value = df["avg_pos_k_bal"].iloc[k]
                cx_values[i] = value
        
        pyr_folder = os.path.join(main_folder, f"year={year}", "null_model_cow=False d=False", "pyr", "s=100 sp=None")
        pyr_file_path = os.path.join(pyr_folder, "pyr_0_combined.csv")
        if os.path.exists(pyr_file_path):
            df = pd.read_csv(pyr_file_path)
            

            if "pos_k_bal" in df.columns and len(df) - 1>= k:
                value = df["pos_k_bal"].iloc[k]
                pyr_values[i] = value
        
        null_model_folder = os.path.join(main_folder, f"year={year}", "null_model_cow=True d=False", "cx", "e=False p=False l=10 s=20")
        null_model_file_path = os.path.join(null_model_folder, "cx_0_combined.csv")
        if os.path.exists(null_model_file_path):
            df = pd.read_csv(null_model_file_path)
            

            if "avg_pos_k_bal" in df.columns and len(df) - 1>= k:
                value = df["avg_pos_k_bal"].iloc[k]
                null_model_values[i] = value
            
            """  if "total_est" in df.columns and len(df) - 1>= k:
                value = df["total_est"].iloc[k]
                total_cycle_counts[i] = value """
        
        null_model_folder = os.path.join(main_folder, f"year={year}", "null_model_cow=True d=False", "pyr", "s=100 sp=None")
        null_model_file_path = os.path.join(null_model_folder, "pyr_0_combined.csv")
        if os.path.exists(null_model_file_path):
            df = pd.read_csv(null_model_file_path)
            

            if "pos_k_bal" in df.columns and len(df) - 1>= k:
                value = df["pos_k_bal"].iloc[k]
                pyr_null_model_values[i] = value

    plt.figure(figsize=(10, 6))
    plt.plot(years, cx_values, marker='o', linestyle='-', color='red', label=f"CX k={k}")
    plt.plot(years, pyr_values, marker='o', linestyle='-', color='blue', label=f"FBE k={k}")
    plt.plot(years, null_model_values, marker='o', linestyle='-', color='black', label=f"Null k={k}")
    plt.xlabel("Year")
    plt.ylabel(f"{k}-balance")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_folder, f"cow_{k}_bal"), dpi=300)

    cx_avg = np.nanmean(cx_values)
    pyr_avg = np.nanmean(pyr_values)
    null_avg = np.nanmean(null_model_values)
    pyr_null_avg = np.nanmean(pyr_null_model_values)

    cx_std = np.nanstd(cx_values)
    pyr_std = np.nanstd(pyr_values)
    null_std = np.nanstd(null_model_values)
    pyr_null_std = np.nanstd(pyr_null_model_values)

    mae = np.nanmean(np.abs(pyr_values - cx_values))

    mape = np.nanmean(np.abs((pyr_values - cx_values) / cx_values))

    results = {
    'Metric': ['cx_avg', 'pyr_avg', 'cx_std', 
               'pyr_std', 'mae', 'mape', 'null_avg', 'null_std', 'pyr_null_avg', 'pyr_null_std'],
    'Value': [cx_avg, pyr_avg, cx_std, pyr_std, mae, mape, null_avg, null_std, pyr_null_avg, pyr_null_std]
}

    df = pd.DataFrame(results)

    output_file = os.path.join(output_folder, "cow_results", f'cow_results_{k}.csv')
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")
