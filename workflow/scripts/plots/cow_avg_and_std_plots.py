import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder_path = '/Users/fredericbusch/Desktop/Figures/cow/cow_results' 

output_folder = "/Users/fredericbusch/Desktop/Figures/cow/more_cow_results"

mae_values = []
mape_values = []

cx_std_values = []
pyr_std_values = []

cx_avg_values = []
pyr_avg_values = []

null_std_values = []
null_avg_values = []

pyr_null_std_values = []
pyr_null_avg_values = []

x_values = list(range(1, 41))  

for i in x_values:
    file_path = os.path.join(folder_path, f'cow_results_{i}.csv')
    
    df = pd.read_csv(file_path, header=None) 
    
    cx_avg_value = df[df[0] == 'cx_avg'][1].values[0]
    pyr_avg_value = df[df[0] == 'pyr_avg'][1].values[0] 
    null_avg_value = df[df[0] == 'null_avg'][1].values[0] 
    pyr_null_avg_value = df[df[0] == 'pyr_null_avg'][1].values[0] 

    #mae_value = df[df[0] == 'mae'][1].values[0]
    #mape_value = df[df[0] == 'mape'][1].values[0]
    cx_std_value = df[df[0] == 'cx_std'][1].values[0]
    pyr_std_value = df[df[0] == 'pyr_std'][1].values[0]
    null_std_value = df[df[0] == 'null_std'][1].values[0]
    pyr_null_std_value = df[df[0] == 'pyr_null_std'][1].values[0]
        
    cx_avg_values.append(cx_avg_value)
    pyr_avg_values.append(pyr_avg_value)
    #mae_values.append(mae_value)
    #mape_values.append(mape_values)
    cx_std_values.append(cx_std_value)
    pyr_std_values.append(pyr_std_value)

    null_std_values.append(null_std_value)
    null_avg_values.append(null_avg_value)

    pyr_null_std_values.append(pyr_null_std_value)
    pyr_null_avg_values.append(pyr_null_avg_value)


cx_avg_values = np.array(cx_avg_values, dtype=float)
pyr_avg_values = np.array(pyr_avg_values, dtype=float)
null_avg_values = np.array(null_avg_values, dtype=float)
null_std_values = np.array(null_std_values, dtype=float)
pyr_null_avg_values = np.array(pyr_null_avg_values, dtype=float)
pyr_null_std_values = np.array(pyr_null_std_values, dtype=float)
#mae_values = np.array(mae_values, dtype=float)
#mape_values = np.array(mape_values, dtype=float)

pyr_std_values = np.array(pyr_std_values, dtype=float)
cx_std_values = np.array(cx_std_values, dtype=float)
x_values = np.array(x_values, dtype=float)

cx_avg_nan_indices = np.isnan(cx_avg_values)
pyr_avg_nan_indices = np.isnan(pyr_avg_values)
null_avg_nan_indices = np.isnan(null_avg_values)
null_std_nan_indices = np.isnan(null_std_values)
pyr_null_avg_nan_indices = np.isnan(pyr_null_avg_values)
pyr_null_std_nan_indices = np.isnan(pyr_null_std_values)

cx_std_nan_indices = np.isnan(cx_std_values)
pyr_std_nan_indices = np.isnan(pyr_std_values)

#mae_nan_indices = np.isnan(mae_values)
#mape_nan_indices = np.isnan(mape_values)

plt.figure(figsize=(10, 6))
plt.ylim(0, 1)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.plot(x_values[~cx_avg_nan_indices], cx_avg_values[~cx_avg_nan_indices], label='cx_avg', marker='o', color='red')
plt.plot(x_values[~pyr_avg_nan_indices], pyr_avg_values[~pyr_avg_nan_indices], label='fbe_avg', marker='s', color='blue')
plt.plot(x_values[~null_avg_nan_indices], null_avg_values[~null_avg_nan_indices], label='cx_ull_avg', marker='x', color='black')
plt.plot(x_values[~pyr_null_avg_nan_indices], pyr_null_avg_values[~pyr_null_avg_nan_indices], label='fbe_null_avg', marker='x', color='grey')

plt.xlabel('Cycle length k')
plt.ylabel('Avg balance')
#plt.title('Comparison of cx_avg and pyr_avg values')
plt.legend()

plt.grid(True)
plt.savefig(os.path.join(output_folder, f"cow_avg_bal"), dpi=300)
print("Saved avg bal")



plt.figure(figsize=(10, 6))
plt.ylim(0, 1)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.plot(x_values[~cx_std_nan_indices], cx_std_values[~cx_std_nan_indices], label='cx_std', marker='o', color='red')
plt.plot(x_values[~pyr_std_nan_indices], pyr_std_values[~pyr_std_nan_indices], label='fbe_std', marker='s', color='blue')
plt.plot(x_values[~null_std_nan_indices], null_std_values[~null_std_nan_indices], label='cx_null_std', marker='x', color='black')
plt.plot(x_values[~pyr_null_std_nan_indices], pyr_null_std_values[~pyr_null_std_nan_indices], label='fbe_null_avg', marker='x', color='grey')

plt.xlabel('Cycle length k')
plt.ylabel('Std balance')
#plt.title('Comparison of cx_avg and pyr_avg values')
plt.legend()

plt.grid(True)
plt.savefig(os.path.join(output_folder, f"cow_std_bal"), dpi=300)
print("Saved std bal")

