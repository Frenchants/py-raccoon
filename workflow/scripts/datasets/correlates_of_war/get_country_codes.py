import pandas as pd


input_file = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/correlates_of_war/country_codes.csv"

output_folder = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/correlates_of_war/country_codes"
data = pd.read_csv(input_file)


if 'CCode' not in data.columns:
    raise ValueError("The file does not contain a 'CCode' column.")

unique_ccodes = data['CCode'].dropna().unique()
unique_ccodes = sorted(unique_ccodes)

with open(output_folder + '/country_codes.txt', 'w') as f:
    for ccode in unique_ccodes:
        f.write(f"{ccode}\n")

print("Unique values from 'CCode' column have been saved to 'unique_ccodes.txt'.")
