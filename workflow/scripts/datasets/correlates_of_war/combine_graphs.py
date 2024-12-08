import os

# Define folder paths
alliances_folder = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/correlates_of_war/alliances_graphs"
disputes_folder = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/correlates_of_war/disputes_graphs"
combined_folder = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/correlates_of_war/combined_graphs"

os.makedirs(combined_folder, exist_ok=True)

years = [f[:-4] for f in os.listdir(alliances_folder) if f.endswith('.txt')]

# Process each year
for year in years:
    alliances_file = os.path.join(alliances_folder, f"{year}.txt")
    disputes_file = os.path.join(disputes_folder, f"{year}.txt")
    combined_file = os.path.join(combined_folder, f"{year}.txt")
    
    with open(combined_file, 'w') as outfile:
        # Append alliances
        if os.path.exists(alliances_file):
            with open(alliances_file, 'r') as af:
                outfile.write(af.read())
        
        # Append disputes
        if os.path.exists(disputes_file):
            with open(disputes_file, 'r') as df:
                outfile.write(df.read())
    
    print(f"Combined file for year {year} saved to '{combined_file}'.")
