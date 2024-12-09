import pandas as pd


filename = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/correlates_of_war/alliances.csv"

output_folder = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/correlates_of_war/alliance_graphs"

data = pd.read_csv(filename) 


required_columns = {'ccode1', 'ccode2', 'year'}
if not required_columns.issubset(data.columns):
    raise ValueError(f"CSV file must contain the following columns: {required_columns}")


data_by_year = data.groupby('year')


for year, group in data_by_year:
    edges = []

    for _, row in group.iterrows():
        edge = {
            "a": row['ccode1'],
            "b": row['ccode2'],
            "c": 1  # Weight is always +1
        }
        edges.append(edge)

    output = output_folder + f"/{year}.txt"
    with open(output, 'w') as f:
        for edge in edges:
            f.write(f"{edge['a']} {edge['b']} {edge['c']}\n")

    print(f"Edges for year {year} have been saved to '{year}.txt'.")