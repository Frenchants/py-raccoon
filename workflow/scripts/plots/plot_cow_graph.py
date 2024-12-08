import os
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()

year = 1816
file_name = f"/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/correlates_of_war/combined_graphs/{year}.txt"

with open(file_name, "r") as file:
                for line in file:
                    x, y, z = line.split() 
                    G.add_edge(int(x), int(y), weight=int(z))


edge_colors = ['red' if G[u][v]['weight'] < 0 else 'blue' for u, v in G.edges]
pos = nx.spring_layout(G, seed=42, k=1.0) 

plt.figure(figsize=(14, 10))  

nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800)
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.7)

plt.axis('off') 
plt.tight_layout()  
plt.show()