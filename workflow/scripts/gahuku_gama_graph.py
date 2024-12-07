import matplotlib.pyplot as plt
import networkx as nx

data = """
1	2	1
1	3	-1
2	3	-1
1	4	-1
3	4	1
1	5	-1
2	5	-1
1	6	-1
2	6	-1
3	6	1
3	7	1
5	7	1
6	7	1
3	8	1
4	8	1
6	8	1
7	8	1
2	9	-1
5	9	1
6	9	-1
2	10	-1
9	10	1
6	11	1
7	11	1
8	11	1
9	11	-1
10	11	-1
1	12	-1
6	12	1
7	12	1
8	12	1
11	12	1
6	13	-1
7	13	1
9	13	1
10	13	1
11	13	-1
5	14	1
8	14	-1
12	14	-1
13	14	1
1	15	1
2	15	1
5	15	-1
9	15	-1
10	15	-1
11	15	-1
12	15	-1
13	15	-1
1	16	1
2	16	1
5	16	-1
6	16	-1
11	16	-1
12	16	-1
13	16	-1
14	16	-1
15	16	1
"""

edges = [tuple(map(int, line.split())) for line in data.strip().split("\n")]

G = nx.Graph()
for u, v, weight in edges:
    G.add_edge(u, v, weight=weight)

edge_colors = ['red' if G[u][v]['weight'] < 0 else 'blue' for u, v in G.edges]


pos = nx.spring_layout(G, seed=42, k=1.0) 

plt.figure(figsize=(14, 10))  

nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800)
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.7)

plt.axis('off') 
plt.tight_layout()  
plt.show()
