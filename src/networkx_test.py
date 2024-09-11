import networkx as nx

# Create a weighted graph
G = nx.Graph()
G.add_edge('A', 'B', weight=5)
G.add_edge('B', 'C', weight=3)
G.add_edge('A', 'C', weight=7)

# Iterate through all edges and get their weights
for u, v, data in G.edges(data=True):
    print(f"Edge ({u}, {v}) has weight {data['weight']}")


print(G['A']['B']['weight'])