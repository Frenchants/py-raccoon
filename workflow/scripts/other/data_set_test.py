import networkx as nx
import random
import numpy as np

# # Create a directed graph
# G = nx.Graph()

# # Read the file and add edges
# with open("workflow/scripts/datasets/soc-sign-epinions.txt", "r") as file:
#     for line in file:
#         x, y, z = line.split()  # Split the line into x, y, z
#         G.add_edge(int(x), int(y), weight=float(z))  # Add an edge with weight

# # Check the graph
# print(f"Number of nodes: {G.number_of_nodes()}")
# print(f"Number of edges: {G.number_of_edges()}")


# # Assume G is your undirected graph
# # Find all connected components (returns sets of nodes)
# connected_components = list(nx.connected_components(G))
# nodes_to_remove = set(G.nodes) - connected_components[1]
# G.remove_nodes_from(nodes_to_remove)

# node_mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes)}


# # Relabel the nodes in the graph
# G = nx.relabel_nodes(G, node_mapping)

edges = [(i, i + 1, random.choice([-1, 1])) for i in range(10 - 1)]

edges = [(i, i + 1, random.choice([-1, 1])) for i in range(10 - 1)]
G = nx.Graph()
G.add_weighted_edges_from(edges)

print(G.nodes)

print(G.edges)


print(type(G[1][2]['weight']))

edges = np.array([(u,v,w) for (u,v,w) in G.edges.data('weight')])

print(G.edges.data('weight'))

