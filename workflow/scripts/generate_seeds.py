import random
import networkx as nx 

seeds = [random.randint(0, 2**32 - 1) for _ in range(50)]
print(seeds)