import networkx as nx

# Create empty graph
G = nx.Graph()

nbr = 105
H=nx.gnp_random_graph(nbr,nbr)
G.add_edges_from(H.edges())

import random
for u,v,a in G.edges(data=True):
    G[u][v]['weight'] = 0.1

from networkx.algorithms import approximation as approx
from datetime import datetime

start = datetime.now()
cycle = approx.greedy_tsp(G)
print("path greedy_tsp", cycle)
print("time of greedy_tsp ", datetime.now()-start)

start=datetime.now()
path1 = approx.christofides(G)
print("path by christofides ", path1)
print("time of christofides ", datetime.now()-start)

start=datetime.now()
path2 = approx.traveling_salesman_problem(G, weight='weight', nodes=None, cycle=False, method=None)
print("path by traveling_salesman_problem ", path2)
print("time of traveling_salesman_problem ", datetime.now()-start)

start=datetime.now()
path3 = approx.threshold_accepting_tsp(G, "greedy", weight='weight', source=None, threshold=1, move='1-1', max_iterations=10, N_inner=100, alpha=0.1, seed=None)
print("path by threshold_accepting_tsp ", path3)
print("time of threshold_accepting_tsp ", datetime.now()-start)

start=datetime.now()
sym_anng = approx.simulated_annealing_tsp(G, (list(G) + [next(iter(G))]))
print("path simulated_annealing_tsp ", sym_anng)
print("time of simulated_annealing_tsp ", datetime.now()-start)
