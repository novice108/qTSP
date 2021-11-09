from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx
import dwave_networkx
#import dwave.inspector

# Create empty graph
G = nx.Graph()

# Add edges to the graph (also adds nodes)
#G.add_weighted_edges_from([ (0, 1, .1), (0, 2, .5), (0, 3, .1), (1, 2, .1), (1, 3, .5), (2, 3, .1) ])

H=nx.gnp_random_graph(30,30)
G.add_edges_from(H.edges())

import random
for u,v,a in G.edges(data=True):
    G[u][v]['weight'] = 0.1
    #G[u][v]['weight'] = random.random()

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters

Q= defaultdict(float)
Q = dwave_networkx.algorithms.tsp.traveling_salesperson_qubo(G, weight='weight')
#print(Q)


import dimod
#import dwave_networkx as dnx
#rex = dnx.traveling_salesperson(G, dimod.ExactSolver(), start=0)
#print("Exact Solution ", rex)

from networkx.algorithms import approximation as approx

#cycle = approx.greedy_tsp(G)
#print("path ", cycle)
path1 = approx.christofides(G)
print("path by  christofides ", path1)
#path2 = approx.traveling_salesman_problem(G)
#print("path by traveling_salesman_problem ", path2)

#https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.simulated_annealing_tsp.html#networkx.algorithms.approximation.traveling_salesman.simulated_annealing_tsp
sym_anng = approx.simulated_annealing_tsp(G, (list(G) + [next(iter(G))]))
print("path simulated_annealing_tsp ", sym_anng)

bqm = dimod.dimod.BQM.from_qubo(Q)
#print(bqm)

from dwave.system import LeapHybridSampler
import numpy as np
import hybrid

result = hybrid.KerberosSampler().sample(bqm, qpu_params={'label': 'TSP - Hybrid Kerberos Cycle Graph'})
#print("Found solution with {} nodes at energy {}.".format(np.sum(result.record.sample), 
#                                                          result.first.energy))

# use the sampler to find low energy states

sample = result.first.sample

route = [None]*len(G)
for (city, time), val in sample.items():
    if val:
        route[time] = city

start = None
if start is not None and route[0] != start:
    # rotate to put the start in front
    idx = route.index(start)
    route = route[-idx:] + route[:-idx]
print("Kerberos route ", route)
