from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx
import dwave_networkx
#import dwave.inspector

# Create empty graph
G = nx.Graph()

H=nx.gnp_random_graph(30,30)
G.add_edges_from(H.edges())

import random
for u,v,a in G.edges(data=True):
    G[u][v]['weight'] = 0.1


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

bqm = dimod.dimod.BQM.from_qubo(Q)
print(bqm)

from dwave.system import LeapHybridSampler
import numpy as np

result = LeapHybridSampler().sample(bqm, label='TSP - Hybrid Computing Cycle Graph')
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
print(route)
