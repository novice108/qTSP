from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx
import dwave_networkx
import dwave.inspector

# Create empty graph
G = nx.Graph()

nbr = 30
H=nx.gnp_random_graph(nbr,nbr)
G.add_edges_from(H.edges())

import random
for u,v,a in G.edges(data=True):
    G[u][v]['weight'] = 0.1


# ------- Run our QUBO on the QPU -------
# Set up QPU parameters

from datetime import datetime

startDW=datetime.now()
Q= defaultdict(float)
Q = dwave_networkx.algorithms.tsp.traveling_salesperson_qubo(G, weight='weight')
#print(Q)


import dimod
#import dwave_networkx as dnx
#rex = dnx.traveling_salesperson(G, dimod.ExactSolver(), start=0)
#print("Exact Solution ", rex)

bqm = dimod.dimod.BQM.from_qubo(Q)
#print(bqm)

from dwave.system import LeapHybridSampler
import numpy as np

result = LeapHybridSampler().sample(bqm, label='TSP Hybrid')
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
print("D-wave Hybrid Route ", route)
print("time of D-wave Hybrid ", datetime.now()-startDW)

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
