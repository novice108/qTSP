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
#G.add_edges_from([(1,2),(1,3),(2,4),(3,4),(3,5),(4,5)])
G.add_weighted_edges_from([ (0, 1, .1), (0, 2, .5), (0, 3, .1), (1, 2, .1), (1, 3, .5), (2, 3, .1) ])

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters

Q= defaultdict(float)
Q = dwave_networkx.algorithms.tsp.traveling_salesperson_qubo(G, weight='weight')
#print(Q)

import dimod
bqm = dimod.dimod.BQM.from_qubo(Q)
#print(bqm)

from dwave.system import LeapHybridSampler
import numpy as np

result = LeapHybridSampler().sample(bqm, label='TSP - Hybrid Computing 3')
print("Found solution with {} nodes at energy {}.".format(np.sum(result.record.sample), 
                                                          result.first.energy))

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

from networkx.algorithms import approximation as approx

cycle = approx.greedy_tsp(G)
print("path ", cycle)
