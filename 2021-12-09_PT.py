from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx
import dwave_networkx

# Create empty graph
G = nx.Graph()

# Add edges to the graph (also adds nodes)
nbr = 30 #number of vertices
G=nx.gnp_random_graph(nbr,nbr)

import random
for u,v,a in G.edges(data=True):
    G[u][v]['weight'] = 0.1

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters
Q= defaultdict(float)
Q = dwave_networkx.algorithms.tsp.traveling_salesperson_qubo(G, weight='weight')
#print(Q)

import dimod #binary quadratic model
bqm = dimod.dimod.BQM.from_qubo(Q)
#print(bqm)

from dwave.system import LeapHybridSampler
import numpy as np
import hybrid

#Start Timer of Problem Computing
from datetime import datetime
startDW = datetime.now()

state_start = hybrid.State.from_problem(bqm)

pt_workflow = hybrid.ParallelTempering(num_replicas=3, max_iter=5)

result = pt_workflow.run(state_start, max_time=4).result()
#print("Found solution with {} nodes at energy {}.".format(np.sum(result.samples.record.sample),
#                                                          result.samples.first.energy))

sample = result.samples.first.sample
#print("result.samples.first.sample ", sample)

route = [None]*len(G)
for (city, time), val in sample.items():
    if val:
        route[time] = city

start = None
if start is not None and route[0] != start:
    # rotate to put the start in front
    idx = route.index(start)
    route = route[-idx:] + route[:-idx]

#end timer, calculate processing time of problem solving
print("time of computing TSP ", datetime.now()-startDW)

print("PT route ", route)
