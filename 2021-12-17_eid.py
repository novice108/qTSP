from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx
import dwave_networkx
import dwave.inspector

# Create empty graph
G = nx.Graph()

nbr = 45 #105
H=nx.gnp_random_graph(nbr,nbr)
G.add_edges_from(H.edges())

import random
for u,v,a in G.edges(data=True):
    G[u][v]['weight'] = 0.1

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
pos = nx.spring_layout(G, seed = 108)

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters
Q= defaultdict(float) #create QUBO for TSP
Q = dwave_networkx.algorithms.tsp.traveling_salesperson_qubo(G, weight='weight')
#print(Q)


import dimod
bqm = dimod.dimod.BQM.from_qubo(Q)
#print(bqm)

from dwave.system import LeapHybridSampler
import numpy as np



import hybrid

initial_state = hybrid.State.from_problem(bqm)
#print(initial_state)
##### NOT WORKING ########################
#eid = hybrid.EnergyImpactDecomposer(size=45)
#print(eid.run(initial_state).result)

#eid = hybrid.EnergyImpactDecomposer(size=15, traversal='bfs')
#print(eid.run(initial_state).result)

#eid = hybrid.EnergyImpactDecomposer(size=15, min_gain=None, rolling=True, rolling_history=1.0, silent_rewind=True)
#print(eid.run(initial_state).result)


#w = hybrid.EnergyImpactDecomposer(size=15, traversal='bfs') | hybrid.QPUSubproblemAutoEmbeddingSampler() | hybrid.SplatComposer()
#print(initial_state.problem)
#result = w.run(initial_state).result()
#print(result.samples)
#print(initial_state.samples.record.energy)
#print(w.timers)
#print(hybrid.print_conters(w))



"""
#print("initial state ==", initial_state)
sub_size = 15
subgraphs = 3

#state_updated = hybrid.EnergyImpactDecomposer(size=sub_size, traversal="bfs").run(initial_state).result()

iteration = (hybrid.EnergyImpactDecomposer(size=sub_size, rolling_history=0.85, traversal="bfs") | 
             hybrid.Lambda(lambda _, s: s.updated(
                               rolled_variables=s.rolled_variables+[list(s.subproblem.variables)])))

workflow = hybrid.LoopUntilNoImprovement(iteration, max_iter=3) 

state_updated = workflow.run(initial_state.updated(rolled_variables=[])).result()



#print(state_updated.subproblem.variables)
#Extract decomposition subproblems data from problem
H0 = nx.from_edgelist(state_updated.rolled_variables[0])
H1 = nx.from_edgelist(state_updated.rolled_variables[1])
H2 = nx.from_edgelist(state_updated.rolled_variables[2])

#draw 3 graphs with decomposition
f, axs = plt.subplots(3,1,figsize=(10,10))
nx.draw_networkx(G, pos=pos, edge_color='w', ax=axs[0], node_color='lightblue')
nx.draw_networkx(H0, pos=pos, edge_color='w', ax=axs[0], node_color='r')

nx.draw_networkx(G, pos=pos, edge_color='w', ax=axs[1], node_color='lightblue')
nx.draw_networkx(H1, pos=pos, edge_color='w', ax=axs[1], node_color='g')

nx.draw_networkx(G, pos=pos, edge_color='w', ax=axs[2], node_color='lightblue')
nx.draw_networkx(H2, pos=pos, edge_color='w', ax=axs[2], node_color='y')

filename = "plot.png" #save to file
plt.savefig(filename, bbox_inches='tight')
"""
