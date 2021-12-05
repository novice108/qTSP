from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx
import dwave_networkx
#import dwave.inspector

# Create empty graph
G = nx.Graph()

nbr = 105
H=nx.gnp_random_graph(nbr,nbr)
G.add_edges_from(H.edges())

import random
for u,v,a in G.edges(data=True):
    G[u][v]['weight'] = 0.1

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
pos = nx.spring_layout(G)


# ------- Run our QUBO on the QPU -------
# Set up QPU parameters

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



import hybrid

initial_state = hybrid.State.from_problem(bqm)
#print("initial state ==")
sub_size = 15
state_updated = hybrid.EnergyImpactDecomposer(size=sub_size, traversal="bfs").run(initial_state).result()
#print("state updated ==", state_updated)

#print(state_updated.subproblem.variables)

H = nx.from_edgelist(state_updated.subproblem.variables)
nx.draw_networkx(G, pos, node_color='b')
nx.draw_networkx(H, pos, node_color='r')

filename = "plot.png"
plt.savefig(filename, bbox_inches='tight')

#dwave.inspector.showr(result)
