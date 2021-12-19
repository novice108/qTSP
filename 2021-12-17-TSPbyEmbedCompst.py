from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import dwave_networkx
import dwave.inspector

# Create graph
G = nx.Graph()
G.add_weighted_edges_from([ (0, 1, .1), (0, 2, .5), (0, 3, .1), (1, 2, .1), (1, 3, .5), (2, 3, .1) ])

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
pos = nx.spring_layout(G, seed = 108)

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters
chainstrength = 2
numruns = 100

Q = defaultdict(float) # QUBO for TSP
Q = dwave_networkx.algorithms.tsp.traveling_salesperson_qubo(G, weight='weight')
#print(Q)

qpu = DWaveSampler()
# Run the QUBO on the solver from your config file
sampler = EmbeddingComposite(qpu, embedding_parameters=dict(timeout=10, tries=100))

response = sampler.sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns, label='Embed-TSP')
#print(response)
#print("response.record.sample", response.record.sample)

#import numpy as np
#print("response.record.sample in node numbers ", np.flatnonzero(response.record.sample == 1))

""" #EXACT SOLUTION ON EMULATOR OF QPU
import dimod
marshrutik = dwave_networkx.traveling_salesperson(G, dimod.ExactSolver(), start=0)
print("Exact Solution ", marshrutik)
"""

marshrutik = dwave_networkx.traveling_salesperson(G, sampler, start=0)
print("Quantum Anneling Approximation ", marshrutik)
dwave.inspector.show(response)

# Output route in directed graph
H = nx.DiGraph()
for z in range(len(marshrutik)-1):
    H.add_edge(z, z+1)

"""
print(H)
for u,v in H.edges():
    print(u,v)
"""

nx.draw_networkx(G, pos=pos, edge_color='r', node_color='lightblue')
nx.draw_networkx(H, pos=pos, edge_color='b', node_color='lightblue')

filename = "plot.png" #save to file
plt.savefig(filename, bbox_inches='tight')

"""
import networkx as nx
from matplotlib import pyplot as plt

path_car1 = marshrutik

paths = [path_car1]
colors = ['Red','Blue']

H = nx.DiGraph()
for path, color in zip(paths, colors):
    for edge in zip(path[:-1], path[1:]):
        H.add_edge(*edge, color=color)

edge_colors = nx.get_edge_attributes(H, 'color')

plt.figure(figsize=(10,7))
pos = nx.spring_layout(H, scale=20)
nx.draw(H, pos, 
        node_color='black',
        with_labels=True, 
        node_size=1200,
        edgelist=G.edges(),
        edge_color=edge_colors.values(),
        arrowsize=15,
        font_color='white',
        width=3,
        alpha=0.9)
"""
