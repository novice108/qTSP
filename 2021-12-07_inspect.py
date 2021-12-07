from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx
import dwave_networkx
import dwave.inspector

# Create empty graph
G = nx.Graph()

nbr = 7
H=nx.gnp_random_graph(nbr,nbr)
G.add_edges_from(H.edges())

import random
for u,v,a in G.edges(data=True):
    G[u][v]['weight'] = 0.1


# ------- Run our QUBO on the QPU -------
# Set up QPU parameters
Q= defaultdict(float)
Q = dwave_networkx.algorithms.tsp.traveling_salesperson_qubo(G, weight='weight')
#print("Q= ", Q)

import dimod
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite

# Define problem
bqm = dimod.dimod.BQM.from_qubo(Q)
#print("bqm= ", bqm)
# Get sampler
sampler = EmbeddingComposite(DWaveSampler(solver=dict(qpu=True)))
#print("sampler= ", sampler)
# Sample with low chain strength
sampleset = sampler.sample(bqm, num_reads=2, chain_strength=2)
#print("sampleset= ", sampleset)
dwave.inspector.show(sampleset)

"""
import dimod
bqm = dimod.dimod.BQM.from_qubo(Q)
#print(bqm)

from dwave.system import LeapHybridSampler
import hybrid
initial_state = hybrid.State.from_problem(bqm)
#print("initial state ==", initial_state)
sub_size = 20

#state_updated = hybrid.EnergyImpactDecomposer(size=sub_size, traversal="bfs").run(initial_state).result()
#print("state updated ==", state_updated)
#print(state_updated.subproblem.variables)

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
pos = nx.spring_layout(G, seed = 120)

H = nx.from_edgelist(state_updated.subproblem.variables)
nx.draw_networkx(G, pos, node_color='b')
nx.draw_networkx(H, pos, node_color='r')
filename = "plot.png"
plt.savefig(filename, bbox_inches='tight')
"""
"""
state = initial_state
workflow = (hybrid.Parallel(hybrid.TabuProblemSampler(num_reads=2, timeout=10),
            hybrid.SimulatedAnnealingProblemSampler(num_reads=1)) |
            hybrid.ArgMin() )
state_updated = workflow.run(state).result()
print("Updated state has {} sample sets with lowest energy {}.".format(
       len(state_updated.samples), state_updated.samples.first.energy))
dimod_sampler = hybrid.HybridSampler(workflow)
solution = dimod_sampler.sample(bqm)
print("Best energy found is {}.".format(solution.first.energy))
#print("solution", solution)
#print("state_updated.samples.first.energy ", state_updated.samples)
#print("state updated ==", state_updated)
"""
