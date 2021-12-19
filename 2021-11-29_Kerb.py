from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx
import dwave_networkx

# Create empty graph
G = nx.Graph()

# Add edges to the graph (also adds nodes)
#G.add_weighted_edges_from([ (0, 1, .1), (0, 2, .5), (0, 3, .1), (1, 2, .1), (1, 3, .5), (2, 3, .1) ])

nbr = 45
G=nx.gnp_random_graph(nbr,nbr)

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
#path1 = approx.christofides(G)
#print("path by  christofides ", path1)
#path2 = approx.traveling_salesman_problem(G)
#print("path by traveling_salesman_problem ", path2)

#https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.simulated_annealing_tsp.html#networkx.algorithms.approximation.traveling_salesman.simulated_annealing_tsp
#sym_anng = approx.simulated_annealing_tsp(G, (list(G) + [next(iter(G))]))
#print("path simulated_annealing_tsp ", sym_anng)

bqm = dimod.dimod.BQM.from_qubo(Q)
#print(bqm)

from dwave.system import LeapHybridSampler
import numpy as np
import hybrid

#result = hybrid.KerberosSampler().sample(bqm, qpu_params={'label': 'TSP - Hybrid Kerberos Cycle Graph'})
#print("Found solution with {} nodes at energy {}.".format(np.sum(result.record.sample), 
#                                                          result.first.energy))

#state_start = hybrid.State.from_problem(bqm)
#pt_workflow = hybrid.ParallelTempering(num_replicas=3, max_iter=5)
#result = pt_workflow.run(state_start, max_time=4).result()
#result = pt_workflow.run(state_start, max_iter=1, max_time=7, num_sweeps=1000).result()

import hybrid

state = hybrid.State.from_problem(bqm)
print("Field problem is of type: {}\nField samples is of type: {}".format(
       type(state.problem), type(state.samples)))

workflow = (hybrid.Parallel(hybrid.TabuProblemSampler(num_reads=1, timeout=1),
            hybrid.SimulatedAnnealingProblemSampler(num_reads=1)) |
            hybrid.ArgMin() )

state_updated = workflow.run(state).result()
print("Updated state has {} sample sets with lowest energy {}.".format(
       len(state_updated.samples), state_updated.samples.first.energy))

dimod_sampler = hybrid.HybridSampler(workflow)
solution = dimod_sampler.sample(bqm)
print("Best energy found is {}.".format(solution.first.energy))

from dwave.system import DWaveSampler

sampler_qpu = DWaveSampler(solver={'num_qubits__gt':1800})

result = hybrid.KerberosSampler().sample(bqm, max_iter=2, 
                                              max_subproblem_size=15,
                                              qpu_sampler=sampler_qpu,
                                              qpu_params={'label': 'Hybrid Computing 07.11 19:38'})

# use the sampler to find low energy states

sample = result.first.sample

#sample = solution.first.energy

route = [None]*len(G)
for (city, time), val in sample.items():
    if val:
        route[time] = city

start = None
if start is not None and route[0] != start:
    # rotate to put the start in front
    idx = route.index(start)
    route = route[-idx:] + route[:-idx]

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

print("Kerberos route ", route)

# Output route in directed graph
H = nx.DiGraph()
for z in range(len(route)-1):
    H.add_edge(route[z], route[z+1])

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
pos = nx.spring_layout(G, seed = 108)

nx.draw_networkx(G, pos=pos, edge_color='w', node_color='lightblue')
nx.draw_networkx(H, pos=pos, edge_color='b', node_color='lightblue')

filename = "plot.png" #save to file
plt.savefig(filename, bbox_inches='tight')

"""
#Make dimond conversion before this
import hybrid
dimod_sampler = hybrid.HybridSampler(workflow)
solution = dimod_sampler.sample(bqm)
print("Best energy found is {}.".format(solution.first.energy))
dwave.inspector.show(solution)
"""
