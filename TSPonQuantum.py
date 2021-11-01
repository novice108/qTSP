from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import dwave_networkx
import dwave.inspector

# Create empty graph
G = nx.Graph()

# Add edges to the graph (also adds nodes)
#G.add_edges_from([(1,2),(1,3),(2,4),(3,4),(3,5),(4,5)])
G.add_weighted_edges_from([ (0, 1, .1), (0, 2, .5), (0, 3, .1), (1, 2, .1), (1, 3, .5), (2, 3, .1) ])

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters
chainstrength = 2
numruns = 1000

Q= defaultdict(float)
Q = dwave_networkx.algorithms.tsp.traveling_salesperson_qubo(G, weight='weight')
#print(Q)


# Run the QUBO on the solver from your config file
sampler = EmbeddingComposite(DWaveSampler())

response = sampler.sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns, label='Example - TSP')
print(response)
import dimod
marshrutik = dwave_networkx.traveling_salesperson(G, dimod.ExactSolver(), start=0)
print(marshrutik)
marshrutik = dwave_networkx.traveling_salesperson(G, sampler, start=0)
print(marshrutik)
dwave.inspector.show(response)
