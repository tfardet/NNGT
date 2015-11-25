import nngt
import numpy as np

from nngt.core import GraphObject
from graph_tool.generation import random_rewire

#~ print(dir(nngt.lib))

graph = GraphObject()
random_rewire(graph)
pop = nngt.NeuralPop.ei_population(1000)
graph = nngt.SpatialNetwork(pop)
nngt.generation.connect_neural_types(graph, -1, 1, "erdos_renyi", {"density": 0.1})
#~ nngt.generation.price_free_scale(2,initial_graph=graph)

graph2 = nngt.generation.newman_watts(20, 0.1, nodes=1000)
nngt.plot.degree_distribution(graph2,use_weights=False, show=False)
graph2.set_weights()

nngt.plot.degree_distribution(graph2,fignum=1)
#~ nngt.plot.degree_distribution(graph, "out")

print(graph.node_nb(),graph.edge_nb())
print(graph2.node_nb(),graph2.edge_nb())
