import nngt

from nngt.core import GraphObject
from graph_tool.generation import random_rewire

print(dir(nngt.lib))

graph = GraphObject()
random_rewire(graph)
graph = nngt.SpatialNetwork(100)
#~ nngt.generation.price_free_scale(2,initial_graph=graph)

graph2 = nngt.generation.newman_watts(20, 0.1, nodes=1000)

print(hasattr(graph,"_init_spatial_properties"))
print(graph.node_nb(),graph.edge_nb())
print(graph2.node_nb(),graph2.edge_nb())
