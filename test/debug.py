import nngt
import nest
import numpy as np
import matplotlib.pyplot as plt

nest.SetKernelStatus({"local_num_threads": 8})

from nngt.core import GraphObject
from nngt.nest import make_nest_network, get_nest_network
from graph_tool.generation import random_rewire

#~ print(dir(nngt.lib))


#-----------------------------------------------------------------------------#
# Build networks
#------------------------
#

nmodel = "iaf_neuron"

pop = nngt.NeuralPop.ei_population(1000, en_model=nmodel, in_model=nmodel)
graph = nngt.SpatialNetwork(pop, weight_prop={"distrib":"gaussian", "distrib_prop":{"avg_distrib":60.}})
#~ graph = nngt.SpatialNetwork(pop, weight_prop={"distrib":"lognormal"})
nngt.generation.connect_neural_types(graph, 1, -1, "erdos_renyi", {"density": 0.03})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "erdos_renyi", {"density": 0.2})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "newman_watts", {"coord_nb":10, "proba_shortcut": 0.1})
nngt.generation.connect_neural_types(graph, 1, 1, "random_scale_free", {"in_exp":2.1, "out_exp":2.9, "density":0.08})
nngt.generation.connect_neural_types(graph, -1, 1, "erdos_renyi", {"density": 0.2})
nngt.generation.connect_neural_types(graph, -1, -1, "erdos_renyi", {"density": 0.01})
#~ nngt.generation.price_free_scale(2,initial_graph=graph)

#~ graph2 = nngt.generation.newman_watts(20, 0.1, nodes=1000)
#~ nngt.plot.degree_distribution(graph2,use_weights=False, show=False)
#~ graph2.set_weights()

#~ nngt.plot.degree_distribution(graph2,fignum=1)
#~ nngt.plot.degree_distribution(graph, "out", use_weights=False, show=False)
#~ nngt.plot.degree_distribution(graph, "out", fignum=1)

#~ nngt.plot.draw_network(graph, spatial=False, esize="betweenness", ncolor="group", nsize="betweenness")


#-----------------------------------------------------------------------------#
# NEST objects
#------------------------
#

subnet, gids = make_nest_network(graph)

#~ print(np.around(mat_adj.todense(),1))
#~ print(np.around(graph.adjacency_matrix().todense(),1))

recorders, record = nngt.nest.monitor_nodes(gids, ["spike_detector", "multimeter"], [["spikes"], ["V_m"]], network=graph)

nngt.nest.set_noise(gids, 0., 80.)
nngt.nest.set_poisson_input(gids[670:870], 100000.)


#-----------------------------------------------------------------------------#
# Names
#------------------------
#

simtime = 2000
nest.Simulate(simtime)

nngt.nest.plot_activity(graph, recorders, record)
