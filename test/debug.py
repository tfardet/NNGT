import nngt
import nest
import numpy as np
import matplotlib.pyplot as plt

nest.SetKernelStatus({"local_num_threads": 8})

from nngt.core import GraphObject
from nngt.nest import make_nest_network, get_nest_network
from graph_tool.generation import random_rewire



#-----------------------------------------------------------------------------#
# Build networks
#------------------------
#

nmodel = "iaf_neuron"
nparam = { "t_ref": 2. }

pop = nngt.NeuralPop.ei_population(1000, en_model=nmodel, in_model=nmodel, en_param=nparam, in_param=nparam)
graph = nngt.SpatialNetwork(pop, weight_prop={"distrib":"gaussian", "distrib_prop":{"avg_distrib": 60.}})
#~ graph = nngt.SpatialNetwork(pop, weight_prop={"distrib":"lognormal"})
nngt.generation.connect_neural_types(graph, 1, -1, "erdos_renyi", {"density": 0.035})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "newman_watts", {"coord_nb":10, "proba_shortcut": 0.1})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "random_scale_free", {"in_exp": 2.1, "out_exp": 2.9, "density": 0.065})
nngt.generation.connect_neural_types(graph, 1, 1, "erdos_renyi", {"density": 0.077})
nngt.generation.connect_neural_types(graph, -1, 1, "erdos_renyi", {"density": 0.2})
nngt.generation.connect_neural_types(graph, -1, -1, "erdos_renyi", {"density": 0.04})


#-----------------------------------------------------------------------------#
# NEST objects
#------------------------
#

subnet, gids = make_nest_network(graph)

recorders, record = nngt.nest.monitor_nodes(gids, ["spike_detector", "multimeter"], [["spikes"], ["V_m"]], network=graph)

nngt.nest.set_noise(gids, 70., 80.)
nngt.nest.set_poisson_input(gids[570:870], 44000.)


#-----------------------------------------------------------------------------#
# Names
#------------------------
#

simtime = 10000
nest.Simulate(simtime)

nngt.nest.plot_activity(graph, recorders, record)
