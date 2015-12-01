import nngt
import nest
import numpy as np
import matplotlib.pyplot as plt

nest.SetKernelStatus({"local_num_threads": 10})

from nngt.nest import (make_nest_network, get_nest_network, monitor_nodes,
                        set_noise, set_poisson_input)
from graph_tool.generation import random_rewire

#~ print(dir(nngt.lib))


#-----------------------------------------------------------------------------#
# Build networks
#------------------------
#

#~ nmodel = "aeif_cond_exp"
nmodel = "aeif_cond_alpha"

pop = nngt.NeuralPop.ei_population(1000, en_model=nmodel, in_model=nmodel)

avg = 33.
if nmodel == "aeif_cond_exp":
    avg = 120.
    
graph = nngt.SpatialNetwork(pop, weight_prop={"distrib":"gaussian", "distrib_prop":{"avg_distrib":avg}})

nngt.generation.connect_neural_types(graph, 1, -1, "erdos_renyi", {"density": 0.03})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "newman_watts", {"coord_nb":30, "proba_shortcut": 0.1})
nngt.generation.connect_neural_types(graph, 1, 1, "erdos_renyi", {"density": 0.09})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "random_scale_free", {"in_exp":2.1, "out_exp":2.9, "density":0.08})
nngt.generation.connect_neural_types(graph, -1, 1, "erdos_renyi", {"density": 0.2})
nngt.generation.connect_neural_types(graph, -1, -1, "erdos_renyi", {"density": 0.01})


#-----------------------------------------------------------------------------#
# NEST objects
#------------------------
#

subnet, gids = make_nest_network(graph)

recorders, record = monitor_nodes(gids, ["spike_detector"], [["spikes"]], network=graph)
recorders2, record2 = monitor_nodes((gids[0],), ["multimeter"], [["V_m","w"]])

set_noise(gids, 0., 100.)

rate = 20000.
if nmodel == "aeif_cond_exp":
    rate = 56000.

set_poisson_input(gids[670:870], rate)


#-----------------------------------------------------------------------------#
# Names
#------------------------
#

simtime = 2000
nest.Simulate(simtime)

nngt.nest.plot_activity(graph, recorders, record)
nngt.nest.plot_activity(graph, recorders2, record2)
