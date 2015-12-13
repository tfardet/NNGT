import nngt
import nest
nest.Install("nngt_module")
import numpy as np
import matplotlib.pyplot as plt

nest.SetKernelStatus({"local_num_threads": 6})

from nngt.simulation import (make_nest_network, get_nest_network,
                             monitor_nodes, set_noise, set_poisson_input)



#-----------------------------------------------------------------------------#
# Build networks
#------------------------
#

#~ nmodel = "aeif_cond_exp"
nmodel = "aeif_cond_exp"

pop = nngt.NeuralPop.ei_population(1000, en_model=nmodel, in_model=nmodel)

avg = 33.
if "aeif_cond_exp" in nmodel:
    avg = 120.
    
graph = nngt.SpatialNetwork(pop, weight_prop={"distrib":"gaussian", "distrib_prop":{"avg":avg}})

#~ nngt.generation.connect_neural_types(graph, 1, -1, "erdos_renyi", {"density": 0.03})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "newman_watts", {"coord_nb":30, "proba_shortcut": 0.1})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "erdos_renyi", {"density": 0.09})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "random_scale_free", {"in_exp":2.1, "out_exp":2.9, "density":0.08})
#~ nngt.generation.connect_neural_types(graph, -1, 1, "erdos_renyi", {"density": 0.2})
#~ nngt.generation.connect_neural_types(graph, -1, -1, "erdos_renyi", {"density": 0.01})
#~ nngt.generation.erdos_renyi(density=0.09, from_graph=graph)
nngt.generation.random_scale_free(2.2, 2.9, density=0.09, from_graph=graph)


#-----------------------------------------------------------------------------#
# NEST objects
#------------------------
#

subnet, gids = make_nest_network(graph)
print nest.GetConnections()[0]
print(nest.GetStatus((nest.GetConnections()[0],)))

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

simtime = 1000.
nest.Simulate(simtime)

nngt.simulation.plot_activity(recorders, record, network=graph)
nngt.simulation.plot_activity(recorders2, record2, network=graph)
