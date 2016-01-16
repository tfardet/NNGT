import nngt
import nest
nest.Install("nngt_module")
import numpy as np
import matplotlib.pyplot as plt

nest.SetKernelStatus({"local_num_threads": 6})

from nngt.simulation import (make_nest_network, get_nest_network,
                             monitor_nodes, set_noise, set_poisson_input,
                             activity_types, plot_activity)



#-----------------------------------------------------------------------------#
# Build networks
#------------------------
#

#~ nmodel = "aeif_cond_exp"
nmodel = "aeif_cond_alpha"

pop = nngt.NeuralPop.ei_population(1000, en_model=nmodel, in_model=nmodel)

avg = 5.
if "exp" in nmodel:
    avg = 33.
    
#~ graph = nngt.SpatialNetwork(pop, weight_prop={"distrib":"gaussian", "distrib_prop":{"avg":avg}})

#~ nngt.generation.connect_neural_types(graph, 1, -1, "erdos_renyi", {"density": 0.03})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "newman_watts", {"coord_nb":10, "proba_shortcut": 0.1})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "erdos_renyi", {"density": 0.05})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "random_scale_free", {"in_exp":2.1, "out_exp":2.9, "density":0.05})
#~ nngt.generation.connect_neural_types(graph, -1, 1, "erdos_renyi", {"density": 0.05})
#~ nngt.generation.connect_neural_types(graph, -1, -1, "erdos_renyi", {"density": 0.01})
graph = nngt.generation.erdos_renyi(density=0.1, population=pop, weight_prop={"distrib":"gaussian", "distrib_prop":{"avg":avg}})
#~ nngt.generation.random_scale_free(2.2, 2.9, density=0.15, from_graph=graph)
#~ graph = nngt.generation.newman_watts(10, 0.1, population=pop)


#-----------------------------------------------------------------------------#
# NEST objects
#------------------------
#

subnet, gids = make_nest_network(graph)

recorders, record = monitor_nodes(gids, ["spike_detector"], [["spikes"]], network=graph)
recorders2, record2 = monitor_nodes((gids[0],), ["multimeter"], [["V_m","w"]])

set_noise(gids, 0., 200.)

rate = 20000.
if nmodel == "aeif_cond_exp":
    rate = 56000.

#~ set_poisson_input(gids[670:870], rate)
set_poisson_input(gids[:800], rate)
set_poisson_input(gids[800:], 0.75*rate)


#-----------------------------------------------------------------------------#
# Names
#------------------------
#

simtime = 1500.
nest.Simulate(simtime)

fignums = plot_activity(recorders, record, network=graph, show=False, hist=False, limits=(0,simtime))
#~ activity_types(graph, recorders, (0,simtime), raster=fignums[0], simplify=False)
activity_types(graph, recorders, (0,simtime), raster=fignums[0], simplify=True)

