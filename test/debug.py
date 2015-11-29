import nngt
#~ import nest
#~ import nest.raster_plot
import numpy as np
import matplotlib.pyplot as plt

#~ nest.SetKernelStatus({"local_num_threads": 8})

from nngt.core import GraphObject
from nngt.nest import make_nest_network, get_nest_network
from graph_tool.generation import random_rewire

#~ print(dir(nngt.lib))


#-----------------------------------------------------------------------------#
# Build networks
#------------------------
#

graph = GraphObject()
random_rewire(graph)
pop = nngt.NeuralPop.ei_population(1000)
graph = nngt.SpatialNetwork(pop, weight_prop={"distrib":"gaussian", "distrib_prop":{"avg_distrib":15.}})
#~ graph = nngt.SpatialNetwork(pop, weight_prop={"distrib":"lognormal"})
nngt.generation.connect_neural_types(graph, 1, -1, "erdos_renyi", {"density": 0.01})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "erdos_renyi", {"density": 0.2})
#~ nngt.generation.connect_neural_types(graph, 1, 1, "newman_watts", {"coord_nb":10, "proba_shortcut": 0.1})
nngt.generation.connect_neural_types(graph, 1, 1, "random_scale_free", {"in_exp":2.1, "out_exp":2.9, "density":0.01})
#~ nngt.generation.connect_neural_types(graph, -1, 1, "erdos_renyi", {"density": 0.15})
nngt.generation.connect_neural_types(graph, -1, -1, "erdos_renyi", {"density": 0.01})
#~ nngt.generation.price_free_scale(2,initial_graph=graph)

#~ graph2 = nngt.generation.newman_watts(20, 0.1, nodes=1000)
#~ nngt.plot.degree_distribution(graph2,use_weights=False, show=False)
#~ graph2.set_weights()

#~ nngt.plot.degree_distribution(graph2,fignum=1)
#~ nngt.plot.degree_distribution(graph, "out", use_weights=False, show=False)
#~ nngt.plot.degree_distribution(graph, "out", fignum=1)

nngt.plot.draw_network(graph, spatial=False, esize="betweenness", ncolor="group", nsize="betweenness")


#-----------------------------------------------------------------------------#
# NEST objects
#------------------------
#

#~ subnet, gids = make_nest_network(graph)

#~ mat_adj = get_nest_network(subnet, graph.id_from_nest_id)
#~ 
#~ print(np.around(mat_adj.todense(),1))
#~ print(np.around(graph.adjacency_matrix().todense(),1))

#~ multimeter = nest.Create("multimeter")
#~ nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})
#~ 
#~ bg_noise = nest.Create("noise_generator")
#~ nest.SetStatus(bg_noise,{"mean": 300., "std": 80.})
#~ noise = nest.Create("poisson_generator")
#~ nest.SetStatus(noise,{"rate": 20000.})
#~ nest.SetStatus(noise,{"rate": 4000.})
#~ nest.Connect(noise,gids[170:270])
#~ nest.Connect(noise,tuple(graph.nest_id[170:270]))
#~ nest.Connect(bg_noise,gids)
#~ nest.Connect(multimeter,(20,))
#~ 
#~ spikes = nest.Create("spike_detector")
#~ nest.SetStatus(spikes,{"label": "spikes","withtime": True, "withgid": True})
#~ nest.Connect(gids, spikes)


#-----------------------------------------------------------------------------#
# Names
#------------------------
#

simtime = 50000
#~ nest.Simulate(simtime)
#~ 
#~ dmm = nest.GetStatus(multimeter)[0]
#~ da_voltage = dmm["events"]["V_m"]
#~ da_time = dmm["events"]["times"]

#~ plt.plot(da_time,da_voltage,'k')

#~ nest.raster_plot.from_device(spikes, hist=True)
#~ plt.show()
