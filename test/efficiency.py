#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Compare the efficiency of the graph generation with graph_tool """

import timeit
import time

import numpy as np

import graph_tool as gt
import nngt
from nngt.core import GraphObject



#-----------------------------------------------------------------------------#
# Prepare the graphs and edges
#------------------------
#

size = 5000
edges = np.random.randint(0, size, (0.2*size**2,2))

g = gt.Graph()
g.add_vertex(size)

start = time.time()
h = nngt.Graph(size)
print("{} nodes without edges:".format(size), time.time() - start)

go = GraphObject(1000)


#-----------------------------------------------------------------------------#
# Prepare the graphs
#------------------------
#

number = 1

graphs = [g, h, go]
cmds = ['graph.add_edge_list(edges)', 'graph.add_edges(edges)', 'graph.new_edges(edges)' ]
times = [0., 0., 0.]

start = time.time()

for test in range(number):
    for i,graph in enumerate(graphs):
        times[i] += timeit.timeit(cmds[i], setup='from __main__ import graph, edges', number=1)
        #~ graph.clear_edges()
        
total = time.time() - start
print(times)
print(np.sum(times),total, timeit.timeit('nngt.generation.erdos_renyi(5000,0.2, multigraph=True)', setup="from __main__ import nngt", number=1))
