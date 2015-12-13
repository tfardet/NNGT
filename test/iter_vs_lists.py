#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Compare the efficiency of iteration versus lists """

import timeit
import time
from multiprocessing import Pool, Array

import numpy as np

import graph_tool as gt
import nngt
from nngt.core import GraphObject



#-----------------------------------------------------------------------------#
# Prepare the graphs and edges
#------------------------
#

p = Pool(5)
size = 10000
edges = np.random.randint(0, size, (0.2*size**2,2))

g = gt.Graph()
g.add_vertex(size)
start = time.time()
g.add_edge_list(edges)
print(">> edges added: {}".format(time.time()-start))

# def edge prop
eprop = g.new_edge_property("double")
g.edge_properties["test"] = eprop

def _set_eprop(eprop, tpl):
    eprop[tpl[1]] = tpl[1]

def iter_version(graph):
    ep = Array('d',graph.edge_properties["test"].a)
    p.map(lambda x: _set_eprop(ep, x), zip(graph.edges(), range(graph.num_edges())))

def list_version(graph):
    values = np.arange(0,graph.num_edges())
    graph.edge_properties["test"].a = values

start = time.time()
iter_version(g)
print(">> MP: {}".format(time.time()-start))
#~ print(timeit.timeit("iter_version(g)", setup="from __main__ import g,iter_version", number=1))
#~ print(timeit.timeit("list_version(g)", setup="from __main__ import g,list_version", number=1))
