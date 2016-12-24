#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Constant values for NNGT """

from os.path import expanduser
import sys

import scipy as sp
import scipy.sparse as ssp

import nngt



#-----------------------------------------------------------------------------#
# All and version
#------------------------
#

__all__ = [
    "version",
    "default_neuron",
    "default_synapse",
    "POS",
    "DIST"
    "WEIGHT",
    "DELAY",
    "TYPE",
    "use_library"
]

version = '0.6'
''' :class:`string`, the current version '''


#-----------------------------------------------------------------------------#
# Python 2/3 compatibility
#------------------------
#

# compatible reload

reload_module = None
if sys.hexversion >= 0x03000000 and sys.hexversion < 0x03040000:
    import imp
    reload_module = imp.reload
elif sys.hexversion >= 0x03040000:
    import importlib
    reload_module = importlib.reload
else:
    reload_module = reload


#-----------------------------------------------------------------------------#
# Config
#------------------------
#

def _load_config(path_config):
    ''' Load `~/.nngt.conf` and parse it, return the settings '''
    config = {
        'graph_library': "",
        'library': None,
        'graph': object,
        'set_logging': False,
        'load_nest': False,
        'with_nest': False,
        'with_plot': False,
        'to_file': False,
        'log_folder': "~/.nngt/database",
        'db_url': "mysql:///nngt_db",
        'backend': None,
        'color_lib': 'matplotlib',
        'palette': 'Set1',
        'multithreading': False,
        'omp': 1,
    }
    with open(path_config, 'r') as fconfig:
        options = [l.strip() for l in fconfig if l.strip() and l[0] != "#"]
        for opt in options:
            sep = opt.find("=")
            opt_name, opt_value = opt[:sep].strip(), opt[sep+1:].strip()
            config[opt_name] = opt_value if opt_value != "False" else False
            print(opt_name, opt_value)
    return config

config = _load_config(nngt.path_config)


def not_implemented(*args, **kwargs):
    return NotImplementedError("Not implemented yet.")

analyze_graph = {
    'adjacency': not_implemented,
    'assortativity': not_implemented,
    'betweenness': not_implemented,
    'clustering': not_implemented,
    'diameter': not_implemented,
    'ebetweenness': not_implemented,
    'get_edges': not_implemented,
    'nbetweenness': not_implemented,
    'reciprocity': not_implemented,
    'scc': not_implemented,
    'wcc': not_implemented,
}

def set_config(dic_config):
    for key in dic_config:
        if key not in nngt.config:
            raise KeyError("Unknown configuration property: {}".format(key))
    nngt.config.update(dic_config)
    if "omp" in dic_config and nngt.config["graph_library"] == "graph-tool":
        nngt.config["library"].openmp_set_num_threads(nngt.config["omp"])

def get_config(key=None):
    if key is None:
        return {key: val for key, val in nngt.config.items()}
    else:
        res = nngt.config[key]
        return res


#-----------------------------------------------------------------------------#
# Graph libraries
#------------------------
#

def _set_graph_tool():
    '''
    Set graph-tool as graph library, store relevant items in config and 
    analyze graph dictionaries.
    '''
    import graph_tool as glib
    from graph_tool import Graph as GraphLib
    config["graph_library"] = "graph-tool"
    config["library"] = glib
    config["graph"] = GraphLib
    # analysis functions
    from graph_tool.spectral import adjacency as _adj
    from graph_tool.centrality import betweenness
    from graph_tool.correlations import assortativity as assort
    from graph_tool.topology import (edge_reciprocity,
                                    label_components, pseudo_diameter)
    from graph_tool.clustering import global_clustering
    # defining the adjacency function
    def adj_mat(graph, weight=None):
        if weight is not None:
            weight = graph.edge_properties[weight]
        return _adj(graph, weight).T
    def get_edges(graph):
        return graph.edges()
    # store the functions
    analyze_graph["assortativity"] = assort
    analyze_graph["betweenness"] = betweenness
    analyze_graph["clustering"] = global_clustering
    analyze_graph["scc"] = label_components
    analyze_graph["wcc"] = label_components
    analyze_graph["diameter"] = pseudo_diameter
    analyze_graph["reciprocity"] = edge_reciprocity
    analyze_graph["adjacency"] = adj_mat
    analyze_graph["get_edges"] = get_edges


def _set_igraph():
    '''
    Set igraph as graph library, store relevant items in config and 
    analyze graph dictionaries.
    '''
    import igraph as glib
    from igraph import Graph as GraphLib
    config["graph_library"] = "igraph"
    config["library"] = glib
    config["graph"] = GraphLib
    # defining the adjacency function
    def adj_mat(graph, weight=None):
        n = graph.node_nb()
        if graph.edge_nb():
            xs, ys = map(sp.array, zip(*graph.get_edgelist()))
            xs, ys = xs.T, ys.T
            data = sp.ones(xs.shape)
            if issubclass(weight.__class__, str):
                data *= sp.array(graph.es[weight])
            else:
                data *= sp.array(weight)
            coo_adj = ssp.coo_matrix((data, (xs, ys)), shape=(n,n))
            return coo_adj.tocsr()
        else:
            return ssp.csr_matrix((n,n))
    def get_edges(graph):
        return graph.get_edgelist()
    # store functions
    analyze_graph["assortativity"] = not_implemented
    analyze_graph["nbetweenness"] = not_implemented
    analyze_graph["ebetweenness"] = not_implemented
    analyze_graph["clustering"] = not_implemented
    analyze_graph["scc"] = not_implemented
    analyze_graph["wcc"] = not_implemented
    analyze_graph["diameter"] = not_implemented
    analyze_graph["reciprocity"] = not_implemented
    analyze_graph["adjacency"] = adj_mat
    analyze_graph["get_edges"] = get_edges


def _set_networkx():
    import networkx as glib
    from networkx import DiGraph as GraphLib
    config["graph_library"] = "networkx"
    config["library"] = glib
    config["graph"] = GraphLib
    # analysis functions
    from networkx.algorithms import ( diameter, 
        strongly_connected_components, weakly_connected_components,
        degree_assortativity_coefficient )
    def overall_reciprocity(g):
        num_edges = g.number_of_edges()
        num_recip = (num_edges - g.to_undirected().number_of_edges()) * 2
        if n_all_edge == 0:
            raise ArgumentError("Not defined for empty graphs")
        else:
            return num_recip/float(num_edges)
    nx_version = glib.__version__
    try:
        from networkx.algorithms import overall_reciprocity
    except ImportError:
        def overall_reciprocity(*args, **kwargs):
            return NotImplementedError("Not implemented for networkx {}; \
try to install latest version.".format(nx_version))
    # defining the adjacency function
    from networkx import to_scipy_sparse_matrix
    def adj_mat(graph, weight=None):
        return to_scipy_sparse_matrix(graph, weight=weight)
    def get_edges(graph):
        return graph.edges_iter(data=False)
    # store functions
    analyze_graph["assortativity"] = degree_assortativity_coefficient
    analyze_graph["diameter"] = diameter
    analyze_graph["reciprocity"] = overall_reciprocity
    analyze_graph["scc"] = strongly_connected_components
    analyze_graph["wcc"] = diameter
    analyze_graph["adjacency"] = adj_mat
    analyze_graph["get_edges"] = get_edges



def use_library(library, reloading=True):
    '''
    Allows the user to switch to a specific graph library.
    
    .. warning:
        If :class:`~nngt.Graph` objects have already been created, they will no
        longer be compatible with NNGT methods.

    Parameters
    ----------
    library : string
        Name of a graph library among 'graph_tool', 'igraph', 'networkx'.
    reload_moduleing : bool, optional (default: True)
        Whether the graph objects should be reload_moduleed (this should always be set
        to True except when NNGT is first initiated!)
    '''
    if library == "graph-tool":
        _set_graph_tool()
    elif library == "igraph":
        _set_igraph()
    elif library == "networkx":
        _set_networkx()
    else:
        raise ValueError("Invalid graph library requested.")
    if reloading:
        sys.modules["nngt"].config = config
        sys.modules["nngt"].analyze_graph = analyze_graph
        reload_module(sys.modules["nngt"].core.base_graph)
        reload_module(sys.modules["nngt"].core.gt_graph)
        reload_module(sys.modules["nngt"].core.ig_graph)
        reload_module(sys.modules["nngt"].core.nx_graph)
        reload_module(sys.modules["nngt"].core)
        reload_module(sys.modules["nngt"].analysis)
        reload_module(sys.modules["nngt"].analysis.gt_analysis)
        reload_module(sys.modules["nngt"].generation)
        reload_module(sys.modules["nngt"].generation.graph_connectivity)
        if config['with_plot']:
            reload_module(sys.modules["nngt"].plot)
        if config['with_nest']:
            reload_module(sys.modules["nngt"].simulation)
        reload_module(sys.modules["nngt"].lib)
        reload_module(sys.modules["nngt"].core.graph_classes)
        from nngt.core.graph_classes import (Graph, SpatialGraph, Network,
                                             SpatialNetwork)
        sys.modules["nngt"].Graph = Graph
        sys.modules["nngt"].SpatialGraph = SpatialGraph
        sys.modules["nngt"].Network = Network
        sys.modules["nngt"].SpatialNetwork = SpatialNetwork


#-----------------------------------------------------------------------------#
# Loading graph library
#------------------------
#

_libs = [ 'graph-tool', 'igraph', 'networkx' ]

try:
    use_library(config['graph_library'], False)
except ImportError:
    idx = _libs.index(config['graph_library'])
    del _libs[idx]
    keep_trying = True
    while _libs and keep_trying:
        try:
            use_library(_libs[-1], False)
            keep_trying = False
        except ImportError:
            _libs.pop()

if not _libs:
    raise ImportError("This module needs one of the following graph libraries \
to work:  `graph_tool`, `igraph`, or `networkx`.")


#-----------------------------------------------------------------------------#
# Names
#------------------------
#

POS = "position"
DIST = "distance"
WEIGHT = "weight"
BWEIGHT = "bweight"
DELAY = "delay"
TYPE = "type"


#-----------------------------------------------------------------------------#
# Basic values
#------------------------
#

default_neuron = "aeif_cond_alpha"
''' :class:`string`, the default NEST neuron model '''
default_synapse = "static_synapse"
''' :class:`string`, the default NEST synaptic model '''
default_delay = 1.
''' :class:`double`, the default synaptic delay in NEST '''
