"""
Core classes and functions. Most of them are not visible in the module as they
are directly loaded at :mod:`nngt` level.

Content
=======
"""

import nngt.globals
from .gt_graph import _GtGraph
from .ig_graph import _IGraph
from .nx_graph import _NxGraph
from .graph_datastruct import Connections, NeuralGroup



di_graphlib = {
    "graph-tool": _GtGraph,
    "igraph": _IGraph,
    "networkx": _NxGraph,
    #~ # "snap": _SnapGraph
}


#: Graph object (reference to one of the main libraries' wrapper
GraphObject = di_graphlib[nngt.globals.config["graph_library"]]
#~ GraphObject = None


__all__ = [
    "Connections",
	"GraphObject",
    "NeuralGroup",
]
