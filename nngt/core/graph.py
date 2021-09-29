#-*- coding:utf-8 -*-
#
# core/graph.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Graph class for graph generation and management """

import logging
import weakref

from collections import defaultdict
from copy import deepcopy

import numpy as np
import scipy.sparse as ssp

import nngt
import nngt.analysis as na

from nngt import save_to_file
from nngt.io.graph_loading import _load_from_file, _library_load, di_get_edges
from nngt.io.io_helpers import _get_format
from nngt.io.graph_saving import _as_string
from nngt.lib import InvalidArgument, nonstring_container
from nngt.lib.connect_tools import _set_degree_type, _unique_rows
from nngt.lib.graph_helpers import _edge_prop, _get_matrices
from nngt.lib.logger import _log_message
from nngt.lib.test_functions import graph_tool_check, is_integer

from .connections import Connections


logger = logging.getLogger(__name__)


# ----- #
# Graph #
# ----- #

class Graph(nngt.core.GraphObject):

    """
    The basic graph class, which inherits from a library class such as
    :class:`graph_tool.Graph`, :class:`networkx.DiGraph`, or ``igraph.Graph``.

    The objects provides several functions to easily access some basic
    properties.
    """

    #-------------------------------------------------------------------------#
    # Class properties

    __num_graphs = 0
    __max_id = 0

    @classmethod
    def num_graphs(cls):
        ''' Returns the number of alive instances. '''
        return cls.__num_graphs

    @classmethod
    def from_library(cls, library_graph, name="ImportedGraph", weighted=True,
                     directed=True, **kwargs):
        '''
        Create a :class:`~nngt.Graph` by wrapping a graph object from one of
        the supported libraries.

        Parameters
        ----------
        library_graph : object
            Graph object from one of the supported libraries (graph-tool,
            igraph, networkx).
        name : str, optional (default: "ImportedGraph")
        **kwargs
            Other standard arguments (see :func:`~nngt.Graph.__init__`)
        '''
        graph = cls(name=name, weighted=False, **kwargs)

        graph._from_library_graph(library_graph, copy=False)

        return graph

    @classmethod
    def from_matrix(cls, matrix, weighted=True, directed=True, population=None,
                    shape=None, positions=None, name=None, **kwargs):
        '''
        Creates a :class:`~nngt.Graph` from a :mod:`scipy.sparse` matrix or
        a dense matrix.

        Parameters
        ----------
        matrix : :mod:`scipy.sparse` matrix or :class:`numpy.ndarray`
            Adjacency matrix.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        population : :class:`~nngt.NeuralPop`
            Population to associate to the new :class:`~nngt.Network`.
        shape : :class:`~nngt.geometry.Shape`, optional (default: None)
            Shape to associate to the new :class:`~nngt.SpatialGraph`.
        positions : (N, 2) array
            Positions, in a 2D space, of the N neurons.
        name : str, optional
            Graph name.

        Returns
        -------
        :class:`~nngt.Graph`
        '''
        mshape = matrix.shape

        graph_name = "FromYMatrix_Z"

        nodes = max(mshape[0], mshape[1])

        if issubclass(matrix.__class__, ssp.spmatrix):
            graph_name = graph_name.replace('Y', 'Sparse')
            if not directed:
                if mshape[0] != mshape[1] or not (matrix.T != matrix).nnz == 0:
                    raise InvalidArgument('Incompatible `directed=False` '
                                          'option provided for non symmetric '
                                          'matrix.')

                matrix = ssp.tril(matrix, format=matrix.format)
        else:
            graph_name = graph_name.replace('Y', 'Dense')
            if not directed:
                if mshape[0] != mshape[1] or not (matrix.T == matrix).all():
                    raise InvalidArgument('Incompatible `directed=False` '
                                          'option provided for non symmetric '
                                          'matrix.')
                matrix = np.tril(matrix)

        edges = np.array(matrix.nonzero()).T

        graph_name = graph_name.replace("Z", str(cls.__num_graphs))

        # overwrite default name if necessary
        if name is not None:
            graph_name = name

        graph = cls(nodes, name=graph_name, weighted=weighted,
                    directed=directed, **kwargs)

        if population is not None:
            cls.make_network(graph, population)

        if shape is not None or positions is not None:
            cls.make_spatial(graph, shape, positions)

        weights = None

        if weighted:
            if issubclass(matrix.__class__, ssp.spmatrix):
                weights = np.array(matrix[edges[:, 0], edges[:, 1]])[0]
            else:
                weights = matrix[edges[:, 0], edges[:, 1]]

                if len(weights.shape) == 2:
                    weights = weights.A1

        attributes = {"weight": weights} if weighted else None

        graph.new_edges(edges, attributes, check_self_loops=False,
                        ignore_invalid=True)

        return graph

    @staticmethod
    def from_file(filename, fmt="auto", separator=" ", secondary=";",
                  attributes=None, attributes_types=None, notifier="@",
                  ignore="#", from_string=False, name=None,
                  directed=True, cleanup=False):
        '''
        Import a saved graph from a file.

        .. versionchanged :: 2.0
            Added optional `attributes_types` and `cleanup` arguments.

        Parameters
        ----------
        filename: str
            The path to the file.
        fmt : str, optional (default: deduced from filename)
            The format used to save the graph. Supported formats are:
            "neighbour" (neighbour list), "ssp" (scipy.sparse), "edge_list"
            (list of all the edges in the graph, one edge per line,
            represented by a ``source target``-pair), "gml" (gml format,
            default if `filename` ends with '.gml'), "graphml" (graphml format,
            default if `filename` ends with '.graphml' or '.xml'), "dot" (dot
            format, default if `filename` ends with '.dot'), "gt" (only
            when using `graph_tool <http://graph-tool.skewed.de/>`_ as library,
            detected if `filename` ends with '.gt').
        separator : str, optional (default " ")
            separator used to separate inputs in the case of custom formats
            (namely "neighbour" and "edge_list")
        secondary : str, optional (default: ";")
            Secondary separator used to separate attributes in the case of
            custom formats.
        attributes : list, optional (default: [])
            List of names for the attributes present in the file. If a
            `notifier` is present in the file, names will be deduced from it;
            otherwise the attributes will be numbered.
            For "edge_list", attributes may also be present as additional
            columns after the source and the target.
        attributes_types : dict, optional (default: str)
            Backup information if the type of the attributes is not specified
            in the file. Values must be callables (types or functions) that
            will take the argument value as a string input and convert it to
            the proper type.
        notifier : str, optional (default: "@")
            Symbol specifying the following as meaningfull information.
            Relevant information are formatted ``@info_name=info_value``, where
            ``info_name`` is in ("attributes", "directed", "name", "size") and
            associated ``info_value`` are of type (``list``, ``bool``, ``str``,
            ``int``).
            Additional notifiers are
            ``@type=SpatialGraph/Network/SpatialNetwork``, which must be
            followed by the relevant notifiers among ``@shape``,
            ``@population``, and ``@graph``.
        from_string : bool, optional (default: False)
            Load from a string instead of a file.
        ignore : str, optional (default: "#")
            Ignore lines starting with the `ignore` string.
        name : str, optional (default: from file information or 'LoadedGraph')
            The name of the graph.
        directed : bool, optional (default: from file information or True)
            Whether the graph is directed or not.
        cleanup : bool, optional (default: False)
           If true, removes nodes before the first one that appears in the
           edges and after the last one and renumber the nodes from 0.

        Returns
        -------
        graph : :class:`~nngt.Graph` or subclass
            Loaded graph.
        '''
        fmt = _get_format(fmt, filename)

        if fmt not in di_get_edges:
            # only partial support for these formats, relying on backend
            libgraph = _library_load(filename, fmt)

            name = "LoadedGraph" if name is None else name

            graph = Graph.from_library(libgraph, name=name, directed=directed)

            return graph

        info, edges, nattr, eattr, struct, shape, pos = _load_from_file(
            filename=filename, fmt=fmt, separator=separator, ignore=ignore,
            secondary=secondary, attributes=attributes,
            attributes_types=attributes_types, notifier=notifier,
            cleanup=cleanup)

        # create the graph
        name = info.get("name", "LoadedGraph") if name is None else name

        graph = Graph(nodes=info["size"], name=name,
                      directed=info.get("directed", directed))

        # make the nodes attributes
        lst_attr, dtpes, lst_values = [], [], []

        if info["node_attributes"]:  # node attributes to add to the graph
            lst_attr   = info["node_attributes"]
            dtpes      = info["node_attr_types"]
            lst_values = [nattr[name] for name in info["node_attributes"]]

        for nattr, dtype, values in zip(lst_attr, dtpes, lst_values):
            graph.new_node_attribute(nattr, dtype, values=values)

        # make the edges and their attributes
        lst_attr, dtpes, lst_values = [], [], []

        if info["edge_attributes"]:  # edge attributes to add to the graph
            lst_attr   = info["edge_attributes"]
            dtpes      = info["edge_attr_types"]
            lst_values = [eattr[name] for name in info["edge_attributes"]]

        if len(edges):
            graph.new_edges(edges, check_duplicates=False,
                            check_self_loops=False, check_existing=False)

        for eattr, dtype, values in zip(lst_attr, dtpes, lst_values):
            graph.new_edge_attribute(eattr, dtype, values=values)

        if struct is not None:
            if isinstance(struct, nngt.NeuralPop):
                nngt.Network.make_network(graph, struct)
            else:
                graph.structure = struct

            struct._parent = weakref.ref(graph)

            for g in struct.values():
                g._struct = weakref.ref(struct)
                g._net    = weakref.ref(graph)

        if pos is not None or shape is not None:
            nngt.SpatialGraph.make_spatial(graph, shape=shape, positions=pos)

        return graph

    @staticmethod
    def make_spatial(graph, shape=None, positions=None, copy=False):
        '''
        Turn a :class:`~nngt.Graph` object into a :class:`~nngt.SpatialGraph`,
        or a :class:`~nngt.Network` into a :class:`~nngt.SpatialNetwork`.

        Parameters
        ----------
        graph : :class:`~nngt.Graph` or :class:`~nngt.SpatialGraph`
            Graph to convert.
        shape : :class:`~nngt.geometry.Shape`, optional (default: None)
            Shape to associate to the new :class:`~nngt.SpatialGraph`.
        positions : (N, 2) array
            Positions, in a 2D space, of the N neurons.
        copy : bool, optional (default: ``False``)
            Whether the operation should be made in-place on the object or if a
            new object should be returned.

        Notes
        -----
        In-place operation that directly converts the original graph if `copy`
        is ``False``, else returns the copied :class:`~nngt.Graph` turned into
        a :class:`~nngt.SpatialGraph`.
        The `shape` argument can be skipped if `positions` are given; in that
        case, the neurons will be embedded in a rectangle that contains them
        all.
        '''
        if copy:
            graph = graph.copy()

        if isinstance(graph, nngt.Network):
            graph.__class__ = nngt.SpatialNetwork
        else:
            graph.__class__ = nngt.SpatialGraph

        graph._init_spatial_properties(shape, positions)

        if copy:
            return graph

    @staticmethod
    def make_network(graph, neural_pop, copy=False, **kwargs):
        '''
        Turn a :class:`~nngt.Graph` object into a :class:`~nngt.Network`, or a
        :class:`~nngt.SpatialGraph` into a :class:`~nngt.SpatialNetwork`.

        Parameters
        ----------
        graph : :class:`~nngt.Graph` or :class:`~nngt.SpatialGraph`
            Graph to convert
        neural_pop : :class:`~nngt.NeuralPop`
            Population to associate to the new :class:`~nngt.Network`
        copy : bool, optional (default: ``False``)
            Whether the operation should be made in-place on the object or if a
            new object should be returned.

        Notes
        -----
        In-place operation that directly converts the original graph if `copy`
        is ``False``, else returns the copied :class:`~nngt.Graph` turned into
        a :class:`~nngt.Network`.
        '''
        if copy:
            graph = graph.copy()

        if isinstance(graph, nngt.SpatialGraph):
            graph.__class__ = nngt.SpatialNetwork
        else:
            graph.__class__ = nngt.Network

        # set delays to 1. or to provided value if they are not already set
        if "delays" not in kwargs and not hasattr(graph, '_d'):
            graph._d = {"distribution": "constant", "value": 1.}
        elif "delays" in kwargs and not hasattr(graph, '_d'):
            graph._d = kwargs["delays"]
        elif "delays" in kwargs:
            _log_message(logger, "WARNING",
                         'Graph already had delays set, ignoring new ones.')

        graph._init_bioproperties(neural_pop)

        if copy:
            return graph

    #-------------------------------------------------------------------------#
    # Constructor/destructor and properties

    def __new__(cls, *args, **kwargs):
        '''
        Create a new Graph object.
        '''
        has_pop = False
        is_sptl = False

        for arg in args:
            if isinstance(arg, nngt.geometry.Shape):
                is_sptl = True
            if isinstance(arg, nngt.NeuralPop):
                has_pop = True

        if "population" in kwargs:
            has_pop = True
        if kwargs.get("shape") is not None \
           or kwargs.get("positions") is not None:
            is_sptl = True

        if is_sptl and has_pop:
            cls = nngt.SpatialNetwork
        elif is_sptl:
            cls = nngt.SpatialGraph
        elif has_pop:
            cls = nngt.Network

        return super().__new__(cls)

    def __init__(self, nodes=None, name="Graph", weighted=True, directed=True,
                 copy_graph=None, structure=None, **kwargs):
        '''
        Initialize Graph instance

        .. versionchanged:: 2.0
            Renamed `from_graph` to `copy_graph`.

        .. versionchanged:: 2.2
            Added `structure` argument.

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`Graph` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        copy_graph : :class:`~nngt.Graph`, optional
            An optional :class:`~nngt.Graph` that will be copied.
        structure : :class:`~nngt.Structure`, optional (default: None)
            A structure dividing the graph into specific groups, which can
            be used to generate specific connectivities and visualise the
            connections in a more coarse-grained manner.
        kwargs : optional keywords arguments
            Optional arguments that can be passed to the graph, e.g. a dict
            containing information on the synaptic weights
            (``weights={"distribution": "constant", "value": 2.3}`` which is
            equivalent to ``weights=2.3``), the synaptic `delays`, or a
            ``type`` information.

        Note
        ----
        When using `copy_graph`, only the topological properties are
        copied (nodes, edges, and attributes), spatial and biological
        properties are ignored.
        To copy a graph exactly, use :func:`~nngt.Graph.copy`.

        Returns
        -------
        self : :class:`~nngt.Graph`
        '''
        self.__id = self.__class__.__max_id
        self._name = name
        self._graph_type = kwargs["type"] if "type" in kwargs else "custom"

        # check the structure
        if structure is not None:
            if nodes is None:
                nodes = structure.size
            else:
                assert nodes == structure.size, \
                    "`nodes` and `structure.size` must be the same."
        else:
            nodes = 0 if nodes is None else nodes

        self._struct = structure

        # Init the core.GraphObject
        super().__init__(nodes=nodes, copy_graph=copy_graph,
                         directed=directed, weighted=weighted)

        # take care of the weights and delays
        if copy_graph is None:
            if weighted:
                self.new_edge_attribute('weight', 'double')
                self._w = _edge_prop(kwargs.get("weights", None))
            if "delays" in kwargs:
                self.new_edge_attribute('delay', 'double')
                self._d = _edge_prop(kwargs.get("delays", None))
            if 'inh_weight_factor' in kwargs:
                self._iwf = kwargs['inh_weight_factor']
        else:
            self._w   = getattr(copy_graph, "_w", None)
            self._d   = getattr(copy_graph, "_d", None)
            self._iwf = getattr(copy_graph, "_iwf", None)

            self._eattr._num_values_set = \
                copy_graph._eattr._num_values_set.copy()

        # check kwargs
        kw_set = {"weights", "delays", "type", "inh_weight_factor"}

        remaining = set(kwargs) - kw_set

        for kw in remaining:
            if kwargs[kw] is not None:
                _log_message(logger, "WARNING", "Unused keyword argument '" +
                             kw + "'.")

        # update the counters
        self.__class__.__num_graphs += 1
        self.__class__.__max_id += 1

    def __del__(self):
        ''' Graph deletion (update graph count) '''
        self.__class__.__num_graphs -= 1

    def __repr__(self):
        ''' Provide unambiguous informations regarding the object. '''
        d = "directed" if self.is_directed() else "undirected"
        w = "weighted" if self.is_weighted() else "binary"
        t = self.type
        n = self.node_nb()
        e = self.edge_nb()

        return "<{directed}/{weighted} {obj} object of type '{net_type}' " \
               "with {nodes} nodes and {edges} edges at 0x{obj_id}>".format(
                    directed=d, weighted=w, obj=type(self).__name__,
                    net_type=t, nodes=n, edges=e, obj_id=id(self))

    def __str__(self):
        '''
        Return the full string description of the object as would be stored
        inside a file when saving the graph.
        '''
        return _as_string(self)

    @property
    def graph(self):
        '''
        Returns the underlying library object.

        .. warning ::
            Do not add or remove edges directly through this object.

        See also
        --------
        :ref:`graph_attr`
        :ref:`graph-analysis`.
        '''
        return self._graph

    @property
    def structure(self):
        '''
        Object structuring the graph into specific groups.

        .. versionadded: 2.2

        Note
        ----
        Points to :py:obj:`~nngt.Network.population` if the graph is a
        :class:`~nngt.Network`.
        '''
        if self.is_network():
            return self.population

        return self._struct

    @structure.setter
    def structure(self, structure):
        if self.is_network():
            self.population = structure
        else:
            if issubclass(structure.__class__, nngt.Structure):
                if self.node_nb() == structure.size:
                    if structure.is_valid:
                        self._struct = structure
                    else:
                        raise AttributeError(
                            "Structure is not valid (not all nodes are "
                            "associated to a group).")
                else:
                    raise AttributeError("Graph and Structure must have same "
                                         "number of nodes.")
            else:
                raise AttributeError(
                    "Expecting Structure but received '{}'.".format(
                        structure.__class__.__name__))

    @property
    def graph_id(self):
        ''' Unique :class:`int` identifying the instance. '''
        return self.__id

    @property
    def name(self):
        ''' Name of the graph. '''
        return self._name

    @property
    def type(self):
        ''' Type of the graph. '''
        return self._graph_type

    #-------------------------------------------------------------------------#
    # Graph actions

    def copy(self):
        '''
        Returns a deepcopy of the current :class:`~nngt.Graph`
        instance
        '''
        if nngt.get_config("mpi"):
            raise NotImplementedError("`copy` is not MPI-safe yet.")

        gc_instance = Graph(name=self._name + '_copy',
                            weighted=self.is_weighted(), copy_graph=self,
                            directed=self.is_directed())

        if self.is_spatial():
            nngt.SpatialGraph.make_spatial(
                gc_instance, shape=self.shape.copy(),
                positions=deepcopy(self._pos))

        if self.is_network():
            nngt.Network.make_network(gc_instance, self.population.copy())

        return gc_instance

    def to_file(self, filename, fmt="auto", separator=" ", secondary=";",
                attributes=None, notifier="@"):
        '''
        Save graph to file; options detailed below.

        See also
        --------
        :py:func:`nngt.lib.save_to_file` function for options.
        '''
        save_to_file(self, filename, fmt=fmt, separator=separator,
                     secondary=secondary, attributes=attributes,
                     notifier=notifier)

    #~ def inhibitory_subgraph(self):
        #~ ''' Create a :class:`~nngt.Graph` instance which graph
        #~ contains only the inhibitory edges of the current instance's
        #~ :class:`graph_tool.Graph` '''
        #~ eprop_b_type = self.new_edge_property(
                       #~ "bool",-self.edge_properties[TYPE].a+1)
        #~ self.set_edge_filter(eprop_b_type)
        #~ inhib_graph = Graph( name=self._name + '_inhib',
                             #~ weighted=self._weighted,
                             #~ from_graph=core.GraphObject(self.prune=True) )
        #~ self.clear_filters()
        #~ return inhib_graph

    #~ def excitatory_subgraph(self):
        #~ '''
        #~ Create a :class:`~nngt.Graph` instance which graph contains only the
        #~ excitatory edges of the current instance's :class:`core.GraphObject`.
        #~ .. warning ::
            #~ Only works for graph_tool
        #~ .. todo ::
            #~ Make this method library independant!
        #~ '''
        #~ eprop_b_type = self.new_edge_property(
                       #~ "bool",self.edge_properties[TYPE].a+1)
        #~ self.set_edge_filter(eprop_b_type)
        #~ exc_graph = Graph( name=self._name + '_exc',
                             #~ weighted=self._weighted,
                             #~ graph=core.GraphObject(self.prune=True) )
        #~ self.clear_filters()
        #~ return exc_graph

    def to_undirected(self, combine_numeric_eattr="sum"):
        '''
        Convert the graph to its undirected variant.

        .. note::
            All non-numeric edge attributes will be discarded from the returned
            undirected graph.

        Parameters
        ----------
        combine_numeric_eattr : str, optional (default: "sum")
            How to combine numeric attributes from reciprocal edges.
            Can be either:

            - "sum" (attributes are summed)
            - "min" (smallest value is kept)
            - "max" (largest value is kept)
            - "mean" (the average of both attributes is taken)

            In addition, `combine_numeric_eattr` can be a dictionary with one
            entry for each edge attribute.
        '''
        shape = self.shape if self.is_spatial() else None
        pos   = self.get_positions() if self.is_spatial() else None

        # Network cannot be undirected so convert NeuralPop to Structure and
        # Network to Graph if necessary
        structure = None

        if isinstance(self.structure, nngt.NeuralPop):
            structure = nngt.Structure.from_groups(self.structure)

        cls = nngt.SpatialGraph if isinstance(self, nngt.SpatialGraph) \
              else nngt.Graph

        g = cls(nodes=self.node_nb(), weighted=self.is_weighted(),
                shape=shape, positions=pos, directed=False,
                structure=structure)

        # replicate node attributes
        for nattr in self.node_attributes:
            g.new_node_attribute(nattr, self.get_attribute_type(nattr, "node"),
                                 self.node_attributes[nattr])

        # prepare edges
        eattrs = set(self.edge_attributes)

        # prepare combine method
        if isinstance(combine_numeric_eattr, str):
            val = str(combine_numeric_eattr)
            combine_numeric_eattr = defaultdict(lambda: val)
        elif isinstance(combine_numeric_eattr, dict):
            combine_numeric_eattr = defaultdict(
                lambda: "sum", **combine_numeric_eattr)

        # find integer eattr
        numeric_eattr = "weight" if "weight" in eattrs else None
        numeric_types = ("int", "double")

        if numeric_eattr is None:
            for eattr in eattrs:
                if self.get_attribute_type(eattr, "edge") in numeric_types:
                    numeric_eattr = eattr
                    break

        if numeric_eattr is not None:
            eattrs.discard(numeric_eattr)

            combine = combine_numeric_eattr[numeric_eattr]

            _, umat = _get_matrices(
                self, directed=False, weights=numeric_eattr, weighted=True,
                combine_weights=combine, remove_self_loops=False)

            umat = ssp.tril(umat, format="csr")

            # create the initial edge attribute
            g.new_edge_attribute(
                numeric_eattr, self.get_attribute_type(numeric_eattr, "edge"))

            indptr = umat.indptr

            diff = np.diff(indptr)
            keep = np.where(diff)[0]
            sources = np.repeat(keep, diff[keep])

            # make and add the edges and the first eattr
            edges = np.array((sources, umat.indices)).T

            g.new_edges(edges, attributes={"weight": umat.data},
                        check_self_loops=False)

            # add all other edge attributes
            for eattr in eattrs:
                etype = self.get_attribute_type(eattr, "edge")

                combine = combine_numeric_eattr[eattr]

                if etype in numeric_types:
                    if np.all(self.edge_attributes[eattr] > 0):
                        _, umat = _get_matrices(
                            self, directed=False, weights=eattr, weighted=True,
                            combine_weights=combine, remove_self_loops=False)

                        umat = ssp.tril(umat, format="csr")

                        g.new_edge_attribute(
                            eattr, self.get_attribute_type(eattr, "edge"),
                            values=umat.data)
                    else:
                        aa = list(self.edge_attributes[eattr])

                        adict = {
                            tuple(e): val
                            for e, val in zip(self.edges_array, aa)
                        }

                        f = None

                        if combine == "max":
                            f = np.max
                        elif combine == "min":
                            f = np.min
                        elif combine == "mean":
                            f = np.mean
                        elif combine == "sum":
                            f = np.sum
                        else:
                            raise ValueError(
                                "Invalid combination mode '{}'.".format(
                                    combine))

                        values = [
                            f([adict[e] for e in {tuple(e0), tuple(e0[::-1])}
                               if e in adict]) for e0 in g.edges_array
                        ]

                        g.new_edge_attribute(
                            eattr, self.get_attribute_type(eattr, "edge"),
                            values=values)
        else:
            # hide existing edge warning
            from nngt.lib.connect_tools import logger as lg

            old_loglevel = lg.level
            lg.setLevel(logging.ERROR)

            g.new_edges(self.edges_array, ignore_invalid=True)

            # restore previous logging level
            lg.setLevel(old_loglevel)

        return g


    def get_structure_graph(self):
        '''
        Return a coarse-grained version of the graph containing one node
        per :class:`nngt.Group`.
        Connections between groups are associated to the sum of all connection
        weights.
        If no structure is present, returns an empty Graph.
        '''
        struct = self.structure

        if struct is None:
            return Graph()

        names = list(struct.keys())
        nodes = len(struct)

        g = nngt.Graph(nodes,
                       name="Structure-graph of '{}'".format(self.name))

        eattr = {"weight": []}

        if self.is_network():
            eattr["delay"] = []

        new_edges = []

        for i, n1 in enumerate(names):
            g1 = struct[n1]

            for j, n2 in enumerate(names):
                g2 = struct[n2]

                edges = self.get_edges(source_node=g1.ids, target_node=g2.ids)

                if len(edges):
                    weights = self.get_weights(edges=edges)
                    w = np.sum(weights)
                    eattr["weight"].append(w)

                    if self.is_network():
                        delays  = self.get_delays(edges=edges)
                        d = np.average(delays)
                        eattr["delay"].append(d)

                    new_edges.append((i, j))

        # add edges and attributes
        if self.is_network():
            g.new_edge_attribute("delay", "double")

        g.new_edges(new_edges, attributes=eattr, check_self_loops=False)

        # set node attributes
        g.new_node_attribute("name", "string", values=names)

        return g

    #-------------------------------------------------------------------------#
    # Getters

    def adjacency_matrix(self, types=False, weights=False, mformat="csr"):
        '''
        Return the graph adjacency matrix.

        .. versionchanged: 2.0
            Added matrix format option (`mformat`).

        Note
        ----
        Source nodes are represented by the rows, targets by the
        corresponding columns.

        Parameters
        ----------
        types : bool, optional (default: False)
            Wether the edge types should be taken into account (negative values
            for inhibitory connections).
        weights : bool or string, optional (default: False)
            Whether the adjacecy matrix should be weighted. If True, all
            connections are multiply bythe associated synaptic strength; if
            weight is a string, the connections are scaled bythe corresponding
            edge attribute.
        mformat : str, optional (default: "csr")
            Type of :mod:`scipy.sparse` matrix that will be returned, by
            default :class:`scipy.sparse.csr_matrix`.

        Returns
        -------
        mat : :mod:`scipy.sparse` matrix
            The adjacency matrix of the graph.
        '''
        weights = "weight" if weights is True else weights

        mat = None

        if types:
            if self.is_network():
                # use inhibitory nodes
                mat = nngt.analyze_graph["adjacency"](self, weights)
                inh = self.population.inhibitory

                if np.any(inh):
                    mat[inh, :] *= -1

            elif 'type' in self.node_attributes:
                mat = nngt.analyze_graph["adjacency"](self, weights)
                tarray = np.where(self.node_attributes['type'] < 0)[0]
                if np.any(tarray):
                    mat[tarray] *= -1
            elif types and 'type' in self.edge_attributes:
                data = None

                if nonstring_container(weights):
                    data = weights
                elif weights in {None, False}:
                    data = np.ones(self.edge_nb())
                else:
                    data = self.get_edge_attributes(name=weights)

                data *= self.get_edge_attributes(name="type")

                edges     = self.edges_array
                num_nodes = self.node_nb()
                mat       = ssp.coo_matrix(
                    (data, (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes)).tocsr()

                if not self.is_directed():
                    mat += mat.T

            return mat.asformat(mformat)

        # untyped
        mat = nngt.analyze_graph["adjacency"](self, weights, mformat=mformat)

        return mat

    @property
    def node_attributes(self):
        '''
        Access node attributes.

        See also
        --------
        :attr:`~nngt.Graph.edge_attributes`,
        :attr:`~nngt.Graph.get_node_attributes`,
        :attr:`~nngt.Graph.new_node_attribute`,
        :attr:`~nngt.Graph.set_node_attribute`.
        '''
        return self._nattr

    @property
    def edge_attributes(self):
        '''
        Access edge attributes.

        See also
        --------
        :attr:`~nngt.Graph.node_attributes`,
        :attr:`~nngt.Graph.get_edge_attributes`,
        :attr:`~nngt.Graph.new_edge_attribute`,
        :attr:`~nngt.Graph.set_edge_attribute`.
        '''
        return self._eattr

    def get_nodes(self, attribute=None, value=None):
        '''
        Return the nodes in the network fulfilling a given condition.

        Parameters
        ----------
        attribute : str, optional (default: all nodes)
            Whether the `attribute` of the returned nodes should have a specific
            value.
        value : object, optional (default : None)
            If an `attribute` name is passed, then only nodes with `attribute`
            being equal to `value` will be returned.

        See also
        --------
        :func:`~nngt.Graph.get_edges`, :attr:`~nngt.Graph.node_attributes`
        '''
        if attribute is None:
            return [i for i in range(self.node_nb())]

        vtype = self._nattr.value_type(attribute)

        if value is None and vtype != "object":
            raise ValueError("`value` cannot be None for attribute '" +
                             attribute + "'.")

        return np.where(
            self.get_node_attributes(name=attribute) == value)[0]

    def get_edges(self, attribute=None, value=None, source_node=None,
                  target_node=None):
        '''
        Return the edges in the network fulfilling a given condition.

        For undirected graphs, edges are always returned in the order
        :math:`(u, v)` where :math:`u <= v`.

        .. warning ::
            Contrary to :func:`~nngt.Graph.edges_array` that returns edges
            ordered by creation time (i.e. corresponding to the order of the
            edge attribute array), this function does not enforce any specific
            edge order.
            This also  means that, if order does not matter, it may be faster
            to call ``get_edges`` that to call ``edges_array``.

        Parameters
        ----------
        attribute : str, optional (default: all nodes)
            Whether the `attribute` of the returned edges should have a
            specific value.
        value : object, optional (default : None)
            If an `attribute` name is passed, then only edges with `attribute`
            being equal to `value` will be returned.
        source_node : int or list of ints, optional (default: all nodes)
            Retrict the edges to those stemming from `source_node`.
        target_node : int or list of ints, optional (default: all nodes)
            Retrict the edges to those arriving at `target_node`.

        Returns
        -------
        A list of edges (2-tuples).

        See also
        --------
        :func:`~nngt.Graph.get_nodes`, :attr:`~nngt.Graph.edge_attributes`,
        :func:`~nngt.Graph.edges_array`
        '''
        edges = None

        if is_integer(source_node) and is_integer(target_node):
            # check that the edge exists, throw error otherwise
            self.edge_id((source_node, target_node))
            edges = [(source_node, target_node)]
        else:
            # backend-specific implementation for source or target
            edges = self._get_edges(source_node=source_node,
                                    target_node=target_node)

        # check attributes
        if attribute is None:
            return edges

        vtype = self._eattr.value_type(attribute)

        if value is None and vtype != "object":
            raise ValueError("`value` cannot be None for attribute '" +
                             attribute + "'.")

        desired = (self.get_edge_attributes(edges, attribute) == value)

        return [tuple(e) for e in self.edges_array[desired]]

    def get_edge_attributes(self, edges=None, name=None):
        '''
        Attributes of the graph's edges.

        Parameters
        ----------
        edges : tuple or list of tuples, optional (default: ``None``)
            Edge whose attribute should be displayed.
        name : str, optional (default: ``None``)
            Name of the desired attribute.

        Returns
        -------
        Dict containing all graph's attributes (synaptic weights, delays...)
        by default. If `edge` is specified, returns only the values for these
        edges. If `name` is specified, returns value of the attribute for each
        edge.

        Note
        ----
        The attributes values are ordered as the edges in
        :func:`~nngt.Graph.edges_array` if `edges` is None.

        See also
        --------
        :func:`~nngt.Graph.get_node_attributes`,
        :func:`~nngt.Graph.new_edge_attribute`,
        :func:`~nngt.Graph.set_edge_attribute`,
        :func:`~nngt.Graph.new_node_attribute`,
        :func:`~nngt.Graph.set_node_attribute`
        '''
        if name is not None and edges is not None:
            if len(edges):
                return self._eattr.get_eattr(edges=edges, name=name)

            return np.array([])
        elif name is None and edges is None:
            return {k: self._eattr[k]
                    for k in self._eattr.keys()}
        elif name is None:
            return self._eattr.get_eattr(edges=edges)

        return self._eattr[name]

    def get_node_attributes(self, nodes=None, name=None):
        '''
        Attributes of the graph's edges.

        .. versionchanged:: 1.0.1
            Corrected default behavior and made it the same as
            :func:`~nngt.Graph.get_edge_attributes`.

        .. versionadded:: 0.9

        Parameters
        ----------
        nodes : list of ints, optional (default: ``None``)
            Nodes whose attribute should be displayed.
        name : str, optional (default: ``None``)
            Name of the desired attribute.

        Returns
        -------
        Dict containing all nodes attributes by default. If `nodes` is
        specified, returns a ``dict`` containing only the attributes of these
        nodes. If `name` is specified, returns a list containing the values of
        the specific attribute for the required nodes (or all nodes if
        unspecified).

        See also
        --------
        :func:`~nngt.Graph.get_edge_attributes`,
        :func:`~nngt.Graph.new_node_attribute`,
        :func:`~nngt.Graph.set_node_attribute`,
        :func:`~nngt.Graph.new_edge_attributes`,
        :func:`~nngt.Graph.set_edge_attribute`
        '''
        res = None

        if name is None:
            res = {k: self._nattr[k] for k in self._nattr.keys()}
        else:
            res = self._nattr[name]

        if nodes is None:
            return res

        if isinstance(nodes, (slice, int)) or nonstring_container(nodes):
            if isinstance(res, dict):
                return {k: v[nodes] for k, v in res.items()}
            return res[nodes]
        else:
            raise ValueError("Invalid `nodes`: "
                             "{}, use slice, int, or list".format(nodes))

    def get_attribute_type(self, attribute_name, attribute_class=None):
        '''
        Return the type of an attribute (e.g. string, double, int).

        Parameters
        ----------
        attribute_name : str
            Name of the attribute.
        attribute_class : str, optional (default: both)
            Whether `attribute_name` is a "node" or an "edge" attribute.

        Returns
        -------
        type : str
            Type of the attribute.
        '''
        if attribute_class is None:
            is_eattr = attribute_name in self._eattr
            is_nattr = attribute_name in self._nattr
            if is_eattr and is_nattr:
                raise RuntimeError("Both edge and node attributes with name '"
                                   + attribute_name + "' exist, please "
                                   "specify `attribute_class`")
            elif is_eattr:
                return self._eattr.value_type(attribute_name)
            elif is_nattr:
                return self._nattr.value_type(attribute_name)
            else:
                raise KeyError("No '{}' attribute.".format(attribute_name))
        else:
            if attribute_class == "edge":
                return self._eattr.value_type(attribute_name)
            elif attribute_class == "node":
                return self._nattr.value_type(attribute_name)
            else:
                raise InvalidArgument(
                    "Unknown attribute class '{}'.".format(attribute_class))

    def get_density(self):
        '''
        Density of the graph: :math:`\\frac{E}{N^2}`, where `E` is the number
        of edges and `N` the number of nodes.
        '''
        return self.edge_nb() / self.node_nb()**2

    def is_weighted(self):
        ''' Whether the edges have weights '''
        return "weight" in self.edge_attributes

    def is_directed(self):
        ''' Whether the graph is directed or not '''
        return self._graph.is_directed()

    def is_connected(self, mode="strong"):
        '''
        Return whether the graph is connected.

        Parameters
        ----------
        mode : str, optional (default: "strong")
            Whether to test connectedness with directed ("strong") or
            undirected ("weak") connections.

        References
        ----------
        .. [ig-connected] :igdoc:`is_connected`
        '''
        return super().is_connected()

    def get_degrees(self, mode="total", nodes=None, weights=None,
                    edge_type="all"):
        '''
        Degree sequence of all the nodes.

        .. versionchanged:: 2.0
            Changed `deg_type` to `mode`, `node_list` to `nodes`, `use_weights`
            to `weights`, and `edge_type` to `edge_type`.

        Parameters
        ----------
        mode : string, optional (default: "total")
            Degree type (among 'in', 'out' or 'total').
        nodes : list, optional (default: None)
            List of the nodes which degree should be returned
        weights : bool or str, optional (default: binary edges)
            Whether edge weights should be considered; if ``None`` or ``False``
            then use binary edges; if ``True``, uses the 'weight' edge
            attribute, otherwise uses any valid edge attribute required.
        edge_type : int or str, optional (default: all)
            Restrict to a given synaptic type ("excitatory", 1, or
            "inhibitory", -1), using either the "type" edge attribute for
            non-:class:`~nngt.Network` or the
            :py:attr:`~nngt.NeuralPop.inhibitory` nodes.

        Returns
        -------
        degrees : :class:`numpy.array`
        
        .. warning ::
            When using MPI with "nngt" (distributed) backend, returns only the
            degrees associated to local edges. "Complete" degrees are obtained
            by taking the sum of the results on all MPI processes.
        '''
        mode = _set_degree_type(mode)

        if edge_type == "all":
            return super().get_degrees(
                mode=mode, nodes=nodes, weights=weights)
        elif edge_type in {"excitatory", 1}:
            edge_type = 1
        elif edge_type in {"inhibitory", -1}:
            edge_type = -1
        else:
            raise InvalidArgument(
                "Invalid edge type '{}'".format(edge_type))

        degrees = np.zeros(self.node_nb())

        if isinstance(self, nngt.Network):
            neurons = []
            for g in self.population.values():
                if g.neuron_type == edge_type:
                    neurons.extend(g.ids)

            if mode in {"in", "all"} or not self.is_directed():
                degrees += self.adjacency_matrix(
                    weights=weights,
                    types=False)[neurons, :].sum(axis=0).A1

            if mode in {"out", "all"} and self.is_directed():
                degrees += self.adjacency_matrix(
                    weights=weights,
                    types=False)[neurons, :].sum(axis=1).A1
        else:
            edges = np.where(
                self.get_edge_attributes(name="type") == edge_type)[0]

            w = None

            if weights is None:
                w = np.ones(len(edges))
            elif weights in self.edge_attributes:
                w = self.edge_attributes[weights]
            elif nonstring_container(weights):
                w = np.array(weights)
            else:
                raise InvalidArgument(
                    "Invalid `weights` '{}'".format(weights))

            # count in-degrees
            if mode in {"in", "all"} or not self.is_directed():
                np.add.at(degrees, edges[1], weights)

            if mode in {"out", "all"} and self.is_directed():
                np.add.at(degrees, edges[0], weights)

        if nodes is None:
            return degrees

        return degrees[nodes]

    def get_betweenness(self, btype="both", weights=None):
        '''
        Returns the normalized betweenness centrality of the nodes and edges.

        Parameters
        ----------
        g : :class:`~nngt.Graph`
            Graph to analyze.
        btype : str, optional (default 'both')
            The centrality that should be returned (either 'node', 'edge', or
            'both'). By default, both betweenness centralities are computed.
        weights : bool or str, optional (default: binary edges)
            Whether edge weights should be considered; if ``None`` or
            ``False`` then use binary edges; if ``True``, uses the 'weight'
            edge attribute, otherwise uses any valid edge attribute required.

        Returns
        -------
        nb : :class:`numpy.ndarray`
            The nodes' betweenness if `btype` is 'node' or 'both'
        eb : :class:`numpy.ndarray`
            The edges' betweenness if `btype` is 'edge' or 'both'

        See also
        --------
        :func:`~nngt.analysis.betweenness`
        '''
        from nngt.analysis import betweenness
        return betweenness(self, btype=btype, weights=weights)

    def get_edge_types(self, edges=None):
        '''
        Return the type of all or a subset of the edges.

        Parameters
        ----------
        edges : (E, 2) array, optional (default: all edges)
            Edges for which the type should be returned.

        Returns
        -------
        the list of types (1 for excitatory, -1 for inhibitory)
        '''
        if TYPE in self.edge_attributes:
            return self.get_edge_attributes(name=TYPE, edges=edges)
        else:
            size = self.edge_nb() if edges is None else len(edges)
            return np.ones(size)

    def get_weights(self, edges=None):
        '''
        Returns the weights of all or a subset of the edges.

        .. versionchanged:: 1.0.1
            Added the possibility to ask for a subset of edges.

        Parameters
        ----------
        edges : (E, 2) array, optional (default: all edges)
            Edges for which the type should be returned.

        Returns
        -------
        the list of weights
        '''
        if self.is_weighted():
            if edges is None:
                return self._eattr["weight"]

            if len(edges) == 0:
                return np.array([])

            return self._eattr.get_eattr(edges, "weight")

        size = self.edge_nb() if edges is None else len(edges)

        return np.ones(size)

    def get_delays(self, edges=None):
        '''
        Returns the delays of all or a subset of the edges.

        .. versionchanged:: 1.0.1
            Added the possibility to ask for a subset of edges.

        Parameters
        ----------
        edges : (E, 2) array, optional (default: all edges)
            Edges for which the type should be returned.

        Returns
        -------
        the list of delays
        '''
        if edges is None:
            return self._eattr["delay"]

        return self._eattr.get_eattr(edges, "delay")

    def neighbours(self, node, mode="all"):
        '''
        Return the neighbours of `node`.

        Parameters
        ----------
        node : int
            Index of the node of interest.
        mode : string, optional (default: "all")
            Type of neighbours that will be returned: "all" returns all the
            neighbours regardless of directionality, "in" returns the
            in-neighbours (also called predecessors) and "out" retruns the
            out-neighbours (or successors).

        Returns
        -------
        neighbours : set
            The neighbours of `node`.
        '''
        return super().neighbours(node, mode=mode)

    def is_spatial(self):
        '''
        Whether the graph is embedded in space (i.e. is a subclass of
        :class:`~nngt.SpatialGraph`).
        '''
        return issubclass(self.__class__, nngt.SpatialGraph)

    def is_network(self):
        '''
        Whether the graph is a subclass of :class:`~nngt.Network` (i.e. if it
        has a :class:`~nngt.NeuralPop` attribute).
        '''
        return issubclass(self.__class__, nngt.Network)

    #-------------------------------------------------------------------------#
    # Setters

    def set_name(self, name=None):
        ''' Set graph name '''
        if name is None:
            self._name = "Graph_" + str(self.__id)
        else:
            self._name = name

    def new_edge_attribute(self, name, value_type, values=None, val=None):
        '''
        Create a new attribute for the edges.

        Parameters
        ----------
        name : str
            The name of the new attribute.
        value_type : str
            Type of the attribute, among 'int', 'double', 'string', or 'object'
        values : array, optional (default: None)
            Values with which the edge attribute should be initialized.
            (must have one entry per node in the graph)
        val : int, float or str , optional (default: None)
            Identical value for all edges.
        '''
        assert name != "eid", "`eid` is a reserved internal edge-attribute."

        self._eattr.new_attribute(
            name, value_type, values=values, val=val)

    def new_node_attribute(self, name, value_type, values=None, val=None):
        '''
        Create a new attribute for the nodes.

        Parameters
        ----------
        name : str
            The name of the new attribute.
        value_type : str
            Type of the attribute, among 'int', 'double', 'string', or 'object'
        values : array, optional (default: None)
            Values with which the node attribute should be initialized.
            (must have one entry per node in the graph)
        val : int, float or str , optional (default: None)
            Identical value for all nodes.

        See also
        --------
        :func:`~nngt.Graph.new_edge_attribute`,
        :func:`~nngt.Graph.set_node_attribute`,
        :func:`~nngt.Graph.get_node_attributes`,
        :func:`~nngt.Graph.set_edge_attribute`,
        :func:`~nngt.Graph.get_edge_attributes`
        '''
        self._nattr.new_attribute(
            name, value_type, values=values, val=val)

    def set_edge_attribute(self, attribute, values=None, val=None,
                           value_type=None, edges=None):
        '''
        Set attributes to the connections between neurons.

        .. warning ::
            The special "type" attribute cannot be modified when using graphs
            that inherit from the :class:`~nngt.Network` class. This is because
            for biological networks, neurons make only one kind of synapse,
            which is determined by the :class:`nngt.NeuralGroup` they
            belong to.

        Parameters
        ----------
        attribute : str
            The name of the attribute.
        value_type : str
            Type of the attribute, among 'int', 'double', 'string'
        values : array, optional (default: None)
            Values with which the edge attribute should be initialized.
            (must have one entry per node in the graph)
        val : int, float or str , optional (default: None)
            Identical value for all edges.
        value_type : str, optional (default: None)
            Type of the attribute, among 'int', 'double', 'string'. Only used
            if the attribute does not exist and must be created.
        edges : list of edges or array of shape (E, 2), optional (default: all)
            Edges whose attributes should be set. Others will remain unchanged.

        See also
        --------
        :func:`~nngt.Graph.set_node_attribute`,
        :func:`~nngt.Graph.get_edge_attributes`,
        :func:`~nngt.Graph.new_edge_attribute`,
        :func:`~nngt.Graph.new_node_attribute`,
        :func:`~nngt.Graph.get_node_attributes`
        '''
        if attribute not in self.edge_attributes:
            assert value_type is not None, "`value_type` is necessary for " +\
                                           "new attributes."
            self.new_edge_attribute(name=attribute, value_type=value_type,
                                    values=values, val=val)
        else:
            num_edges = self.edge_nb() if edges is None else len(edges)
            if values is None:
                if val is not None:
                    values = [deepcopy(val) for _ in range(num_edges)]
                else:
                    raise InvalidArgument("At least one of the `values` and "
                        "`val` arguments should not be ``None``.")
            self._eattr.set_attribute(attribute, values, edges=edges)

    def set_node_attribute(self, attribute, values=None, val=None,
                           value_type=None, nodes=None):
        '''
        Set attributes to the connections between neurons.

        Parameters
        ----------
        attribute : str
            The name of the attribute.
        value_type : str
            Type of the attribute, among 'int', 'double', 'string'
        values : array, optional (default: None)
            Values with which the edge attribute should be initialized.
            (must have one entry per node in the graph)
        val : int, float or str , optional (default: None)
            Identical value for all edges.
        value_type : str, optional (default: None)
            Type of the attribute, among 'int', 'double', 'string'. Only used
            if the attribute does not exist and must be created.
        nodes : list of nodes, optional (default: all)
            Nodes whose attributes should be set. Others will remain unchanged.

        See also
        --------
        :func:`~nngt.Graph.set_edge_attribute`,
        :func:`~nngt.Graph.new_node_attribute`,
        :func:`~nngt.Graph.get_node_attributes`,
        :func:`~nngt.Graph.new_edge_attribute`,
        :func:`~nngt.Graph.get_edge_attributes`,
        '''
        if attribute not in self.node_attributes:
            assert value_type is not None, "`value_type` is necessary for " +\
                                           "new attributes."
            self.new_node_attribute(name=attribute, value_type=value_type,
                                    values=values, val=val)
        else:
            num_nodes = self.node_nb() if nodes is None else len(nodes)
            if values is None:
                if val is not None:
                    values = [deepcopy(val) for _ in  range(num_nodes)]
                else:
                    raise InvalidArgument("At least one of the `values` and "
                        "`val` arguments should not be ``None``.")
            self._nattr.set_attribute(attribute, values, nodes=nodes)

    def set_weights(self, weight=None, elist=None, distribution=None,
                    parameters=None, noise_scale=None):
        '''
        Set the synaptic weights.

        Parameters
        ----------
        weight : float or class:`numpy.array`, optional (default: None)
            Value or list of the weights (for user defined weights).
        elist : class:`numpy.array`, optional (default: None)
            List of the edges (for user defined weights).
        distribution : class:`string`, optional (default: None)
            Type of distribution (choose among "constant", "uniform",
            "gaussian", "lognormal", "lin_corr", "log_corr").
        parameters : dict, optional (default: {})
            Dictionary containing the properties of the weight distribution.
            Properties are as follow for the distributions

            - 'constant': 'value'
            - 'uniform': 'lower', 'upper'
            - 'gaussian': 'avg', 'std'
            - 'lognormal': 'position', 'scale'

        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.

        Note
        ----
        If `distribution` and `parameters` are provided and the weights are set
        for the whole graph (`elist` is None), then the distribution properties
        will be kept as the new default for subsequent edges. That is, if new
        edges are created without specifying their weights, then these new
        weights will automatically be drawn from this previous distribution.
        '''
        if isinstance(weight, float):
            size = self.edge_nb() if elist is None else len(elist)
            self._w = {"distribution": "constant", "value": weight}
            weight = np.repeat(weight, size)
        elif not nonstring_container(weight) and weight is not None:
            raise AttributeError("Invalid `weight` value: must be either "
                                 "float, array-like or None.")
        elif weight is not None:
            self._w = {"distribution": "custom"}
        elif None not in (distribution, parameters) and elist is None:
            self._w = {"distribution": distribution}
            self._w.update(parameters)

        if distribution is None:
            distribution = self._w.get("distribution", None)

        if parameters is None:
            parameters = self._w

        Connections.weights(
            self, elist=elist, wlist=weight, distribution=distribution,
            parameters=parameters, noise_scale=noise_scale)

    def set_types(self, edge_type, nodes=None, fraction=None):
        '''
        Set the synaptic/connection types.

        .. versionchanged :: 2.0
            Changed `syn_type` to `edge_type`.

        .. warning ::
            The special "type" attribute cannot be modified when using graphs
            that inherit from the :class:`~nngt.Network` class. This is because
            for biological networks, neurons make only one kind of synapse,
            which is determined by the :class:`nngt.NeuralGroup` they
            belong to.

        Parameters
        ----------
        edge_type : int, string, or array of ints
            Type of the connection among 'excitatory' (also `1`) or
            'inhibitory' (also `-1`).
        nodes : int, float or list, optional (default: `None`)
            If `nodes` is an int, number of nodes of the required type that
            will be created in the graph (all connections from inhibitory nodes
            are inhibitory); if it is a float, ratio of `edge_type` nodes in the
            graph; if it is a list, ids of the `edge_type` nodes.
        fraction : float, optional (default: `None`)
            Fraction of the selected edges that will be set as `edge_type` (if
            `nodes` is not `None`, it is the fraction of the specified nodes'
            edges, otherwise it is the fraction of all edges in the graph).

        Returns
        -------
        t_list : :class:`numpy.ndarray`
            List of the types in an order that matches the `edges` attribute of
            the graph.
        '''
        inhib_nodes = None

        if nonstring_container(edge_type):
            return Connections.types(self, values=edge_type)
        elif edge_type in ('excitatory', 1):
            if is_integer(nodes):
                inhib_nodes = self.node_nb() - nodes
            elif nonstring_container(nodes):
                inhib_nodes = list(range(self.node_nb()))
                nodes.sort()
                for node in nodes[::-1]:
                    del inhib_nodes[node]
            elif nodes is not None:
                raise ValueError("`nodes` should be integer or array of ids.")
        elif edge_type in ('inhibitory', -1):
            if is_integer(nodes) or nonstring_container(nodes):
                inhib_nodes = nodes
            elif nodes is not None:
                raise ValueError("`nodes` should be integer or array of ids.")

        return Connections.types(self, inhib_nodes, fraction)

    def set_delays(self, delay=None, elist=None, distribution=None,
                   parameters=None, noise_scale=None):
        '''
        Set the delay for spike propagation between neurons.

        Parameters
        ----------
        delay : float or class:`numpy.array`, optional (default: None)
            Value or list of delays (for user defined delays).
        elist : class:`numpy.array`, optional (default: None)
            List of the edges (for user defined delays).
        distribution : class:`string`, optional (default: None)
            Type of distribution (choose among "constant", "uniform",
            "gaussian", "lognormal", "lin_corr", "log_corr").
        parameters : dict, optional (default: {})
            Dictionary containing the properties of the delay distribution.
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the delays.
        '''
        # check special cases and set self._d
        if isinstance(delay, float):
            size = self.edge_nb() if elist is None else len(elist)
            self._d = {"distribution": "constant", "value": delay}
            delay = np.repeat(delay, size)
        elif not nonstring_container(delay) and delay is not None:
            raise AttributeError("Invalid `delay` value: must be either "
                                 "float, array-like or None")
        elif delay is not None:
            self._d = {"distribution": "custom"}
        elif None not in (distribution, parameters):
            self._d = {"distribution": distribution}
            self._d.update(parameters)

        if distribution is None:
            if hasattr(self, "_d"):
                distribution = self._d["distribution"]
            else:
                raise AttributeError(
                    "Invalid `distribution` value: cannot be None if "
                    "default delays were not set at graph creation.")

        if parameters is None:
            if hasattr(self, "_d"):
                parameters = self._d
            else:
                raise AttributeError(
                    "Invalid `parameters` value: cannot be None if default"
                    " delays were not set at graph creation.")

        return Connections.delays(
            self, elist=elist, dlist=delay, distribution=distribution,
            parameters=parameters, noise_scale=noise_scale)
