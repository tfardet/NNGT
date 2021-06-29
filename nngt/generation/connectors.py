#-*- coding:utf-8 -*-
#
# generation/connectors.py
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

""" Connector functions """

import numpy as np

import nngt
from nngt.generation import graph_connectivity as gc
from nngt.lib import is_iterable, nonstring_container
from nngt.lib.test_functions import deprecated
from nngt.lib.rng_tools import _generate_random


__all__ = [
    'connect_neural_groups',
    'connect_groups',
    'connect_neural_types',
    'connect_nodes'
]


# generator dictionary
_di_gen_edges = {
    "all_to_all": gc._all_to_all,
    "circular": gc._circular,
    "distance_rule": gc._distance_rule,
    "erdos_renyi": gc._erdos_renyi,
    "fixed_degree": gc._fixed_degree,
    "from_degree_list": gc._from_degree_list,
    "gaussian_degree": gc._gaussian_degree,
    "newman_watts": gc._newman_watts,
    "watts_strogatz": gc._watts_strogatz,
    "price_scale_free": gc._price_scale_free,
    "random_scale_free": gc._random_scale_free
}


_one_pop_models = {
    "circular", "newman_watts", "price_scale_free", "watts_strogatz",
}


def connect_nodes(network, sources, targets, graph_model, density=None,
                  edges=None, avg_deg=None, unit='um', weighted=True,
                  directed=True, multigraph=False, check_existing=True,
                  ignore_invalid=False, **kwargs):
    '''
    Function to connect nodes with a given graph model.

    .. versionchanged:: 2.0
        Added `check_existing` and `ignore_invalid` arguments.

    Parameters
    ----------
    network : :class:`Network` or :class:`SpatialNetwork`
        The network to connect.
    sources : list
        Ids of the source nodes.
    targets : list
        Ids of the target nodes.
    graph_model : string
        The name of the connectivity model (among "erdos_renyi",
        "random_scale_free", "price_scale_free", and "newman_watts").
    check_existing : bool, optional (default: True)
        Check whether some of the edges that will be added already exist in the
        graph.
    ignore_invalid : bool, optional (default: False)
        Ignore invalid edges: they are not added to the graph and are
        silently dropped. Unless this is set to true, an error is raised
        if an existing edge is re-generated.
    **kwargs : keyword arguments
        Specific model parameters. or edge attributes specifiers such as
        `weights` or `delays`.

    Note
    ----
    For graph generation methods which set the properties of a
    specific degree (e.g. :func:`~nngt.generation.gaussian_degree`), the
    nodes which have their property sets are the `sources`.
    '''
    if network.is_spatial() and 'positions' not in kwargs:
        kwargs['positions'] = network.get_positions().astype(np.float32).T
    if network.is_spatial() and 'shape' not in kwargs:
        kwargs['shape'] = network.shape

    if graph_model in _one_pop_models:
        assert np.array_equal(sources, targets), \
            "'" + graph_model + "' can only work on a single set of nodes."

    sources  = np.array(sources, dtype=np.uint)
    targets  = np.array(targets, dtype=np.uint)
    distance = []

    elist = _di_gen_edges[graph_model](
        sources, targets, density=density, edges=edges,
        avg_deg=avg_deg, weighted=weighted, directed=directed,
        multigraph=multigraph, distance=distance, **kwargs)

    # Attributes are not set by subfunctions
    attr = {}

    if 'weights' in kwargs:
        ww = kwargs['weights']

        if isinstance(ww, dict):
            attr['weight'] = _generate_random(len(elist), ww)
        elif nonstring_container(ww):
            attr['weight'] = ww
        else:
            attr['weight'] = np.full(len(elist), ww)

    if 'delays' in kwargs:
        dd = kwargs['delays']

        if isinstance(ww, dict):
            attr['delay'] = _generate_random(len(elist), dd)
        elif nonstring_container(dd):
            attr['weight'] = dd
        else:
            attr['weight'] = np.full(len(elist), dd)

    if network.is_spatial() and distance:
        attr['distance'] = distance

    # call only on root process (for mpi) unless using distributed backend
    if nngt.on_master_process() or nngt.get_config("backend") == "nngt":
        elist = network.new_edges(
            elist, attributes=attr, check_duplicates=False,
            check_self_loops=False, check_existing=check_existing,
            ignore_invalid=ignore_invalid)

    if not network._graph_type.endswith('_connect'):
        network._graph_type += "_nodes_connect"

    return elist


def connect_neural_types(network, source_type, target_type, graph_model,
                         density=None, edges=None, avg_deg=None, unit='um',
                         weighted=True, directed=True, multigraph=False,
                         check_existing=True, ignore_invalid=False, **kwargs):
    '''
    Function to connect excitatory and inhibitory population with a given graph
    model.

    .. versionchanged:: 2.0
        Added `check_existing` and `ignore_invalid` arguments.

    Parameters
    ----------
    network : :class:`Network` or :class:`SpatialNetwork`
        The network to connect.
    source_type : int or list
        The type of source neurons (``1`` for excitatory, ``-1`` for
        inhibitory neurons).
    target_type : int or list
        The type of target neurons.
    graph_model : string
        The name of the connectivity model (among "erdos_renyi",
        "random_scale_free", "price_scale_free", and "newman_watts").
    check_existing : bool, optional (default: True)
        Check whether some of the edges that will be added already exist in the
        graph.
    ignore_invalid : bool, optional (default: False)
        Ignore invalid edges: they are not added to the graph and are
        silently dropped. Unless this is set to true, an error is raised
        if an existing edge is re-generated.
    kwargs : keyword arguments
        Specific model parameters. or edge attributes specifiers such as
        `weights` or `delays`.

    Note
    ----
    For graph generation methods which set the properties of a
    specific degree (e.g. :func:`~nngt.generation.gaussian_degree`), the
    nodes which have their property sets are the `source_type`.
    '''
    assert network.is_network(), "This function requires a Network object."

    elist, source_ids, target_ids = None, [], []

    if network.is_spatial() and 'positions' not in kwargs:
        kwargs['positions'] = network.get_positions().astype(np.float32).T

    if network.is_spatial() and 'shape' not in kwargs:
        kwargs['shape'] = network.shape

    if not nonstring_container(source_type):
        source_type = [source_type]

    if not nonstring_container(target_type):
        target_type = [target_type]

    for group in network._population.values():
        if group.neuron_type in source_type:
            source_ids.extend(group.ids)

        if group.neuron_type in target_type:
            target_ids.extend(group.ids)

    source_ids = np.array(source_ids, dtype=np.uint)
    target_ids = np.array(target_ids, dtype=np.uint)

    elist = connect_nodes(
        network, source_ids, target_ids, graph_model, density=density,
        edges=edges, avg_deg=avg_deg, unit=unit, weighted=weighted,
        directed=directed, multigraph=multigraph,
        check_existing=check_existing, ignore_invalid=ignore_invalid,
        **kwargs)

    if not network._graph_type.endswith('_neural_type_connect'):
        network._graph_type += "_neural_type_connect"

    return elist


@deprecated("1.3.1", reason="the library is moving to more generic names",
            alternative="connect_groups", removal="3.0")
def connect_neural_groups(*args, **kwargs):
    ''' Deprecatd alias of :func:`connect_groups`. '''
    return connect_groups(*args, **kwargs)


def connect_groups(network, source_groups, target_groups, graph_model,
                   density=None, edges=None, avg_deg=None, unit='um',
                   weighted=True, directed=True, multigraph=False,
                   check_existing=True, ignore_invalid=False, **kwargs):
    '''
    Function to connect groups with a given graph model.

    .. versionchanged:: 2.0
        Added `check_existing` and `ignore_invalid` arguments.

    Parameters
    ----------
    network : :class:`Network` or :class:`SpatialNetwork`
        The network to connect.
    source_groups : str, :class:`NeuralGroup`, or iterable
        Names of the source groups (which contain the pre-synaptic neurons) or
        directly the group objects themselves.
    target_groups : str, :class:`NeuralGroup`, or iterable
        Names of the target groups (which contain the post-synaptic neurons) or
        directly the group objects themselves.
    graph_model : string
        The name of the connectivity model (among "erdos_renyi",
        "random_scale_free", "price_scale_free", and "newman_watts").
    check_existing : bool, optional (default: True)
        Check whether some of the edges that will be added already exist in the
        graph.
    ignore_invalid : bool, optional (default: False)
        Ignore invalid edges: they are not added to the graph and are
        silently dropped. Unless this is set to true, an error is raised
        if an existing edge is re-generated.
    kwargs : keyword arguments
        Specific model parameters. or edge attributes specifiers such as
        `weights` or `delays`.

    Note
    ----
    For graph generation methods which set the properties of a
    specific degree (e.g. :func:`~nngt.generation.gaussian_degree`), the
    groups which have their property sets are the `source_groups`.
    '''
    source_ids, target_ids = [], []

    if network.is_spatial():
        if 'positions' not in kwargs:
            kwargs['positions'] = network.get_positions().astype(np.float32).T
        if 'shape' not in kwargs:
            kwargs['shape'] = network.shape

    if isinstance(source_groups, str) or not is_iterable(source_groups):
        source_groups = [source_groups]
    if isinstance(target_groups, str) or not is_iterable(target_groups):
        target_groups = [target_groups]

    for s in source_groups:
        if isinstance(s, nngt.Structure):
            source_ids.extend(s.ids)
        elif isinstance(s, nngt.Group):
            source_ids.extend(s.ids)
        else:
            source_ids.extend(network.structure[s].ids)

    for t in target_groups:
        if isinstance(t, nngt.Structure):
            target_ids.extend(t.ids)
        elif isinstance(t, nngt.Group):
            target_ids.extend(t.ids)
        else:
            target_ids.extend(network.structure[t].ids)

    source_ids = np.array(source_ids, dtype=np.uint)
    target_ids = np.array(target_ids, dtype=np.uint)

    elist = connect_nodes(
        network, source_ids, target_ids, graph_model, density=density,
        edges=edges, avg_deg=avg_deg, unit=unit, weighted=weighted,
        directed=directed, multigraph=multigraph,
        check_existing=check_existing, ignore_invalid=ignore_invalid,
        **kwargs)

    if not network._graph_type.endswith('_neural_group_connect'):
        network._graph_type += "_neural_group_connect"

    return elist
