#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Connectivity generators for Graph """

from .. import Graph, SpatialGraph, Network, Connections
from ..core import GraphObject
from ..lib.connect_tools import *



#
#---
# Erdos-Renyi
#------------------------

def erdos_renyi(nodes=0, density=0.1, edges=-1, avg_deg=-1., reciprocity=-1.,
                weighted=True, directed=True, multigraph=False, name="ER",
                shape=None, positions=None, population=None, from_graph=None):
    """
    Generate a random graph as defined by Erdos and Renyi but with a
    reciprocity that can be chosen.

    Parameters
    ----------
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    density : double, optional (default: 0.1)
        Structural density given by `edges` / `nodes`:math:`^2`.
    edges : int (optional)
        The number of edges between the nodes
    avg_deg : double, optional
        Average degree of the neurons given by `edges` / `nodes`.
    reciprocity : double, optional (default: -1 to let it free)
        Fraction of edges that are bidirectional (only for
        directed graphs -- undirected graphs have a reciprocity of 1 by
        definition)
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.core.Shape`, optional (default: None)
        Shape of the neurons' environment.
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_er : :class:`~nngt.Graph`, or subclass
        A new generated graph or the modified `from_graph`.

    Notes
    -----
    `nodes` is required unless `from_graph` or `population` is provided.
    If an `from_graph` is provided, all preexistant edges in the
    object will be deleted before the new connectivity is implemented.
    """
    # set node number and library graph
    graph_obj_er, graph_er = None, from_graph
    if graph_er is not None:
        nodes = graph_er.node_nb()
        graph_er.clear_edges()
        graph_obj_er = graph_er.graph
    else:
        nodes = population.size if population is not None else nodes
        graph_obj_er = GraphObject(nodes, directed=directed)
    # add edges
    ia_edges = None
    if nodes > 1:
        ids = range(nodes)
        ia_edges = _erdos_renyi(ids, ids, density, edges, avg_deg, reciprocity,
                                directed, multigraph)
        graph_obj_er.new_edges(ia_edges)
    # generate container
    if graph_er is None:
        graph_er = Graph(name=name, libgraph=graph_obj_er, data={
                'edges': ia_edges})
    else:
        graph_er.set_weights()
    # set options
    if issubclass(graph_er.__class__, Network):
        Connections.delays(graph_er)
    elif population is not None:
        Network.make_network(graph_er, population)
    if shape is not None:
        SpatialGraph.make_spatial(graph_er, shape, positions)
    return graph_er


#
#---
# Scale-free models
#------------------------

def random_scale_free(in_exp, out_exp, nodes=0, density=0.1, edges=-1,
                      avg_deg=-1, reciprocity=0., directed=True,
                      multigraph=False, name="RandomSF", shape=None, 
                      positions=None, population=None, from_graph=None):
    """
    Generate a free-scale graph of given reciprocity and otherwise
    devoid of correlations.

    Parameters 
    ----------
    in_exp : float
        Absolute value of the in-degree exponent :math:`\gamma_i`, such that
        :math:`p(k_i) \propto k_i^{-\gamma_i}
    out_exp : float
        Absolute value of the out-degree exponent :math:`\gamma_o`, such that
        :math:`p(k_o) \propto k_o^{-\gamma_o}
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    density: double, optional (default: 0.1)
        Structural density given by `edges` / (`nodes`*`nodes`).
    edges : int (optional)
        The number of edges between the nodes
    avg_deg : double, optional
        Average degree of the neurons given by `edges` / `nodes`.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes. can contain multiple edges between two
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.core.Shape`, optional (default: None)
        Shape of the neurons' environment.
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`)
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_fs : :class:`~nngt.Graph`
    
    Notes
    -----
    As reciprocity increases, requested values of `in_exp` and `out_exp` will
    be less and less respected as the distribution will converge to a common
    exponent :math:`\gamma = \frac{\gamma_i + \gamma_o}{2}`.
    Parameter `nodes` is required unless `from_graph` or `population` is
    provided.
    """
    # set node number and library graph
    graph_obj_rsf, graph_rsf = None, from_graph
    if graph_rsf is not None:
        nodes = graph_rsf.node_nb()
        graph_rsf.clear_edges()
        graph_obj_rsf = graph_rsf.graph
    else:
        nodes = population.size if population is not None else nodes
        graph_obj_rsf = GraphObject(nodes, directed=directed)
    # add edges
    ia_edges = None
    if nodes > 1:
        ids = range(nodes)
        ia_edges = _random_scale_free(ids, ids, in_exp, out_exp, density,
                          edges, avg_deg, reciprocity, directed, multigraph)
        graph_obj_rsf.new_edges(ia_edges)
    # generate container
    if graph_rsf is None:
        graph_rsf = Graph(name=name, libgraph=graph_obj_rsf, data={
                'edges': ia_edges})
    else:
        graph_rsf.set_weights()
    # set options
    if issubclass(graph_rsf.__class__, Network):
        Connections.delays(graph_rsf)
    elif population is not None:
        Network.make_network(graph_rsf, population)
    if shape is not None:
        SpatialGraph.make_spatial(graph_rsf, shape, positions)
    return graph_rsf

def price_scale_free(m, c=None, gamma=1, nodes=0, directed=True,
                     seed_graph=None, multigraph=False, name="PriceSF",
                     shape=None, positions=None, population=None,
                     from_graph=None, **kwargs):
    """
    @todo: make the algorithm.
    Generate a Price graph model (Barabasi-Albert if undirected).

    Parameters 
    ----------
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    m : int
        The number of edges each new node will make.
    c : double
        Constant added to the probability of a vertex receiving an edge.
    gamma : double
        Preferential attachment power.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.core.Shape`, optional (default: None)
        Shape of the neurons' environment
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space
    from_graph : :class:`~nngt.Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.
    
    Returns
    -------
    graph_price : :class:`~nngt.Graph` or subclass.
    
    Notes
    -----
    `nodes` is required unless `from_graph` or `population` is provided.
    """
    nodes = ( ( population.size if population is not None else nodes )
              if from_graph is None else from_graph.node_nb() )
    #~ c = c if c is not None else 0 if directed else 1
    
    graph_obj_price = GraphObject.to_graph_object(
                        price_network(nodes,m,c,gamma,directed,seed_graph))
    if from_graph is not None:
        from_graph.graph = graph_obj_price
    
    graph_price = (Graph(name=name, libgraph=graph_obj_price, data={
                'edges': ia_edges}) if from_graph is None else from_graph)
    
    if issubclass(graph_price.__class__, Network):
        Connections.delays(graph_price, ia_edges)
    elif population is not None:
        Network.make_network(graph_price, population)
    if shape is not None:
        make_spatial(graph_price, shape, positions)
    return graph_price

#
#---
# Small-world models
#------------------------

def newman_watts(coord_nb, proba_shortcut, nodes=0, directed=True,
                 multigraph=False, name="ER", shape=None, positions=None,
                 population=None, from_graph=None, **kwargs):
    """
    Generate a small-world graph using the Newman-Watts algorithm.
    @todo: generate the edges of a circular graph to not replace the graph
    of the `from_graph` and implement chosen reciprocity.
    
    Parameters
    ----------
    coord_nb : int
        The number of neighbours for each node on the initial topological 
        lattice.
    proba_shortcut : double
        Probability of adding a new random (shortcut) edge for each existing 
        edge on the initial lattice.
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    density: double, optional (default: 0.1)
        Structural density given by `edges` / (`nodes`*`nodes`).
    edges : int (optional)
        The number of edges between the nodes
    avg_deg : double, optional
        Average degree of the neurons given by `edges` / `nodes`.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.core.Shape`, optional (default: None)
        Shape of the neurons' environment
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.
    
    Returns
    -------
    graph_nw : :class:`~nngt.Graph` or subclass
    
    Notes
    -----
    `nodes` is required unless `from_graph` or `population` is provided.
    """
    # set node number and library graph
    graph_obj_nw, graph_nw = None, from_graph
    if graph_nw is not None:
        nodes = graph_nw.node_nb()
        graph_nw.clear_edges()
        graph_obj_nw = graph_nw.graph
    else:
        nodes = population.size if population is not None else nodes
        graph_obj_nw = GraphObject(nodes, directed=directed)
    # add edges
    ia_edges = None
    if nodes > 1:
        ids = range(nodes)
        ia_edges = _newman_watts(ids, ids, coord_nb, proba_shortcut, directed,
                                 multigraph)
        graph_obj_nw.new_edges(ia_edges)
    # generate container
    if graph_nw is None:
        graph_nw = Graph(name=name, libgraph=graph_obj_nw, data={
                'edges': ia_edges})
    else:
        graph_nw.set_weights()
    # set options
    if issubclass(graph_nw.__class__, Network):
        Connections.delays(graph_nw)
    elif population is not None:
        Network.make_network(graph_nw, population)
    if shape is not None:
        SpatialGraph.make_spatial(graph_nw, shape, positions)
    return graph_nw


#
#---
# Distance-based models
#------------------------

def gen_edr(dicProperties):
    pass
    #~ np.random.seed()
    #~ # on définit toutes les grandeurs de base
    #~ rRho2D = dicProperties["Rho"]
    #~ rLambda = dicProperties["Lambda"]
    #~ nNodes = 0
    #~ nEdges = 0
    #~ rDens = 0.0
    #~ if "Nodes" in dicProperties.keys():
        #~ nNodes = dicProperties["Nodes"]
        #~ if "Edges" in dicProperties.keys():
            #~ nEdges = dicProperties["Edges"]
            #~ rDens = nEdges / float(nNodes**2)
            #~ dicProperties["Density"] = rDens
        #~ else:
            #~ rDens = dicProperties["Density"]
            #~ nEdges = int(np.floor(rDens*nNodes**2))
            #~ dicProperties["Edges"] = nEdges
    #~ else:
        #~ nEdges = dicProperties["Edges"]
        #~ rDens = dicProperties["Density"]
        #~ nNodes = int(np.floor(np.sqrt(nEdges/rDens)))
        #~ dicProperties["Nodes"] = nNodes
    #~ rSideLength = np.sqrt(nNodes/rRho2D)
    #~ rAverageDistance = np.sqrt(2)*rSideLength / 3
    #~ # generate the positions of the neurons
    #~ lstPos = np.array([np.random.uniform(0,rSideLength,nNodes),np.random.uniform(0,rSideLength,nNodes)])
    #~ lstPos = np.transpose(lstPos)
    #~ numDesiredEdges = int(float(rDens*nNodes**2))
    #~ graphEDR,pos = geometric_graph(lstPos,0)
    #~ graphEDR.set_directed(True)
    #~ graphEDR.vertex_properties["pos"] = pos
    #~ # test edges building on random neurons
    #~ nEdgesTot = graphEDR.edge_nb()
    #~ numTest = 0
    #~ while nEdgesTot < numDesiredEdges and numTest < n_MAXTESTS:
        #~ nTests = int(np.minimum(1.1*np.ceil(numDesiredEdges-nEdgesTot)*np.exp(np.divide(rAverageDistance,rLambda)),1e7))
        #~ lstVertSrc = np.random.randint(0,nNodes,nTests)
        #~ lstVertDest = np.random.randint(0,nNodes,nTests)
        #~ lstDist = np.linalg.norm(lstPos[lstVertDest]-lstPos[lstVertSrc],axis=1)
        #~ lstDist = np.exp(np.divide(lstDist,-rLambda))
        #~ lstCreateEdge = np.random.uniform(size=nTests)
        #~ lstCreateEdge = np.greater(lstDist,lstCreateEdge)
        #~ nEdges = np.sum(lstCreateEdge)
        #~ if nEdges+nEdgesTot > numDesiredEdges:
            #~ nEdges = numDesiredEdges - nEdgesTot
            #~ lstVertSrc = lstVertSrc[lstCreateEdge][:nEdges]
            #~ lstVertDest = lstVertDest[lstCreateEdge][:nEdges]
            #~ lstEdges = np.array([lstVertSrc,lstVertDest]).astype(int)
        #~ else:
            #~ lstEdges = np.array([lstVertSrc[lstCreateEdge],lstVertDest[lstCreateEdge]]).astype(int)
        #~ graphEDR.new_edges(np.transpose(lstEdges))
        #~ # make graph simple and connected
        #~ delete_self_loops(graphEDR)
        #~ delete_parallel_edges(graphEDR)
        #~ nEdgesTot = graphEDR.edge_nb()
        #~ numTest += 1
    #~ graphEDR.reindex_edges()
    #~ nNodes = graphEDR.node_nb()
    #~ nEdges = graphEDR.edge_nb()
    #~ rDens = nEdges / float(nNodes**2)
    #~ # generate types
    #~ rInhibFrac = dicProperties["InhibFrac"]
    #~ lstTypesGen = np.random.uniform(0,1,nEdges)
    #~ lstTypeLimit = np.full(nEdges,rInhibFrac)
    #~ lstIsExcitatory = np.greater(lstTypesGen,lstTypeLimit)
    #~ nExc = np.count_nonzero(lstIsExcitatory)
    #~ epropType = graphEDR.new_edge_property("int",np.multiply(2,lstIsExcitatory)-np.repeat(1,nEdges)) # excitatory (True) or inhibitory (False)
    #~ graphEDR.edge_properties["type"] = epropType
    #~ # and weights
    #~ if dicProperties["Weighted"]:
        #~ lstWeights = dicGenWeights[dicProperties["Distribution"]](graphEDR,dicProperties,nEdges,nExc) # generate the weights
        #~ epropW = graphEDR.new_edge_property("double",lstWeights) # crée la propriété pour stocker les poids
        #~ graphEDR.edge_properties["weight"] = epropW
    #~ return graphEDR


#-----------------------------------------------------------------------------#
# Connecting groups
#------------------------
#

di_gen_func = { "erdos_renyi": _erdos_renyi, 
    "random_scale_free": _random_scale_free,
    "price_scale_free": _price_scale_free,
    "newman_watts": _newman_watts }

di_default = {  "density": 0.1,
                "edges": -1,
                "avg_deg": -1,
                "reciprocity": -1,
                "directed": True,
                "multigraph": False }

one_pop_models = ("newman_watts",)

def connect_neural_types(network, source_type, target_type, graph_model,
                         model_param):
    '''
    Function to connect excitatory and inhibitory population with a given graph
    model.
    
    Parameters
    ----------
    network : :class:`Network` or :class:`SpatialNetwork`
        The network to connect.
    source_type : int
        The type of source neurons (``1`` for excitatory, ``-1 for 
        inhibitory neurons).
    source_type : int
        The type of target neurons.
    graph_model : string
        The name of the connectivity model (among "erdos_renyi", 
        "random_scale_free", "price_scale_free", and "newman_watts").
    model_param : dict
        Dictionary containing the model parameters (the keys are the keywords
        of the associated generation function --- see above).
    '''
    edges, source_ids, target_ids = None, [], []
    di_param = di_default.copy()
    di_param.update(model_param)
    for group in network._population.itervalues():
        if group.neuron_type == source_type:
            source_ids.extend(group._id_list)
        elif group.neuron_type == target_type:
            target_ids.extend(group._id_list)
    if source_type == target_type:
        edges = di_gen_func[graph_model](source_ids,source_ids,**di_param)
        network.add_edges(edges)
    else:
        edges = di_gen_func[graph_model](source_ids,target_ids,**di_param)
        network.add_edges(edges)
    if "edges" in network.attributes():
        network._data["edges"] = np.concatenate((network._data["edges"],edges))
    else:
        network._data["edges"] = edges
    network.set_weights(edges)
    Connections.delays(network, edges)
    if issubclass(network.__class__, SpatialGraph):
        Connections.distances(network, edges)

def connect_neural_groups(network, source_groups, target_groups, graph_model,
                         model_param):
    '''
    Function to connect excitatory and inhibitory population with a given graph
    model.
    
    Parameters
    ----------
    network : :class:`Network` or :class:`SpatialNetwork`
        The network to connect.
    source_groups : tuple of strings
        Names of the source groups (which contain the pre-synaptic neurons)
    target_groups : tuple of strings
        Names of the target groups (which contain the post-synaptic neurons)
    graph_model : string
        The name of the connectivity model (among "erdos_renyi", 
        "random_scale_free", "price_scale_free", and "newman_watts").
    model_param : dict
        Dictionary containing the model parameters (the keys are the keywords
        of the associated generation function --- see above).
    '''
    edges, source_ids, target_ids = None, [], []
    di_param = di_default.copy()
    di_param.update(model_param)
    for name, group in network._population.iteritems():
        if name in source_groups:
            source_ids.extend(group._id_list)
        elif name in target_groups:
            target_ids.extend(group._id_list)
    if source_groups == target_groups:
        edges = di_gen_func[graph_model](source_ids,source_ids,**di_param)
        network.add_edges(edges)
    else:
        edges = di_gen_func[graph_model](source_ids, target_ids, **di_param)
        network.add_edges(edges)
    if "edges" in network.attributes():
        network._data["edges"] = np.concatenate((network._data["edges"],edges))
    else:
        network._data["edges"] = edges
    network.set_weights(edges)
    Connections.delays(network, edges)
    if issubclass(network.__class__, SpatialGraph):
       Connections.distances(network, edges)
