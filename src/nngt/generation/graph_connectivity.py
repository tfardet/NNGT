#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Connectivity generators for Graph """

import numpy as np # remove it in the end

from .. import Graph, SpatialGraph
from ..core import GraphObject
from ..lib.utils import (delete_self_loops, delete_parallel_edges,
                         adjacency_matrix, make_spatial) # remove in the end
from ..lib.connect_tools import *



n_MAXTESTS = 1000 # ensure that generation will finish # remove in the end


#
#---
# Erdos-Renyi
#------------------------

def erdos_renyi(nodes=None, density=0.1, edges=-1, avg_deg=-1.,
                reciprocity=-1., directed=True, multigraph=False,
                name="ER", shape=None, positions=None, initial_graph=None):
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
    initial_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_er : :class:`~nngt.Graph`, or subclass
        A new generated graph or the modified `initial_graph`.

    Notes
    -----
    `nodes` is required unless `initial_graph` is provided.
    If an `initial_graph` is provided, all preexistant edges in the
    object will be deleted before the new connectivity is implemented.
    """
    np.random.seed()
    nodes = nodes if initial_graph is None else initial_graph.node_nb()
    # generate graph object
    graph_obj_er = (GraphObject(nodes, directed=directed) if
                    initial_graph == None else initial_graph.graph)
    graph_obj_er.clear_edges()
    # add edges
    ids = np.arange(nodes).astype(int)
    ia_edges = _erdos_renyi(ids, ids, density, edges, avg_deg, reciprocity,
                            directed, multigraph)
    graph_obj_er.add_edge_list(ia_edges)
    # generate container
    graph_er = (Graph(name=name, libgraph=graph_obj_er) if
                initial_graph is None else initial_graph)
    if shape is not None:
        make_spatial(graph_er, shape, positions)
    return graph_er


#
#---
# Scale-free models
#------------------------

def random_scale_free(in_exp, out_exp, nodes=None, density=0.1, edges=-1,
                      avg_deg=-1, reciprocity=0., directed=True,
                      multigraph=False, name="ER", shape=None, positions=None,
                      initial_graph=None):
    """
    @todo
    Generate a free-scale graph of given reciprocity and otherwise
    devoid of correlations.

    Parameters 
    ----------
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

    Returns
    -------
    graph_fs : :class:`~nngt.Graph`
    
    Notes
    -----
    `nodes` is required unless `initial_graph` is provided.
    """
    np.random.seed()
    nodes = nodes if initial_graph is None else initial_graph.node_nb() 
    edges = _compute_connections(nodes, density, edges, avg_deg)
    frac_recip = 0.
    pre_recip_edges = edges
    if not directed:
        edges = int(edges/2)
    elif reciprocity > 0.:
        frac_recip = reciprocity/(2.0-reciprocity)
        pre_recip_edges = edges / (1+frac_recip)

    # define necessary constants

    # for probability functions F(x) = c * x^{-tau} (for in- and out-degrees)
    c_in = edges*(in_exp-1)/(nodes)
    c_out = edges*(out_exp-1)/(nNodes)
    # for the average of the pareto 2 = lomax distribution
    avg_in = 1/(in_exp-2.)
    avg_out = 1/(out_exp-2.)
    # lists containing the in/out-degrees for nodes i
    lst_in_deg = np.random.pareto(in_exp,nodes)+1
    lst_out_deg = np.random.pareto(out_exp,nodes)+1
    lst_in_deg = np.floor(np.multiply(c_in/np.mean(lst_in_deg),
                                      lst_in_deg)).astype(int)
    lst_out_deg = np.floor(np.multiply(c_out/np.mean(lst_out_deg),
                                       lst_out_deg)).astype(int)

    # count and generate the stubs (half edges)
    num_in_stubs = int(np.sum(lst_in_deg))
    num_out_stubs = int(np.sum(lst_out_deg))
    lstInStubs = np.zeros(num_in_stubs)
    lstOutStubs = np.zeros(num_out_stubs)
    nStartIn = 0
    nStartOut = 0
    for node in range(nNodes):
        nInDegVert = lstInDeg[node]
        nOutDegVert = lstOutDeg[node]
        for j in range(np.max([nInDegVert,nOutDegVert])):
            if j < nInDegVert:
                lstInStubs[nStartIn+j] += node
            if j < nOutDegVert:
                lstOutStubs[nStartOut+j] += node
        nStartOut+=nOutDegVert
        nStartIn+=nInDegVert
    # on vérifie qu'on a à peu près le nombre voulu d'edges
    while nInStubs*(1+rFracRecip)/float(nArcs) < 0.95 :
        node = np.random.randint(0,nNodes)
        nAddInStubs = int(np.floor(Ai/rMi*(np.random.pareto(rInDeg)+1)))
        lstInStubs = np.append(lstInStubs,np.repeat(node,nAddInStubs)).astype(int)
        nInStubs+=nAddInStubs
    while nOutStubs*(1+rFracRecip)/float(nArcs) < 0.95 :
        nAddOutStubs = int(np.floor(Ao/rMo*(np.random.pareto(rOutDeg)+1)))
        lstOutStubs = np.append(lstOutStubs,np.repeat(node,nAddOutStubs)).astype(int)
        nOutStubs+=nAddOutStubs
    # on s'assure d'avoir le même nombre de in et out stubs (1.13 is an experimental correction)
    nMaxStubs = int(1.13*(2.0*nArcs)/(2*(1+rFracRecip)))
    if nInStubs > nMaxStubs and nOutStubs > nMaxStubs:
        np.random.shuffle(lstInStubs)
        np.random.shuffle(lstOutStubs)
        lstOutStubs.resize(nMaxStubs)
        lstInStubs.resize(nMaxStubs)
        nOutStubs = nInStubs = nMaxStubs
    elif nInStubs < nOutStubs:
        np.random.shuffle(lstOutStubs)
        lstOutStubs.resize(nInStubs)
        nOutStubs = nInStubs
    else:
        np.random.shuffle(lstInStubs)
        lstInStubs.resize(nOutStubs)
        nInStubs = nOutStubs
    # on crée le graphe, les noeuds et les stubs
    nRecip = int(np.floor(nInStubs*rFracRecip))
    nEdges = nInStubs + nRecip +1
    # les stubs réciproques
    np.random.shuffle(lstInStubs)
    np.random.shuffle(lstOutStubs)
    lstInRecip = lstInStubs[0:nRecip]
    lstOutRecip = lstOutStubs[0:nRecip]
    lstEdges = np.array([np.concatenate((lstOutStubs,lstInRecip)),np.concatenate((lstInStubs,lstOutRecip))]).astype(int)
    # add edges
    graphFS.add_edge_list(np.transpose(lstEdges))
    delete_self_loops(graphFS)
    delete_parallel_edges(graphFS)
    graphFS.reindex_edges()
    nNodes = graphFS.node_nb()
    nEdges = graphFS.edge_nb()
    rDens = nEdges / float(nNodes**2)
    # generate types
    rInhibFrac = dicProperties["InhibFrac"]
    lstTypesGen = np.random.uniform(0,1,nEdges)
    lstTypeLimit = np.full(nEdges,rInhibFrac)
    lstIsExcitatory = np.greater(lstTypesGen,lstTypeLimit)
    nExc = np.count_nonzero(lstIsExcitatory)
    epropType = graphFS.new_edge_property("int",np.multiply(2,lstIsExcitatory)-np.repeat(1,nEdges)) # excitatory (True) or inhibitory (False)
    graphFS.edge_properties["type"] = epropType
    # and weights
    if dicProperties["Weighted"]:
        lstWeights = dicGenWeights[dicProperties["Distribution"]](graphFS,dicProperties,nEdges,nExc) # generate the weights
        epropW = graphFS.new_edge_property("double",lstWeights) # crée la propriété pour stocker les poids
        graphFS.edge_properties["weight"] = epropW
    return graphFS

def price_scale_free(m, c=None, gamma=1, nodes=None, directed=True,
                     seed_graph=None, multigraph=False, name="ER", shape=None,
                     positions=None, initial_graph=None):
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
    initial_graph : :class:`~nngt.Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.
    
    Returns
    -------
    graph_price : :class:`~nngt.Graph` or subclass.
    
    Notes
    -----
    `nodes` is required unless `initial_graph` is provided.
    """
    np.random.seed()
    nodes = nodes if initial_graph is None else initial_graph.node_nb()
    #~ c = c if c is not None else 0 if directed else 1
    
    graph_obj_price = GraphObject.to_graph_object(
                        price_network(nodes,m,c,gamma,directed,seed_graph))
    if initial_graph is not None:
        initial_graph.graph = graph_obj_price
    
    graph_price = (Graph(name=name, libgraph=graph_obj_price) if
                initial_graph == None else initial_graph)
    if shape is not None:
        make_spatial(graph_price, shape, positions)
    return graph_price

#
#---
# Small-world models
#------------------------

def newman_watts(coord_nb, proba_shortcut, nodes=None, density=0.1, edges=-1,
                  avg_deg=-1., reciprocity=-1., directed=True,
                  multigraph=False, name="ER", shape=None, positions=None,
                  initial_graph=None):
    """
    Generate a small-world graph using the Newman-Watts algorithm.
    @todo: generate the edges of a circular graph to not replace the graph
    of the `initial_graph` and implement chosen reciprocity.
    
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
    initial_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.
    
    Returns
    -------
    graph_nw : :class:`~nngt.Graph` or subclass
    
    Notes
    -----
    `nodes` is required unless `initial_graph` is provided.
    """
    np.random.seed()
    nodes = nodes if initial_graph is None else initial_graph.node_nb()
    # generate graph object
    graph_obj_nw = (GraphObject(nodes, directed=directed) if
                    initial_graph == None else initial_graph.graph)
    graph_obj_nw.clear_edges()
    # add edges
    ia_edges = _newman_watts(coord_nb, proba_shortcut, nodes, density, edges,
                             avg_deg, reciprocity, directed, multigraph)
    graph_obj_nw.add_edge_list(ia_edges)
    # generate container
    graph_nw = (Graph(name=name, libgraph=graph_obj_nw) if
                initial_graph == None else initial_graph)
    if shape is not None:
        make_spatial(graph_nw, shape, positions)
    return graph_nw


#
#---
# Distance-based models
#------------------------

def gen_edr(dicProperties):
    np.random.seed()
    # on définit toutes les grandeurs de base
    rRho2D = dicProperties["Rho"]
    rLambda = dicProperties["Lambda"]
    nNodes = 0
    nEdges = 0
    rDens = 0.0
    if "Nodes" in dicProperties.keys():
        nNodes = dicProperties["Nodes"]
        if "Edges" in dicProperties.keys():
            nEdges = dicProperties["Edges"]
            rDens = nEdges / float(nNodes**2)
            dicProperties["Density"] = rDens
        else:
            rDens = dicProperties["Density"]
            nEdges = int(np.floor(rDens*nNodes**2))
            dicProperties["Edges"] = nEdges
    else:
        nEdges = dicProperties["Edges"]
        rDens = dicProperties["Density"]
        nNodes = int(np.floor(np.sqrt(nEdges/rDens)))
        dicProperties["Nodes"] = nNodes
    rSideLength = np.sqrt(nNodes/rRho2D)
    rAverageDistance = np.sqrt(2)*rSideLength / 3
    # generate the positions of the neurons
    lstPos = np.array([np.random.uniform(0,rSideLength,nNodes),np.random.uniform(0,rSideLength,nNodes)])
    lstPos = np.transpose(lstPos)
    numDesiredEdges = int(float(rDens*nNodes**2))
    graphEDR,pos = geometric_graph(lstPos,0)
    graphEDR.set_directed(True)
    graphEDR.vertex_properties["pos"] = pos
    # test edges building on random neurons
    nEdgesTot = graphEDR.edge_nb()
    numTest = 0
    while nEdgesTot < numDesiredEdges and numTest < n_MAXTESTS:
        nTests = int(np.minimum(1.1*np.ceil(numDesiredEdges-nEdgesTot)*np.exp(np.divide(rAverageDistance,rLambda)),1e7))
        lstVertSrc = np.random.randint(0,nNodes,nTests)
        lstVertDest = np.random.randint(0,nNodes,nTests)
        lstDist = np.linalg.norm(lstPos[lstVertDest]-lstPos[lstVertSrc],axis=1)
        lstDist = np.exp(np.divide(lstDist,-rLambda))
        lstCreateEdge = np.random.uniform(size=nTests)
        lstCreateEdge = np.greater(lstDist,lstCreateEdge)
        nEdges = np.sum(lstCreateEdge)
        if nEdges+nEdgesTot > numDesiredEdges:
            nEdges = numDesiredEdges - nEdgesTot
            lstVertSrc = lstVertSrc[lstCreateEdge][:nEdges]
            lstVertDest = lstVertDest[lstCreateEdge][:nEdges]
            lstEdges = np.array([lstVertSrc,lstVertDest]).astype(int)
        else:
            lstEdges = np.array([lstVertSrc[lstCreateEdge],lstVertDest[lstCreateEdge]]).astype(int)
        graphEDR.add_edge_list(np.transpose(lstEdges))
        # make graph simple and connected
        delete_self_loops(graphEDR)
        delete_parallel_edges(graphEDR)
        nEdgesTot = graphEDR.edge_nb()
        numTest += 1
    graphEDR.reindex_edges()
    nNodes = graphEDR.node_nb()
    nEdges = graphEDR.edge_nb()
    rDens = nEdges / float(nNodes**2)
    # generate types
    rInhibFrac = dicProperties["InhibFrac"]
    lstTypesGen = np.random.uniform(0,1,nEdges)
    lstTypeLimit = np.full(nEdges,rInhibFrac)
    lstIsExcitatory = np.greater(lstTypesGen,lstTypeLimit)
    nExc = np.count_nonzero(lstIsExcitatory)
    epropType = graphEDR.new_edge_property("int",np.multiply(2,lstIsExcitatory)-np.repeat(1,nEdges)) # excitatory (True) or inhibitory (False)
    graphEDR.edge_properties["type"] = epropType
    # and weights
    if dicProperties["Weighted"]:
        lstWeights = dicGenWeights[dicProperties["Distribution"]](graphEDR,dicProperties,nEdges,nExc) # generate the weights
        epropW = graphEDR.new_edge_property("double",lstWeights) # crée la propriété pour stocker les poids
        graphEDR.edge_properties["weight"] = epropW
    return graphEDR


#
#---
# Connecting groups
#------------------------

di_gen_func = { "erdos_renyi": _erdos_renyi, 
    "random_scale_free": _random_scale_free,
    "price_scale_free": _price_scale_free,
    "newman_watts": _newman_watts }

di_default = {  "density": -1.,
                "edges": -1,
                "avg_deg": -1,
                "reciprocity": -1,
                "directed": True,
                "multigraph": False }

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
    source_ids, target_ids = [], []
    di_param = di_default.copy()
    di_param.update(model_param)
    for group in network._population.itervalues():
        if group.neuron_type == source_type:
            source_ids.extend(group._id_list)
        elif group.neuron_type == target_type:
            target_ids.extend(group._id_list)
    if source_type == target_type:
        edges = di_gen_func[graph_model](source_ids, source_ids, **di_param)
        network.add_edges(edges)
    elif graph_model in ("newman_watts",):
        raise ArgumentError("This graph model can only be used if source and \
            target populations are the same")
    else:
        edges = di_gen_func[graph_model](source_ids, target_ids, **di_param)
        network.add_edges(edges)

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
    source_ids, target_ids = [], []
    for name, group in network._population.iteritems():
        if name in source_groups:
            source_ids.extend(group._id_list)
        elif name in target_groups:
            target_ids.extend(group._id_list)
    if source_groups == target_groups:
        edges = di_gen_func[graph_model](source_ids, source_ids, **model_param)
        network.add_edges(edges)
    elif graph_model in ("newman_watts",):
        raise ArgumentError("This graph model can only be used if source and \
            target populations are the same")
    else:
        edges = di_gen_func[graph_model](source_ids, target_ids, **model_param)
        network.add_edges(edges)
