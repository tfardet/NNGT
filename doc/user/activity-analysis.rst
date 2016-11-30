=================
Activity analysis
=================

Principle
=========

The interesting fact about having a link between the graph and the simulation
is that you can easily analyze the activity be taking into account what you
know from the graph structure.


Sorted rasters
==============

Rater plots can be sorted depending on some specific node property, e.g. the
degree or the betweenness:

.. code-block:: python
    
    import nest
    
    import nngt
    from nngt.simulation import make_nest_network, monitor_nodes, plot_activity
    
    pop = nngt.NeuralPop.uniform(1000, neuron_model="aeif_psc_alpha")
    net = nngt.generation.gaussian_degree(100, 20, population=pop)
    
    _, nodes = make_nest_network(net)
    recorders, recordables = monitor_nodes(nodes)
    simtime = 1000.
    nest.Simulate(simtime)
    
    fignums = plot_activity(
        recorders, recordables, network=net, show=True, hist=False,
        limits=(0.,simtime), sort="in-degree")


Activity properties
===================

NNGT can also be used to analyze the general properties of a raster.

Either from a .gdf file containing the raster data

.. code-block:: python
        
    import nngt
    from nngt.simulation import analyze_raster
    
    a = analyze_raster("path/to/raster.gdf")
    print(a.phases)
    print(a.properties)

Or from a spike detector gid ``sd``:

.. code-block:: python
    
    a = analyze_raster(sd)


.. toctree::
   :maxdepth: 1
   
   ../modules/simulation
