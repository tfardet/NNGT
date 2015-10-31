Graph generation
================

Principle
---------

In order to keep the code as generic and easy to maintain as possible, the generation of graphs or networks is divided in several steps:

* **Structured connectivity:** first, a simple graph is generated as an assembly of nodes and edges, without any biological properties. This allows us to implement known graph-theoretical algorithms in a straightforward fashion.
* **Populations:** once the basic structure has been generated, more detailed properties can be added to the graph, such as inhibitory synapses and separation of the neurons into inhibitory and excitatory populations -- these can be done while respecting user-defined constraints.
* **Synaptic properties:** eventually, synaptic properties such as weight/strength and delays can be added to the network.

.. toctree::
   :maxdepth: 1
   
   ../modules/generation
