Graph generation
================

Principle
---------

In order to keep the code as generic and easy to maintain as possible, the generation of graphs or networks is divided in several steps:

* **Structured connectivity:** a simple graph is generated as an assembly of nodes and edges, without any biological properties. This allows us to implement known graph-theoretical algorithms in a straightforward fashion.
* **Populations:** detailed properties can be implemented, such as inhibitory synapses and separation of the neurons into inhibitory and excitatory populations -- these can be done while respecting user-defined constraints.
* **Synaptic properties:** eventually, synaptic properties such as weight/strength and delays can be added to the network.

Modularity
----------

The library as been designed so that these various operations can be realized in any order!

Juste to get work on a topological graph/network:
	1) Create graph class
	2) Connect
	3) Set connection weights (optional)
	4) Spatialize  (optional)
	5) Set types (optional: to use with NEST)

To work on a really spatially embedded graph/network:
	1) Create spatial graph/network
	2) Connect (can depend on positions)
	3) Set connection weights (optional, can depend on positions)
	4) Set types (optional)

Or to model a complex neural network in NEST:
	1) Create spatial network (with space and neuron types)
	2) Connect (can depend on types and positions)
	3) Set connection weights and types (optional, can depend on types and positions)

.. toctree::
   :maxdepth: 1
   
   ../modules/generation
