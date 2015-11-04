Properties of graph components
==============================

Components
----------

In the graph libraries used by NNGT, the main components of a graph are *nodes* (also called *vertices* in graph theory), which correspond to *neurons* in neural networks, and *edges*, which link *nodes* and correspond to synaptic connections between neurons in biology.

Node properties
---------------

If you are just working with basic graphs (for instance looking at the influence of topology with purely excitatory networks), then your nodes do not need to have properties. This is the same if you consider only the average effect of inhibitory neurons by including inhibitory connections between the neurons but not a clear distinction between populations of purely excitatory and purely inhibitory neurons.
To model more realistic networks, however, you might want to define these two types of populations and connect them in specific ways.

The :class:`nngt.properties.NeuralModel` class allows you to define specific populations of neurons (only "excitatory" and "inhibitory" populations are implemented so far, but the library can easily be extended with custom populations). Once these populations are defined, you can constrain the connections between those populations.

