"""
Content
=======
"""


#
#---
# Dependencies
#---------------------

depends = ['nest', 'graph_tool', 'nngt.core']

from .nest_graph import *
from .nest_utils import *
from .nest_activity import *
from nngt import config


#
#---
# Declare functions
#---------------------

__all__ = [
	'make_nest_network',
	'get_nest_network',
    'set_noise',
    'set_poisson_input',
    'set_set_step_currents',
    'monitor_nodes',
    'plot_activity',
    'activity_types',
    'raster_plot'
]

# test import of simulation plotting tools

if config['with_plot']:
    from .nest_plot import plot_activity
    __all__.append("plot_activity")
