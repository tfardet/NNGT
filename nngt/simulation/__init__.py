"""
Content
=======
"""

import sys
sys.argv.append('--quiet')

import nngt


#
#---
# Dependencies
#---------------------

depends = ['nest', 'nngt.core']

from .nest_graph import *
from .nest_utils import *
from .nest_activity import *

nngt.__all__.append('simulation')


#
#---
# Declare functions
#---------------------

__all__ = [
    'activity_types',
    'analyse_raster',
	'get_nest_network',
	'make_nest_network',
    'monitor_groups',
    'monitor_nodes',
    'plot_activity',
    'raster_plot',
    'set_noise',
    'set_poisson_input',
    'set_set_step_currents',
]

# test import of simulation plotting tools

if nngt._config['with_plot']:
    from .nest_plot import plot_activity
    __all__.append("plot_activity")
