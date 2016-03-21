"""
Module dedicated to logging the simulations and networks generated via the
library.

Depending on the settings in `$HOME/.nngt.conf`, the data will either be stored
in a in a SQL database or in CSV files.

Content
=======
"""

from .db_main import nngt_db


#-----------------------------------------------------------------------------#
# Declare content
#------------------------
#

__all__ = [
	'nngt_db',
]
