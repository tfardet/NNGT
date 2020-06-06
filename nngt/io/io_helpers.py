#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" IO helpers """

import nngt
from nngt.lib import InvalidArgument


# ------------ #
# Saving tools #
# ------------ #

def _get_format(fmt, filename):
    if fmt == "auto":
        if filename.endswith('.gml'):
            fmt = 'gml'
        elif filename.endswith('.graphml') or filename.endswith('.xml'):
            fmt = 'graphml'
        elif filename.endswith('.dot'):
            fmt = 'dot'
        elif (filename.endswith('.gt') and
              nngt._config["backend"] == "graph-tool"):
            fmt = 'gt'
        elif filename.endswith('.nn'):
            fmt = 'neighbour'
        elif filename.endswith('.el'):
            fmt = 'edge_list'
        else:
            raise InvalidArgument('Could not determine format from filename '
                                  'please specify `fmt`.')
    return fmt

