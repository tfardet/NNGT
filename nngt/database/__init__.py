#-*- coding:utf-8 -*-
#
# database/__init__.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Module dedicated to logging the simulations and networks generated via the
library.

Depending on the settings in `$HOME/.nngt.conf`, the data will either be stored
in a in a SQL database or in CSV files.

Content
=======
"""

import os, errno

from peewee import *
from playhouse.db_url import connect

import nngt


__all__ = [
	'get_results',
	'is_clear',
	'log_simulation_end',
	'log_simulation_start',
	'reset',
]



# --------------------------------------- #
# Parse config file and generate database #
# --------------------------------------- #

def _set_main_db():
    if nngt.get_config("db_url") is None or nngt.get_config("db_to_file"):
        # check for db_folder
        abs_dbfolder = os.path.abspath(
            os.path.expanduser(nngt.get_config("db_folder")))
        if not os.path.isdir(abs_dbfolder):
            try:
                os.mkdir(abs_dbfolder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        # create database
        db_file = abs_dbfolder + "/" + nngt.get_config("db_name") + ".db"

        nngt._main_db = SqliteDatabase(
            db_file,
            pragmas=(('journal_mode', 'wal'), ('cache_size', -1024 * 64),
                     ('foreign_keys', 'on'),))
    else:
        nngt._main_db = connect(
            nngt.get_config('db_url'), fields={'longblob': 'longblob'})


# ----------------- #
# Set NNGT database #
# ----------------- #

_set_main_db()

# IMPORTANT: this MUST come after the call to _set_main_db
from .db_main import NNGTdb as _db

if nngt._db is None:
    nngt._db = _db()

# --------- #
# Functions #
# --------- #

get_results          = nngt._db.get_results
is_clear             = nngt._db.is_clear
log_simulation_end   = nngt._db.log_simulation_end
log_simulation_start = nngt._db.log_simulation_start
reset                = nngt._db.reset
