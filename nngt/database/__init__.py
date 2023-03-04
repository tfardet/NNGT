# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/database/__init__.py

"""
Module dedicated to logging the simulations and networks generated via the
library.

Depending on the settings in `$HOME/.nngt.conf`, the data will either be stored
in a in a SQL database or in CSV files.

Content
=======
"""

import os, errno
import logging

from peewee import *
from playhouse.db_url import connect

import nngt
from nngt.lib.logger import _log_message


__all__ = [
    'get_results',
    'is_clear',
    'log_simulation_end',
    'log_simulation_start',
    'reset',
]


logger = logging.getLogger(__name__)


# --------------------------------------- #
# Parse config file and generate database #
# --------------------------------------- #

def _set_main_db():
    _log_message(logger, "WARNING", "The database module will be removed in "
                 "version 2.8.0, please get in touch if you are using it.")
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
