#!/usr/bin/env python
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

""" Logging for the NNGT module """

import os
import logging
import warnings
from datetime import date

import scipy.sparse as ssp

import nngt
from .test_functions import mpi_checker


# ignore scipy.sparse efficiency warnings

warnings.filterwarnings("ignore", category=ssp.SparseEfficiencyWarning)

# check that log folder exists, otherwise create it

nngt._config["log_folder"] = os.path.expanduser(nngt._config["log_folder"])
if not os.path.isdir(nngt._config["log_folder"]):
    os.mkdir(nngt._config["log_folder"])


# configure logger

def _init_logger(logger):
    logger.handlers = []  # necessary for ipython
    logger.setLevel(logging.INFO)
    logConsoleFormatter = logging.Formatter(
        '[%(levelname)s @ %(name)s]: %(message)s')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logConsoleFormatter)
    consoleHandler.setLevel(nngt._config["log_level"])
    logger.addHandler(consoleHandler)


def _configure_logger(logger):
    logger.setLevel(logging.INFO)
    is_writing_to_file = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(nngt._config["log_level"])
        elif isinstance(handler, logging.FileHandler):
            is_writing_to_file = True
    if nngt._config["log_to_file"]:
        _log_to_file(logger, not is_writing_to_file)


# initialize file log

def _log_to_file(logger, create_writer):
    if create_writer:
        logFileFormatter = logging.Formatter(
            '[%(levelname)s @ %(name)s] %(asctime)s:\n\t%(message)s')
        today = date.today()
        fileName = "/nngt_{}-{}-{}".format(today.month, today.day, today.year)
        fileHandler = logging.FileHandler(
            "{}/{}.log".format(nngt._config["log_folder"], fileName))
        fileHandler.setFormatter(logFileFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)


# log message

@mpi_checker(logging=True)
def _log_message(logger, level, message):
    if level == 'DEBUG':
        logger.debug(message)
    elif level == 'WARNING':
        logger.warning(message)
    elif level == 'INFO':
        logger.info(message)
    else:
        logger.critical(message)
