# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/lib/logger.py

""" Logging for the NNGT module """

import inspect
import logging
import os
import warnings

from datetime import date
from platformdirs import user_log_dir

import scipy.sparse as ssp

import nngt
from .test_functions import mpi_checker


# ignore scipy.sparse efficiency warnings
warnings.filterwarnings("ignore", category=ssp.SparseEfficiencyWarning)

# ignore igraph unconnected network warnings
warnings.filterwarnings("ignore", message="Couldn't reach some vertices at")

# check that log folder exists, otherwise create it

logdir = user_log_dir("nngt", appauthor=False)

nngt._config["log_folder"] = logdir

os.makedirs(logdir, exist_ok=True)


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
            '[%(levelname)s @ %(funcName)s] %(asctime)s:\n\t%(message)s')
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

    stack = inspect.stack()

    fn = stack[-1][1]
    ln = stack[-1][2]

    location = 'from ' + fn[fn.rfind("/") + 1:] + ' (L{}) - '.format(ln)

    message = location + message

    if level == 'DEBUG':
        logger.debug(message)
    elif level == 'WARNING':
        logger.warning(message)
    elif level == 'INFO':
        logger.info(message)
    else:
        logger.critical(message)
