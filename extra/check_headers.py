"""
Header generation for all python files

NB: this file must be run from inside the NNGT folder via
``python extra/format_headers.py``.
"""

import datetime
import os
import sys


# check current working directory
cwd = os.path.abspath(os.path.curdir)

assert cwd.endswith("NNGT"), \
    "Script must be called via `python extra/format_headers.py`."

# load header string
headerfile = os.path.join(cwd, "extra/header.txt")

with open(headerfile, "r") as h:
    header = h.read()

# get year
year = datetime.datetime.now().year

sourcedir = os.path.join(cwd, "nngt")

for (dirpath, dirnames, filenames) in os.walk(sourcedir):
    path = dirpath[dirpath.find("nngt") + 5:]
    path += "/" if len(path) else ""

    if not (path.startswith("geometry") or path.startswith("plot/chord_diag")):
        for f in filenames:
            if f.endswith(".py"):
                abspath = os.path.join(dirpath, f)

                with open(abspath, "r") as rf:
                    ref = header.format(filename=path + f, year=year)
                    data = rf.read()

                    if data[:len(ref)] != ref:
                        print(f"Incorrect header for {path + f}.")
                        sys.exit(1)

sys.exit(0)
