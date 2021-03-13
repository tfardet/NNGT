"""
Header generation for all python files

NB: this file must be run from inside the NNGT folder via
``python extra/format_headers.py``.
"""

import datetime
import os


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
                    data = rf.readlines()

                with open(abspath, "w") as wf:
                    wf.write(header.format(filename=path + f, year=year))

                    is_header = True

                    for line in data:
                        if not line.startswith("#") or not is_header:
                            wf.write(line)
                            is_header = False
