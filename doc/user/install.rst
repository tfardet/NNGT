============
Installation
============


Dependencies
============

This package depends on several libraries (the number varies according to which
modules you want to use).


Basic dependencies
------------------

Regardless of your needs, the following libraries are required:

* `numpy <http://www.numpy.org/>`_ (>= 1.11 required for full support)
* `scipy <http://www.scipy.org/scipylib/index.html>`_

Though NNGT implements a default (very basic) backend, installing one of the
following libraries is highly recommended to do some proper network
analysis:

* graph_tool_ (> 2.22 recommended)
* or igraph_
* or networkx_ (>= 2.0)


Additionnal dependencies
------------------------

* matplotlib_ (optional but will limit the functionalities if not present)
* shapely_ for complex spatial embedding
* `peewee>3` for database features

.. note::
    If they are not present on your computer, `pip` will directly try to
    install scipy and numpy.
    However, if you want advanced network analysis features, you will have to
    install the graph library yourself (only `networkx` can be installed
    directly using `pip`)


Simple install
==============

Linux
-----

Install the requirements (through ``aptitude`` or ``apt-get`` on
debian/ubuntu/mint, ``pacman`` and ``yaourt`` on arch-based distributions, or
your `.rpm` manager on fedora. Otherwise you can also install the latest
versions via `pip`: ::

    pip install --user numpy scipy matplotlib networkx

To install the last stable release, just use: ::

    pip install --user nngt

Under most linux distributions, the simplest way to get the latest version
of NNGT is to install to install both
`pip <https://pip.pypa.io/en/stable/installing/>`_ and
`git <https://git-scm.com/>`_, then simply type into a terminal: ::

    pip install --user git+https://github.com/Silmathoron/NNGT.git


Mac
---

I recommend using `Homebrew <https://brew.sh/>`_ or `Macports
<https://guide.macports.org/#installing>`_ with which you can install all
required features to use `NEST` and `NNGT` with `graph-tool`. The following
command lines are used with `python 2.7` since it is what people are used to
but I recommend using version `3.5` or higher (replace all 27/2.7 by 35/3.5).

**Homebrew** ::

    brew tap homebrew/core
    brew tap brewsci/science

    brew install gcc-7 cmake gsl autoconf automake libtool
    brew install python

if you want nest, add ::

    brew install nest --with-python

(note that setting ``--with-python=3`` might be necessary)

**Macports** ::

    sudo port select gcc mp-gcc7 && sudo port install gsl +gcc7
    sudo port install autoconf automake libtool
    sudo port install python27 pip
    sudo port select python python27
    sudo port install py27-cython
    sudo port select cython cython27
    sudo port install py27-numpy py27-scipy py27-matplotlib py27-ipython
    sudo port select ipython ipython-2.7
    sudo port install py-graph-tool gtk3

Once the installation is done, you can just install: ::

    export CC=gcc-7
    export CXX=gcc-7
    pip install --user nngt


Windows
-------

It's the same as Linux for windows users once you've installed
`Python <http://docs.python-guide.org/en/latest/starting/install/win/>`_ and
`pip`, but `NEST <http://www.nest-simulator.org/>`_ won't work anyway...

.. note ::
    `igraph` can be installed on windows if you need something faster than
    `networkx`.

**Using the multithreaded algorithms**

Install a compiler (the default `msvc` should already be present, otherwise
you can install VisualStudio) before you make the installation.

In case of problems with `msvc`:

* install `MinGW <http://mingw.org/>`_ or
  `MinGW-W64 <https://mingw-w64.org/doku.php>`_
* use it to install gcc with g++ support
* open a terminal, add the compiler to your `PATH` and set it as default:
  e.g. ::

    set PATH=%PATH%;C:\MinGW\bin
    set CC=C:\MinGW\bin\mingw32-gcc.exe
    set CXX=C:\MinGW\bin\mingw32-g++.exe
* in that same terminal window, run ``pip install --user nngt``


Local install
=============

If you want to modify the library more easily, you can also install it locally,
then simply add it to your ``PYTHONPATH`` environment variable: ::

    cd && mkdir .nngt-install
    cd .nngt-install
    git clone https://github.com/Silmathoron/NNGT.git .
    git submodule init
    git submodule update
    nano .bash_profile

Then add: ::

    export PYTHONPATH="/path/to/your/home/.nngt-install/src/:PYTHONPATH"

In order to update your local repository to keep it up to date, you will need
to run the two following commands: ::

    git pull origin master
    git submodule update --remote --merge


Configuration
=============

The configuration file is created in ``~/.nngt/nngt.conf`` after you first run
``import nngt`` in `python`. Here is the default file:

.. literalinclude:: ../../nngt/nngt.conf.default

It can be necessary to modify this file to use the desired graph library, but
mostly to correct problems with GTK and matplotlib (if the `plot` module
complains, try ``Gtk3Agg`` and ``Qt4Agg``/``Qt5Agg``).


Using NEST
==========

If you want to simulate activities on your complex networks, NNGT can directly
interact with the `NEST simulator`_ to implement the network inside `PyNEST`.
For this, you will need to install NEST with Python bindings, which requires:

* the python headers (`python-dev` package on debian-based distribs)
* `autoconf`
* `automake`
* `libtool`
* `libltdl`
* `libncurses`
* `readlines`
* `gsl` (the GNU Scientific Library) for many neuronal models


.. _graph_tool: http://graph-tool.skewed.de
.. _igraph: http://igraph.org/
.. _matplotlib: http://matplotlib.org/
.. _NEST simulator: http://www.nest-simulator.org/
.. _networkx: https://networkx.github.io/
.. _shapely: http://shapely.readthedocs.io/en/latest/index.html
