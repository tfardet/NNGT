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

Though NNGT implements a default (limited) backend, installing one of the
following libraries is highly recommended to do some proper network
analysis:

* graph_tool_ (> 2.22)
* or igraph_
* or networkx_ (>= 2.4)


Additionnal dependencies
------------------------

* matplotlib_ (optional but will limit the functionalities if not present)
* shapely_ for complex spatial embedding
* peewee (> 3) for database features

.. note::
    If they are not present on your computer, :command:`pip` will directly try
    to install scipy and numpy.
    However, if you want advanced network analysis features, you will have to
    install the graph library yourself (only `networkx` can be installed
    directly using :command:`pip`)


Simple install
==============

Linux
-----

Install the requirements (through :command:`apt` on debian/ubuntu/mint,
:command:`pacman` and :command:`trizen` on arch-based distributions, or
:command:`yum` on fedora/centos. Otherwise you can also install the latest
versions via :command:`pip`: ::

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
command lines are used with `python 3.7` but you can use any python >= 3.5
(just replace all 37/3.7 by the desired version).

**Homebrew** ::

    brew tap homebrew/core
    brew tap brewsci/science

    brew install gcc-8 cmake gsl autoconf automake libtool
    brew install python

if you want nest, add ::

    brew install nest --with-python

(note that setting ``--with-python=3`` might be necessary)

**Macports** ::

    sudo port select gcc mp-gcc8 && sudo port install gsl +gcc8
    sudo port install autoconf automake libtool
    sudo port install python37 pip
    sudo port select python python37
    sudo port install py37-cython
    sudo port select cython cython37
    sudo port install py37-numpy py37-scipy py37-matplotlib py37-ipython
    sudo port select ipython ipython-3.7
    sudo port install py-graph-tool gtk3

Once the installation is done, you can just install: ::

    export CC=gcc-8
    export CXX=gcc-8
    pip install --user nngt


Windows
-------

It's the same as Linux for windows users once you've installed
`Python <http://docs.python-guide.org/en/latest/starting/install/win/>`_ and
:command:`pip`, but `NEST <http://www.nest-simulator.org/>`_ won't work.

.. note ::
    `igraph` can be installed on windows if you need something faster than
    `networkx`.

**Using the multithreaded algorithms**

Install a compiler (the default :command:`msvc` should already be present,
otherwise you can install VisualStudio) before you make the installation.

In case of problems with :command:`msvc`:

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
