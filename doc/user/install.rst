============
Installation
============


Dependencies
============

This package depends on several libraries (the number varies according to which modules you want to use).

Basic dependencies
------------------

Regardless of your needs, the following libraries are required:

* `numpy <http://www.numpy.org/>`_ 
* `scipy <http://www.scipy.org/scipylib/index.html>`_
* `graph_tool <http://graph-tool.skewed.de>`_ (recommended)
* or `igraph <http://igraph.org/>`_
* or `networkx <https://networkx.github.io/>`_

Additionnal dependencies
-----------------------

* `matplotlib <http://matplotlib.org/>`_ (optional but will limit the functionalities if not present)
* `peewee` for database features

.. note::
    If they are not present on your computer, `pip` will directly try to install the three first libraries, however:

    * `lapack <http://www.netlib.org/lapack/>`_ is necessary for `scipy` and `pip` cannot install it on its own
    * you will have to install the graph library yourself (only `networkx` can be installed directly using `pip`)
    

Simple install
==============

Linux
-----

Install the requirements (through ``aptitude`` or ``apt-get`` on debian/ubuntu/mint, ``pacman`` and ``yaourt`` on arch-based distributions, or your `.rpm` manager on fedora. Otherwise you can also install the latest versions via `pip`: ::

    sudo pip install numpy scipy matplotlib networkx

Under most linux distributions, the simplest way is to install `pip <https://pip.pypa.io/en/stable/installing/>`_ and `git <https://git-scm.com/>`_, then simply type into a terminal: ::

    sudo pip install git+https://github.com/Silmathoron/NNGT.git

Mac
---

I recommend using `Macports <https://guide.macports.org/#installing>`_ with which you can install all required features to use `NEST` and `NNGT` with `graph-tool`. The following command lines are used with `python 2.7` since it is what people are used to but I recommend using version `3.5` or higher (replace all 27/2.7 by 35/3.5). ::

    sudo port select gcc mp-gcc5 && sudo port install gsl +gcc5 && sudo port install autoconf automake libtool && sudo port install python27 pip && sudo port select python python27 && sudo port install py27-cython && sudo port select cython cython27 && sudo port install py27-numpy py27-scipy py27-matplotlib py27-ipython && sudo port select ipython ipython-2.7 && sudo port install py-graph-tool gtk3

Windows
-------

It's the same as linux for windows users once you've installed `Python <http://docs.python-guide.org/en/latest/starting/install/win/>`_ and `pip`, but `NEST <http://www.nest-simulator.org/>`_ won't work anyway...

.. note ::
    `igraph` can be installed on windows if you need something faster than `networkx`.


Local install
=============

If you want to modify the library more easily, you can also install it locally, then simply add it to your ``PYTHONPATH`` environment variable: ::

    cd && mkdir .nngt-install
    cd .nngt-install
    git clone https://github.com/Silmathoron/NNGT.git .
    nano .bash_profile

Then add: ::

    export PYTHONPATH="/path/to/your/home/.nngt-install/src/:PYTHONPATH"


Configuration
=============

The configuration file is created in ``~/.nngt/nngt.conf`` after you first run ``import nngt`` in `python`. Here is the default file: ::

    ###########################
    # NNGT configuration file #
    ###########################

    ## default graph library
    # (choose among "graph-tool", "igraph", "networkx")
    graph_library = graph-tool

    ## Matplotlib backend
    # Uncomment and choose among your available backends (http://matplotlib.org/faq/usage_faq.html#what-is-a-backend)
    #backend = Qt5Agg

    ## settings for data logging
    set_logging = False

    # use a database (if False, results will be stored in CSV files)
    to_file = False
    #log_folder = ~/.nngt/database

    # database url or temporary database used if use_database = False
    # example of real database url: db_url = mysql://user:password@host:port/my_db
    db_url = mysql:///nngt_db

It can be necessary to modify this file to use the desired graph library, but mostly to correct problems with GTK and matplotlib (if the `plot` module complains, try ``Gtk3Agg`` and ``Qt4Agg``).


Using NEST
==========

If you want to simulate activities on your complex networks, NNGT can directly interact with the `NEST simulator`_ to implement the network inside `PyNEST`. For this, you will need to install NEST with Python bindings, which requires:

* the python headers (`python-dev` package on debian-based distribs)
* `autoconf`
* `automake`
* `libtool`
* `libltdl`
* `libncurses`
* `readlines`
* `gsl` (the GNU Scientific Library) for many neuronal models

.. _NEST simulator: http://www.nest-simulator.org/
