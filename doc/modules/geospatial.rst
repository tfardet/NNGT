================
Geospatial module
================

.. currentmodule:: nngt.geospatial

.. automodule:: nngt.geospatial
    :no-members:


Content
=======

The module provides four main tools:

* :func:`~nngt.geospatial.draw_map` to plot a network on a map
* ``maps``, a dictionary of :obj:`~geopandas.GeoDataFrame` containing data
  from the NaturalEarth_ project:

  - ``"adaptive"`` entry contains 280 entries (in 2021) at the coarsest
    scale available for each entry,
  - ``"110m"`` entry contains 177 countries at 110m resolution,
  - ``"50m"`` entry contains 241 countries at 50m resolution,
  - ``"10m"`` entry contains 295 map subunits at 10m resolution.

* ``code_to_names`` is a dictionary converting A3 ISO codes to the associated
  unit's name (available for all four scales: adaptive and 110/50/10m).

* ``cities`` is a :obj:`~geopandas.GeoDataFrame` containing cities' data from
  from NaturalEarth `"populated places"`_.


.. note::
    This data is automatically downloaded when the module is loaded for the
    first time.
    It is stored in ``cartopy.config['data_dir']``.


Details
=======

.. autofunction:: draw_map


.. Links

.. _NaturalEarth: https://www.naturalearthdata.com
.. _`"populated places"`: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places
