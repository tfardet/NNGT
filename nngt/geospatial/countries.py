# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/geospatial/countries.py

import os

import cartopy
import geopandas as gpd
import numpy as np
import pandas as pd

from cartopy.io.shapereader import natural_earth
from shapely.geometry import Point


# --------------------------------------------- #
# Load and store the worlds at different scales #
# --------------------------------------------- #

try:
    fname_110 = natural_earth(resolution='110m', category='cultural',
                              name='admin_0_countries')

    fname_50 = natural_earth(resolution='50m', category='cultural',
                             name='admin_0_countries')

    fname_10 = natural_earth(resolution='10m', category='cultural',
                             name='admin_0_map_units')

    fname_cities = natural_earth(resolution='10m', category='cultural',
                                 name='populated_places_simple')
except Exception as e:
    print("Error excountered:", e)
    raise RuntimeError("The `geospatial` module needs to download the "
                       "NaturalEarth maps on its first run, please make sure "
                       "you are connected to the internet or manually add the "
                       "files to '{}'".format(cartopy.config['data_dir']))

world_110 = gpd.read_file(fname_110)
world_50  = gpd.read_file(fname_50)
world_10  = gpd.read_file(fname_10)
cities    = gpd.read_file(fname_cities)

maps = {
    "110m": world_110,
    "50m": world_50,
    "10m": world_10,
    "cities": cities,
}


# ----------------------------------------- #
# Store countries and their world/positions #
# ----------------------------------------- #

country_codes_adaptive = {}
country_codes_110 = {}
country_codes_50 = {}
country_codes_10 = {}

country_names_adaptive = {}
country_names_110 = {}
country_names_50 = {}
country_names_10 = {}

ctn_adaptive = {}
ctn_110 = {}
ctn_50 = {}
ctn_10 = {}

countries = {"Country", "Sovereign country", 'Geo unit'}

codes = ("GU_A3", "SOV_A3", "ADM0_A3")

for i, v in world_110.iterrows():
    name = v.NAME_LONG
    code = v.SU_A3

    cvalues = [v[c] for c in codes]

    country_names_110[name] = i
    country_names_adaptive[name] = i

    country_codes_110[code] = i
    country_codes_adaptive[code] = i

    ctn_110[code] = name
    ctn_adaptive[code] = name

    for cval in cvalues:
        if v.TYPE in countries and cval not in country_codes_adaptive:
            country_codes_110[cval] = i
            country_codes_adaptive[cval] = i
            ctn_110[cval] = name
            ctn_adaptive[cval] = name


new_countries = set()
size_adaptive = len(world_110)

for i, v in world_50.iterrows():
    name = v.NAME_LONG
    code = v.SU_A3

    cvalues = [v[c] for c in codes]

    country_names_50[name] = i
    country_codes_50[code] = i

    ctn_50[code] = name

    if name not in country_names_adaptive:
        country_names_adaptive[name] = len(new_countries) + size_adaptive
        country_codes_adaptive[code] = len(new_countries) + size_adaptive
        ctn_adaptive[code] = name

        new_countries.add(i)

    for cval in cvalues:
        if v.TYPE in countries and cval not in country_codes_50:
            country_codes_50[cval] = i
            ctn_50[cval] = name

            if cval not in country_codes_adaptive:
                country_codes_adaptive[cval] = \
                    len(new_countries) + size_adaptive
                ctn_adaptive[cval] = name

world = pd.concat((world_110, world_50.iloc[list(new_countries)]),
                  ignore_index=True)

new_countries = []
size_adaptive = len(world)

for i, v in world_10.iterrows():
    name = v.NAME_LONG
    code = v.SU_A3

    cvalues = [v[c] for c in codes]

    country_names_10[name] = i
    country_codes_10[code] = i

    ctn_10[code] = name

    for cval in cvalues:
        if v.TYPE in countries and cval not in country_codes_10:
            country_codes_10[cval] = i
            ctn_10[cval] = name

    if name not in country_names_adaptive:
        sovc = v.SOV_A3

        # check if it's a subunit, if so, add only if it does not overlap
        # with sovereign territory (special case for Antigua and Barbuda)
        # ignore Israel unrecognized territories
        if v.LEVEL == 3 and sovc not in ("ATG", "IS1"):
            geom = v.geometry

            # get soverign territory
            idx = country_codes_adaptive[sovc]

            sov_geom = world.iloc[idx].geometry

            overlap = geom.intersection(sov_geom).area > 0.1*geom.area

            if overlap:
                continue

        country_names_adaptive[name] = len(new_countries) + size_adaptive
        country_codes_adaptive[code] = len(new_countries) + size_adaptive

        ctn_adaptive[code] = name

        if v.TYPE in countries:
            for cval in cvalues:
                if cval not in country_codes_adaptive:
                    country_codes_adaptive[cval] = \
                        len(new_countries) + size_adaptive
                    ctn_adaptive[cval] = name

        new_countries.append(i)


world = pd.concat((world, world_10.iloc[new_countries]), ignore_index=True)

maps["adaptive"] = world

country_names = {
    "110m": country_names_110,
    "50m": country_names_50,
    "10m": country_names_10,
    "adaptive": country_names_adaptive
}

country_codes = {
    "110m": country_codes_110,
    "50m": country_codes_50,
    "10m": country_codes_10,
    "adaptive": country_codes_adaptive
}

codes_to_names = {
    "110m": ctn_110,
    "50m": ctn_50,
    "10m": ctn_10,
    "adaptive": ctn_adaptive
}


# ------------------------------------- #
# Add usual covertors for country names #
# ------------------------------------- #

convertors = {
    'Democratic Republic of Korea': 'Dem. Rep. Korea',
    'Iran (Islamic Republic of)': 'Iran',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Bolivia (Plurinational State of)': 'Bolivia',
    'China, Taiwan Province of': 'Taiwan',
    'North Macedonia': 'Macedonia',
    'China, mainland': 'China',
    'Serbia and Montenegro': 'Serbia',
    'Sudan (former)': 'Sudan',
    'Belgium-Luxembourg': 'Belgium',
    'Ethiopia PDR': 'Ethiopia',
    'Yugoslav SFR': 'Serbia',
    'Cabo Verde': 'Republic of Cabo Verde',
    'Eswatini': 'eSwatini',
    'Sao Tome and Principe': 'São Tomé and Principe',
    'China, Hong Kong SAR': 'Hong Kong',
    'China, Macao SAR': 'Macao',
    'Congo': 'Republic of the Congo',
    'Czechia': 'Czech Republic',
    'Gambia': 'The Gambia',
    'Netherlands Antilles (former)': 'Curaçao',
    'Falkland Islands (Malvinas)': 'Falkland Islands',
    'Viet Nam': 'Vietnam',
    'Saint Helena, Ascension and Tristan da Cunha': 'Saint Helena',
    'Pitcairn': 'Pitcairn Islands',
    'Micronesia (Federated States of)': 'Federated States of Micronesia',
    'French Southern Territories': 'French Southern and Antarctic Lands',
    'Palestinian Territory': 'Palestine'
}


for k, v in zip(world.FORMAL_EN, world.NAME_LONG):
    if k is not None:
        convertors[k] = v


convertors.update(
    {k: v for k, v in zip(world.NAME_EN, world.NAME_LONG)})


# ----------------------------------- #
# Representative points for countries #
# ----------------------------------- #

points = []

for _, v in world.iterrows():
    if v.TYPE in countries:
        # check capital
        idx = np.where((cities.sov_a3 == v.SOV_A3)*cities.adm0cap)[0]

        if len(idx) == 1:
            points.append(cities.iloc[idx[0]].geometry)
            continue
        elif len(idx) > 1:
            pop = cities.iloc[idx].pop_max
            points.append(cities.iloc[pop.idxmax()].geometry)
            continue
        else:
            idx = np.where((cities.sov_a3 == v.SU_A3)*cities.adm0cap)[0]

            if len(idx) == 1:
                points.append(cities.iloc[idx[0]].geometry)
                continue
            elif len(idx) > 1:
                pop = cities.iloc[idx].pop_max
                points.append(cities.iloc[pop.idxmax()].geometry)
                continue

    points.append(v.geometry.representative_point())


country_points = gpd.GeoDataFrame({
    'country': world.NAME_LONG,
    'geocode': world.SU_A3,
    'geometry': points
})

country_points.set_crs(epsg=world.crs.to_epsg(), inplace=True)
