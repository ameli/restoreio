# SPDX-FileCopyrightText: Copyright 2016, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import numpy
from mpl_toolkits.basemap import Basemap

__all__ = ['draw_map']


# ========
# Draw map
# ========

def draw_map(
        axis,
        lon,
        lat,
        draw_features=False):
    """
    Returns a basemap object for plotting maps.
    """

    # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
    percent = 0.05   # For Monterey Dataset
    # percent = 0.1     # For Martha Dataset
    lon_offset = percent * numpy.abs(lon[-1] - lon[0])
    lat_offset = percent * numpy.abs(lat[-1] - lat[0])

    min_lon = numpy.min(lon)
    min_lon_with_offset = min_lon - lon_offset
    mid_lon = numpy.mean(lon)
    max_lon = numpy.max(lon)
    max_lon_with_offset = max_lon + lon_offset
    min_lat = numpy.min(lat)
    min_lat_with_offset = min_lat - lat_offset
    mid_lat = numpy.mean(lat)
    max_lat = numpy.max(lat)
    max_lat_with_offset = max_lat + lat_offset

    # Basemap (set resolution to 'i' for faster rasterization and 'f' for
    # full resolution but very slow.)
    map = Basemap(
            ax=axis,
            projection='aeqd',
            llcrnrlon=min_lon_with_offset,
            llcrnrlat=min_lat_with_offset,
            urcrnrlon=max_lon_with_offset,
            urcrnrlat=max_lat_with_offset,
            area_thresh=0.1,
            lon_0=mid_lon,
            lat_0=mid_lat,
            resolution='i')

    # Map features
    if draw_features:
        map.drawcoastlines()
        # map.drawstates()
        # map.drawcountries()
        # map.drawcounties()
        map.drawlsmask(land_color='Linen', ocean_color='#C7DCEF', lakes=True,
                       zorder=-2)
        # map.fillcontinents(color='red', lake_color='white', zorder=0)
        map.fillcontinents(color='moccasin', zorder=-1)

        # map.bluemarble()
        # map.shadedrelief()
        # map.etopo()

        # lat and lon lines
        lon_lines = numpy.linspace(min_lon, max_lon, 2)
        lat_lines = numpy.linspace(min_lat, max_lat, 2)
        map.drawparallels(lat_lines, labels=[1, 0, 0, 0], fontsize=10)
        map.drawmeridians(lon_lines, labels=[0, 0, 0, 1], fontsize=10)

    return map
