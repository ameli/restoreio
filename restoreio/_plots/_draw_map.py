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
from ._plot_utilities import Polygon

__all__ = ['draw_map', 'draw_axis']


# ========
# Draw map
# ========

def draw_map(
        ax,
        lon,
        lat,
        draw_features=False,
        draw_coastlines=True,
        percent=0.05):
    """
    Returns a basemap object for plotting maps.

    If ax is None, the patches on ax will not be produced. Rather, the user
    should call "draw_axis" later when ax is not None.

    The reason to pass a None axis to this function is to create a generic map
    object that can be used on multiple axes by setting map.ax = ax. Hence,
    to initialize a map in the first place, we can just pass a None ax to the
    function to create a map, then assign whatever axis to it later. However,
    in doing so, the patches on the axis will not be drawn. Hence, the user
    should call "draw_axis" on each axis later.

    For Marta's Vineyard, set percent to 0.1.
    For Monterey Bay, set percent to 0.05.
    """

    # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
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

    # Config
    # resolution = 'i'  # low res
    resolution = 'h'  # high res
    # resolution = 'f'  # full res

    # Basemap (set resolution to 'i' for faster rasterization and 'f' for
    # full resolution but very slow.)
    map = Basemap(
            ax=ax,
            projection='aeqd',
            llcrnrlon=min_lon_with_offset,
            llcrnrlat=min_lat_with_offset,
            urcrnrlon=max_lon_with_offset,
            urcrnrlat=max_lat_with_offset,
            area_thresh=0.1,
            lon_0=mid_lon,
            lat_0=mid_lat,
            resolution=resolution)

    # Draw axis (corner patches and background color)
    if ax is not None:
        draw_axis(ax, lon, lat, map, draw_coastlines, draw_features, percent)

    return map


# =========
# Draw Axis
# =========

def draw_axis(
        ax,
        lon,
        lat,
        map,
        draw_coastlines=True,
        draw_features=True,
        percent=0.05):
    """
    Creates patches around axis to hide plots that extends outside of the
    meridian and parallel lines. Also sets the background color for the
    axis.
    """

    # Make sure axis is set in basemap object
    map.ax = ax

    # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
    lon_offset = percent * numpy.abs(lon[-1] - lon[0])
    lat_offset = percent * numpy.abs(lat[-1] - lat[0])

    # Bounds
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

    # Bounds on map
    min_lon_on_map, min_lat_on_map = map(min_lon, min_lat)
    max_lon_on_map, max_lat_on_map = map(max_lon, max_lat)
    diff_lon_on_map = max_lon_on_map - min_lon_on_map

    # Config
    ocean_color = '#C7DCEF'
    land_color = 'moccasin'

    # Patches to hide scalar fields that extend outside of the parallel and
    # meridian grid lines
    if percent > 0.0:

        # Bottom rectangle patch
        rect_bottom_lons = [min_lon_with_offset, max_lon_with_offset,
                            max_lon_with_offset, min_lon_with_offset]
        rect_bottom_lats = [min_lat_with_offset, min_lat_with_offset,
                            min_lat, min_lat]
        rect_bottom_x, rect_bottom_y = map(rect_bottom_lons, rect_bottom_lats)
        rect_bottom_xy = zip(rect_bottom_x, rect_bottom_y)
        poly_bottom = Polygon(list(rect_bottom_xy), facecolor=ocean_color,
                              edgecolor=ocean_color)
        ax.add_patch(poly_bottom)

        # Top rectangle patch
        rect_top_lons = [min_lon_with_offset, max_lon_with_offset,
                         max_lon_with_offset, min_lon_with_offset]
        rect_top_lats = [max_lat, max_lat, max_lat_with_offset,
                         max_lat_with_offset]
        rect_top_x, rect_top_y = map(rect_top_lons, rect_top_lats)
        rect_top_xy = zip(rect_top_x, rect_top_y)
        poly_top = Polygon(list(rect_top_xy), facecolor=ocean_color,
                           edgecolor=ocean_color)
        ax.add_patch(poly_top)

        # Left rectangle patch
        rect_left_lons = [min_lon_with_offset, min_lon, min_lon,
                          min_lon_with_offset]
        rect_left_lats = [min_lat_with_offset, min_lat_with_offset,
                          max_lat_with_offset, max_lat_with_offset]
        rect_left_x, rect_left_y = map(rect_left_lons, rect_left_lats)
        rect_left_xy = zip(rect_left_x, rect_left_y)
        poly_left = Polygon(list(rect_left_xy), facecolor=ocean_color,
                            edgecolor=ocean_color)
        ax.add_patch(poly_left)

        # Right rectangle patch
        rect_right_lons = [max_lon, max_lon_with_offset, max_lon_with_offset,
                           max_lon]
        rect_right_lats = [min_lat_with_offset, min_lat_with_offset,
                           max_lat_with_offset, max_lat_with_offset]
        rect_right_x, rect_right_y = map(rect_right_lons, rect_right_lats)
        rect_right_xy = zip(rect_right_x, rect_right_y)
        poly_right = Polygon(list(rect_right_xy), facecolor=ocean_color,
                             edgecolor=ocean_color)
        ax.add_patch(poly_right)

    # Draw coastlines
    if draw_coastlines:

        ax.set_facecolor(ocean_color)
        map.drawcoastlines(zorder=10)

        # map.drawstates()
        # map.drawcountries()
        # map.drawcounties()

        # Set background color
        # map.drawlsmask(land_color='Linen', ocean_color=ocean_color,
        #                lakes=True, zorder=-2)

        # map.fillcontinents(color='red', lake_color='white', zorder=0)
        map.fillcontinents(color=land_color, zorder=1)

        # map.bluemarble()
        # map.shadedrelief()
        # map.etopo()

    # Map features
    if draw_features:

        # lat and lon lines
        lon_lines = numpy.linspace(min_lon, max_lon, 2)
        lat_lines = numpy.linspace(min_lat, max_lat, 2)
        parallels = map.drawparallels(lat_lines, labels=[1, 0, 0, 0],
                                      fontsize=10, rotation=90)
        meridians = map.drawmeridians(lon_lines, labels=[0, 0, 0, 1],
                                      fontsize=10)

        # Align meridian tick label to left and right
        min_meridians, max_meridians = numpy.sort([*meridians])
        meridians[min_meridians][1][0].set_ha('left')
        meridians[max_meridians][1][0].set_ha('right')

        # Align meridian tick label to left and right
        min_parallels, max_parallels = numpy.sort([*parallels])
        parallels[min_parallels][1][0].set_va('bottom')
        parallels[max_parallels][1][0].set_va('top')

        # Draw Mapscale
        distance = 0.1 * diff_lon_on_map / 1000.0  # in Km
        distance = 5 * int(distance / 5.0 + 0.5)
        shift_lon = +0.75 * (max_lon - mid_lon)
        shift_lat = -0.80 * (max_lat - mid_lat)
        map.drawmapscale(mid_lon+shift_lon, mid_lat+shift_lat, mid_lon,
                         mid_lat, distance, barstyle='simple', units='km',
                         labelstyle='simple', fontsize='7', zorder=100)
