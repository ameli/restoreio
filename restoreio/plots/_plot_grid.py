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

# 2021/05/20. I added this line fix the error: KeyError: 'PROJ_LIB'
# import os
# PROJ_LIB = '/opt/miniconda3/share/proj'
# if not os.path.isdir(PROJ_LIB):
#     raise FileNotFoundError('The directory %s does not exists.' % PROJ_LIB)
# os.environ['PROJ_LIB'] = PROJ_LIB

import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from ._draw_map import draw_map

# Change font family
plt.rc('font', family='serif')

__all__ = ['plot_grid']


# =========
# Plot Grid
# =========

def plot_grid(
        lon,
        lat,
        valid_points_coord,
        land_points_coord,
        all_missing_points_coord,
        missing_points_coord_inside_hull,
        missing_points_coord_outside_hull,
        hull_points_coord_list):
    """
    Plot grid consisting of missing points, valid points, land points, etc.
    """

    # Domain bounds
    mid_lon = numpy.mean(lon)
    min_lat = numpy.min(lat)
    mid_lat = numpy.mean(lat)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    map_11 = draw_map(ax[0], lon, lat, draw_features=True)
    map_12 = draw_map(ax[1], lon, lat, draw_features=True)

    # Draw Mapscale
    # Index = int(lat.size / 4)
    # x0, y0 = map_11(lon[0], lat[0])
    # x1, y1 = map_11(lon[Index], lat[0])
    # distance = (x1 - x0) / 1000 # Length of scale in Km
    distance = 40  # For Monterey Dataset
    # distance = 5  # For Martha Dataset
    map_11.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                        barstyle='simple', units='km', labelstyle='simple',
                        fontsize='7')
    map_12.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                        barstyle='simple', units='km', labelstyle='simple',
                        fontsize='7')

    # Get Map coordinates (Valid points, missing points inside and outside
    # hull, land points)
    valid_points_coord_X, valid_points_coord_Y = map_11(
            valid_points_coord[:, 0], valid_points_coord[:, 1])
    all_missing_points_coord_X, all_missing_points_coord_Y = map_11(
            all_missing_points_coord[:, 0], all_missing_points_coord[:, 1])
    missing_points_coord_inside_hull_X, \
        missing_points_coord_inside_hull_Y = \
        map_11(missing_points_coord_inside_hull[:, 0],
               missing_points_coord_inside_hull[:, 1])
    missing_points_coord_outside_hull_X, \
        missing_points_coord_outside_hull_Y = \
        map_11(missing_points_coord_outside_hull[:, 0],
               missing_points_coord_outside_hull[:, 1])
    if bool(numpy.any(numpy.isnan(land_points_coord))) is False:
        land_points_coord_X, land_points_coord_Y = map_11(
                land_points_coord[:, 0], land_points_coord[:, 1])

    # Plot All missing points
    marker_size = 4
    map_11.plot(valid_points_coord_X, valid_points_coord_Y, 'o',
                markerfacecolor='lightgreen', markeredgecolor='lightgreen',
                markersize=marker_size)
    # map_11.plot(valid_points_coord_X, valid_points_coord_Y, 'o',
    #             markerfacecolor='lightgreen', markersize=marker_size)
    map_11.plot(all_missing_points_coord_X, all_missing_points_coord_Y,
                'o', markerfacecolor='red', markeredgecolor='red',
                markersize=marker_size)
    # map_11.plot(all_missing_points_coord_X, all_missing_points_coord_Y,
    #             'o', markerfacecolor='red', markersize=marker_size)
    if bool(numpy.any(numpy.isnan(land_points_coord))) is False:
        map_11.plot(land_points_coord_X, land_points_coord_Y, 'o',
                    markerfacecolor='red', markeredgecolor='red',
                    markersize=marker_size)
        # map_11.plot(land_points_coord_X, land_points_coord_Y, 'o',
        #             markerfacecolor='red', markersize=marker_size)
    fig.suptitle('All unavailable points')

    # Plot all hulls boundary polygons
    num_hull_polygons = len(hull_points_coord_list)
    hull_polygons = [None] * num_hull_polygons
    for i in range(num_hull_polygons):
        hull_points_X, hull_points_Y = map_11(
                hull_points_coord_list[i][:, 0],
                hull_points_coord_list[i][:, 1])
        hull_points_XY = numpy.vstack(
                (hull_points_X, hull_points_Y)).T.tolist()
        # hull_polygons[i] = Polygon(hull_points_XY,
        #                            facecolor='lightgoldenrodyellow',
        #                            alpha=0.6, closed=True, linewidth=1)
        hull_polygons[i] = Polygon(hull_points_XY, facecolor='honeydew',
                                   edgecolor='none', alpha=0.6,
                                   closed=True, linewidth=2)
        ax[0].add_patch(hull_polygons[i])

    # Plot Hull and missing inside/outside the hull, and land points
    map_12.plot(valid_points_coord_X, valid_points_coord_Y, 'o',
                markerfacecolor='lightgreen', markeredgecolor='lightgreen',
                markersize=marker_size)
    # map_12.plot(valid_points_coord_X, valid_points_coord_Y, 'o',
    #             markerfacecolor='lightgreen', markersize=marker_size)
    map_12.plot(missing_points_coord_inside_hull_X,
                missing_points_coord_inside_hull_Y, 'o',
                markerfacecolor='red', markeredgecolor='red',
                markersize=marker_size)
    # map_12.plot(missing_points_coord_inside_hull_X,
    #             missing_points_coord_inside_hull_Y, 'o',
    #             markerfacecolor='red', markersize=marker_size)
    map_12.plot(missing_points_coord_outside_hull_X,
                missing_points_coord_outside_hull_Y, 'o',
                markerfacecolor='royalblue', markeredgecolor='royalblue',
                markersize=marker_size)
    # map_12.plot(missing_points_coord_outside_hull_X,
    #             missing_points_coord_outside_hull_Y, 'o',
    #             markerfacecolor='royalblue', markersize=marker_size)
    if bool(numpy.any(numpy.isnan(land_points_coord))) is False:
        map_12.plot(land_points_coord_X, land_points_coord_Y, 'o',
                    markerfacecolor='sandybrown',
                    markeredgecolor='sandybrown', markersize=marker_size)
        # map_12.plot(land_points_coord_X, land_points_coord_Y, 'o',
        #             markerfacecolor='sandybrown', markersize=marker_size)
    fig.suptitle('Missing data inside convex hull')

    # another hull polygon on the top layer of all plots without facecolor.
    hull_polygons_2 = [None] * num_hull_polygons
    for i in range(num_hull_polygons):
        hull_points_X, hull_points_Y = map_11(
                hull_points_coord_list[i][:, 0],
                hull_points_coord_list[i][:, 1])
        hull_points_XY = numpy.vstack(
                (hull_points_X, hull_points_Y)).T.tolist()
        hull_polygons_2[i] = Polygon(hull_points_XY, facecolor='none',
                                     edgecolor='black', alpha=0.6,
                                     closed=True, linewidth=2)
        ax[0].add_patch(hull_polygons_2[i])

    # Configurations
    ax[0].tick_params(axis='y', labelrotation=45)
    ax[1].tick_params(axis='y', labelrotation=45)
