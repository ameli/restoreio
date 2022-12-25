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
from ._plot_utilities import load_plot_settings, save_plot, plt
from matplotlib.patches import Polygon
from ._draw_map import draw_map

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
        hull_points_coord_list,
        save=True):
    """
    Plot grid consisting of missing points, valid points, land points, etc.
    """

    load_plot_settings()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.6))
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    map_1 = draw_map(ax[0], lon, lat, draw_features=True)
    map_2 = draw_map(ax[1], lon, lat, draw_features=True)

    # Get Map coordinates (Valid points, missing points inside and outside
    # hull, land points)
    valid_points_coord_X, valid_points_coord_Y = map_1(
            valid_points_coord[:, 0], valid_points_coord[:, 1])
    all_missing_points_coord_X, all_missing_points_coord_Y = map_1(
            all_missing_points_coord[:, 0], all_missing_points_coord[:, 1])
    missing_points_coord_inside_hull_X, \
        missing_points_coord_inside_hull_Y = \
        map_1(missing_points_coord_inside_hull[:, 0],
              missing_points_coord_inside_hull[:, 1])
    missing_points_coord_outside_hull_X, \
        missing_points_coord_outside_hull_Y = \
        map_1(missing_points_coord_outside_hull[:, 0],
              missing_points_coord_outside_hull[:, 1])
    if bool(numpy.any(numpy.isnan(land_points_coord))) is False:
        land_points_coord_X, land_points_coord_Y = map_1(
                land_points_coord[:, 0], land_points_coord[:, 1])

    # Config
    title_fontsize = 13
    marker_size = 4
    valid_color = '#81DD81'
    missing_color = 'red'
    ocean_color = 'royalblue'
    land_color = 'sandybrown'
    hull_color = 'honeydew'

    # Plot All missing points
    map_1.plot(valid_points_coord_X, valid_points_coord_Y, 'o',
               markerfacecolor=valid_color, markeredgecolor=valid_color,
               markersize=marker_size, label='Known data')
    map_1.plot(all_missing_points_coord_X, all_missing_points_coord_Y,
               'o', markerfacecolor=missing_color,
               markeredgecolor=missing_color, markersize=marker_size,
               label='Unknown data')
    if bool(numpy.any(numpy.isnan(land_points_coord))) is False:
        map_1.plot(land_points_coord_X, land_points_coord_Y, 'o',
                   markerfacecolor=missing_color,
                   markeredgecolor=missing_color, markersize=marker_size,
                   label='Land overlap')
    ax[0].set_title('(a) Original Grid of Known and Unknown Data Points',
                    fontsize=title_fontsize)
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')

    # Plot Hull and missing inside/outside the hull, and land points
    map_2.plot(valid_points_coord_X, valid_points_coord_Y, 'o',
               markerfacecolor=valid_color, markeredgecolor=valid_color,
               markersize=marker_size, label='Known data')
    map_2.plot(missing_points_coord_inside_hull_X,
               missing_points_coord_inside_hull_Y, 'o',
               markerfacecolor=missing_color, markeredgecolor=missing_color,
               markersize=marker_size, label='Unknown (inside domain)')
    map_2.plot(missing_points_coord_outside_hull_X,
               missing_points_coord_outside_hull_Y, 'o',
               markerfacecolor=ocean_color, markeredgecolor=ocean_color,
               markersize=marker_size, label='Ocean (outside domain)')
    if bool(numpy.any(numpy.isnan(land_points_coord))) is False:
        map_2.plot(land_points_coord_X, land_points_coord_Y, 'o',
                   markerfacecolor=land_color, markeredgecolor=land_color,
                   markersize=marker_size, label='Land')

    # Plot all hulls boundary polygons
    num_hull_polygons = len(hull_points_coord_list)   # with face color
    hull_polygons_2 = [None] * num_hull_polygons      # without face color
    hull_polygons = [None] * num_hull_polygons
    for i in range(num_hull_polygons):
        hull_points_X, hull_points_Y = map_1(
                hull_points_coord_list[i][:, 0],
                hull_points_coord_list[i][:, 1])
        hull_points_XY = numpy.vstack(
                (hull_points_X, hull_points_Y)).T.tolist()

        hull_polygons[i] = Polygon(hull_points_XY, facecolor=hull_color,
                                   edgecolor='none', alpha=0.6,
                                   closed=True, linewidth=2)
        hull_polygons_2[i] = Polygon(hull_points_XY, facecolor='none',
                                     edgecolor='black', alpha=0.6,
                                     closed=True, linewidth=2)
        p0 = ax[1].add_patch(hull_polygons[i])
        p1 = ax[1].add_patch(hull_polygons_2[i])                    # noqa F401

        if i == 0:
            p0.set_label('Data Domain')

    ax[1].set_title('(b) Differentiate Unknown, Ocean, and Land Points',
                    fontsize=title_fontsize)
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')

    fig.set_tight_layout(True)
    fig.patch.set_alpha(0)

    # Save plot
    if save:
        filename = 'grid'
        save_plot(filename, transparent_background=False, pdf=True,
                  bbox_extra_artists=None, verbose=True)
