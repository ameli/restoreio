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
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ._draw_map import draw_map

# Change font family
plt.rc('font', family='serif')

__all__ = ['plot_velocities']


# ===============
# Plot Velocities
# ===============

def plot_velocities(
        lon,
        lat,
        lon_grid,
        lat_grid,
        valid_points_coord,
        land_points_coord,
        all_missing_points_coord,
        missing_points_coord_inside_hull,
        missing_points_coord_outside_hull,
        U_original,
        V_original,
        U_inpainted,
        V_inpainted):
    """
    plot U and V velocities.
    """

    # Domain bounds
    mid_lon = numpy.mean(lon)
    min_lat = numpy.min(lat)
    mid_lat = numpy.mean(lat)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    ax[0, 0].set_rasterization_zorder(0)
    ax[0, 1].set_rasterization_zorder(0)
    ax[1, 0].set_rasterization_zorder(0)
    ax[1, 1].set_rasterization_zorder(0)
    map_2_11 = draw_map(ax[0, 0], lon, lat, draw_features=True)
    map_2_12 = draw_map(ax[0, 1], lon, lat, draw_features=True)
    map_2_21 = draw_map(ax[1, 0], lon, lat, draw_features=True)
    map_2_22 = draw_map(ax[1, 1], lon, lat, draw_features=True)

    # Draw Mapscale
    # Index = int(lat.size / 4)
    # x0, y0 = map_2_11(lon[0], lat[0])
    # x1, y1 = map_2_11(lon[Index], lat[0])
    # distance = (x1 - x0) / 1000 # Length of scale in Km
    # distance = 40  # For Monterey Dataset
    distance = 5  # For Martha Dataset
    map_2_11.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                          barstyle='simple', units='km',
                          labelstyle='simple', fontsize='7')
    map_2_12.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                          barstyle='simple', units='km',
                          labelstyle='simple', fontsize='7')
    map_2_21.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                          barstyle='simple', units='km',
                          labelstyle='simple', fontsize='7')
    map_2_22.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                          barstyle='simple', units='km',
                          labelstyle='simple', fontsize='7')

    contour_levels = 300

    lon_grid_on_map, lat_grid_on_map = map_2_11(lon_grid, lat_grid)
    contour_U_original = map_2_11.contourf(
            lon_grid_on_map, lat_grid_on_map, U_original, contour_levels,
            corner_mask=False, cmap=cm.jet, zorder=-1, rasterized=True)
    contour_U_inpainted = map_2_12.contourf(
            lon_grid_on_map, lat_grid_on_map, U_inpainted, contour_levels,
            corner_mask=False, cmap=cm.jet, zorder=-1, rasterized=True)
    contour_V_original = map_2_21.contourf(
            lon_grid_on_map, lat_grid_on_map, V_original, contour_levels,
            corner_mask=False, cmap=cm.jet, zorder=-1, rasterized=True)
    contour_V_inpainted = map_2_22.contourf(
            lon_grid_on_map, lat_grid_on_map, V_inpainted, contour_levels,
            corner_mask=False, cmap=cm.jet, zorder=-1)

    # Colorbars
    divider_00 = make_axes_locatable(ax[0, 0])
    cax_00 = divider_00.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(contour_U_original, cax=cax_00)

    divider_01 = make_axes_locatable(ax[0, 1])
    cax_01 = divider_01.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(contour_U_inpainted, cax=cax_01)

    divider_10 = make_axes_locatable(ax[1, 0])
    cax_10 = divider_10.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(contour_V_original, cax=cax_10)

    divider_11 = make_axes_locatable(ax[1, 1])
    cax_11 = divider_11.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(contour_V_inpainted, cax=cax_11)

    ax[0, 0].set_title('Original East velocity')
    ax[0, 1].set_title('Restored East velocity')
    ax[1, 0].set_title('Original North velocity')
    ax[1, 1].set_title('Restored North velocity')
