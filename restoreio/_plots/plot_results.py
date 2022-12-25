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
from ._plot_utilities import plt
from ._plot_grid import plot_grid
from ._plot_velocities import plot_velocities
from ._plot_streamlines import plot_streamlines                     # noqa F401
from ._plot_quiver import plot_quiver                               # noqa F401

__all__ = ['plot_results']


# ============
# Plot Results
# ============

def plot_results(
        lon,
        lat,
        land_indices,
        all_missing_indices,
        missing_indices_inside_hull,
        missing_indices_outside_hull,
        valid_indices,
        hull_points_coord_list,
        U_original,
        V_original,
        U_inpainted,
        V_inpainted,
        save=True):
    """
    This function is called from the main() function, but is commented. To
    plot, uncomment this function in main(). You may disable iteration through
    all time_index, and only plot for one time_index inside the main().

    Note: Inside this function, there is a nested function "draw_map()", which
    calls Basemap. If the  attribute "resolution" of the basemap is set to 'f'
    (meaning full resolution), it takes a lot time to generate the image. For
    faster rendering, set the resolution to 'i'.

    It plots 3 figures:

    Figure 1:
        Axes[0]: Plot of all valid points and all issuing points.
        Axes[1]: Plot of all valid points, missing points inside the convex
        hull, and missing points outside the convex hull.

    Figure 2:
        Axes[0]: Plot of original east velocity.
        Axes[1]: Plot of restored east velocity.

    Figure 3:
        Axes[0]: Plot of original north velocity.
        Axes[1]: Plot of restored north velocity.
    """

    # Mesh grid
    lon_grid, lat_grid = numpy.meshgrid(lon, lat)

    # All Missing points coordinates
    all_missing_lon = lon_grid[all_missing_indices[:, 0],
                               all_missing_indices[:, 1]]
    all_missing_lat = lat_grid[all_missing_indices[:, 0],
                               all_missing_indices[:, 1]]
    all_missing_points_coord = numpy.vstack((all_missing_lon,
                                             all_missing_lat)).T

    # Missing points coordinates inside hull
    missing_lon_inside_hull = lon_grid[missing_indices_inside_hull[:, 0],
                                       missing_indices_inside_hull[:, 1]]
    missing_lat_inside_hull = lat_grid[missing_indices_inside_hull[:, 0],
                                       missing_indices_inside_hull[:, 1]]
    missing_points_coord_inside_hull = numpy.vstack(
            (missing_lon_inside_hull, missing_lat_inside_hull)).T

    # Missing points coordinates outside hull
    missing_lon_outside_hull = lon_grid[missing_indices_outside_hull[:, 0],
                                        missing_indices_outside_hull[:, 1]]
    missing_lat_outside_hull = lat_grid[missing_indices_outside_hull[:, 0],
                                        missing_indices_outside_hull[:, 1]]
    missing_points_coord_outside_hull = numpy.vstack(
            (missing_lon_outside_hull, missing_lat_outside_hull)).T

    # Valid points coordinates
    valid_lons = lon_grid[valid_indices[:, 0], valid_indices[:, 1]]
    valid_lats = lat_grid[valid_indices[:, 0], valid_indices[:, 1]]
    valid_points_coord = numpy.c_[valid_lons, valid_lats]

    # Land Point Coordinates
    if numpy.any(numpy.isnan(land_indices)) is False:
        land_lons = lon_grid[land_indices[:, 0], land_indices[:, 1]]
        land_lats = lat_grid[land_indices[:, 0], land_indices[:, 1]]
        land_points_coord = numpy.c_[land_lons, land_lats]
    else:
        land_points_coord = numpy.nan

    # Plot grid of valid and missing points
    plot_grid(
            lon, lat, valid_points_coord, land_points_coord,
            all_missing_points_coord, missing_points_coord_inside_hull,
            missing_points_coord_outside_hull, hull_points_coord_list,
            save=save)

    # Plot original and inpainted velocity vector field components
    plot_velocities(
            lon, lat, lon_grid, lat_grid, valid_points_coord,
            land_points_coord, all_missing_points_coord,
            missing_points_coord_inside_hull,
            missing_points_coord_outside_hull, U_original, V_original,
            U_inpainted, V_inpainted, save=save)

    # Plot original and inpainted velocity streamlines
    plot_streamlines(
            lon, lat, lon_grid, lat_grid, valid_points_coord,
            land_points_coord, all_missing_points_coord,
            missing_points_coord_inside_hull,
            missing_points_coord_outside_hull, U_original, V_original,
            U_inpainted, V_inpainted, save=save)

    # Plot original and inpainted velocity quiver
    # plot_quiver(
    #         lon, lat, lon_grid, lat_grid, valid_points_coord,
    #         land_points_coord, all_missing_points_coord,
    #         missing_points_coord_inside_hull,
    #         missing_points_coord_outside_hull, U_original, V_original,
    #         U_inpainted, V_inpainted, save=save)

    if not save:
        plt.show()
