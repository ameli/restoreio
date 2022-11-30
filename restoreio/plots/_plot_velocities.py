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

from ._plot_utilities import save_plot, plt, make_axes_locatable
from matplotlib import cm
from ._draw_map import draw_map

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
        V_inpainted,
        save=True):
    """
    plot U and V velocities.
    """

    # Config
    title_fontsize = 13
    label_fontsize = 10

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 9.2))
    ax[0, 0].set_rasterization_zorder(0)
    ax[0, 1].set_rasterization_zorder(0)
    ax[1, 0].set_rasterization_zorder(0)
    ax[1, 1].set_rasterization_zorder(0)
    map_11 = draw_map(ax[0, 0], lon, lat, draw_features=True)
    map_12 = draw_map(ax[0, 1], lon, lat, draw_features=True)
    map_21 = draw_map(ax[1, 0], lon, lat, draw_features=True)
    map_22 = draw_map(ax[1, 1], lon, lat, draw_features=True)

    contour_levels = 300

    lon_grid_on_map, lat_grid_on_map = map_11(lon_grid, lat_grid)
    contour_U_original = map_11.contourf(
            lon_grid_on_map, lat_grid_on_map, U_original, contour_levels,
            corner_mask=False, cmap=cm.jet, zorder=-1, rasterized=True)
    contour_U_inpainted = map_12.contourf(
            lon_grid_on_map, lat_grid_on_map, U_inpainted, contour_levels,
            corner_mask=False, cmap=cm.jet, zorder=-1, rasterized=True)
    contour_V_original = map_21.contourf(
            lon_grid_on_map, lat_grid_on_map, V_original, contour_levels,
            corner_mask=False, cmap=cm.jet, zorder=-1, rasterized=True)
    contour_V_inpainted = map_22.contourf(
            lon_grid_on_map, lat_grid_on_map, V_inpainted, contour_levels,
            corner_mask=False, cmap=cm.jet, zorder=-1)

    # Colorbars
    divider_00 = make_axes_locatable(ax[0, 0])
    cax_00 = divider_00.append_axes("right", size="5%", pad=0.07)
    cbar_00 = plt.colorbar(contour_U_original, cax=cax_00)
    cbar_00.ax.tick_params(labelsize=label_fontsize)

    divider_01 = make_axes_locatable(ax[0, 1])
    cax_01 = divider_01.append_axes("right", size="5%", pad=0.07)
    cbar_01 = plt.colorbar(contour_U_inpainted, cax=cax_01)
    cbar_01.ax.tick_params(labelsize=label_fontsize)

    divider_10 = make_axes_locatable(ax[1, 0])
    cax_10 = divider_10.append_axes("right", size="5%", pad=0.07)
    cbar_10 = plt.colorbar(contour_V_original, cax=cax_10)
    cbar_10.ax.tick_params(labelsize=label_fontsize)

    divider_11 = make_axes_locatable(ax[1, 1])
    cax_11 = divider_11.append_axes("right", size="5%", pad=0.07)
    cbar_11 = plt.colorbar(contour_V_inpainted, cax=cax_11)
    cbar_11.ax.tick_params(labelsize=label_fontsize)

    ax[0, 0].set_title('(a) Original East Velocity', fontsize=title_fontsize)
    ax[0, 1].set_title('(b) Reconstructed East Velocity',
                       fontsize=title_fontsize)
    ax[1, 0].set_title('(c) Original North Velocity', fontsize=title_fontsize)
    ax[1, 1].set_title('(d) Reconstructed North Velocity',
                       fontsize=title_fontsize)

    fig.set_tight_layout(True)
    fig.patch.set_alpha(0)

    # Save plot
    if save:
        filename = 'velocities'
        save_plot(filename, transparent_background=False, pdf=True,
                  bbox_extra_artists=None, verbose=True)
