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
from ._plot_utilities import save_plot, plt, matplotlib, make_axes_locatable
import matplotlib.colors
from ._draw_map import draw_map

__all__ = ['plot_streamlines']


# ================
# Plot Streamlines
# ================

def plot_streamlines(
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
    Plot streamlines of the velocity vector field.
    """

    # Config
    title_fontsize = 13
    label_fontsize = 10

    # plot streamlines
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.6))
    ax[0].set_rasterization_zorder(0)
    ax[1].set_rasterization_zorder(0)

    map_1 = draw_map(ax[0], lon, lat, draw_features=True)
    map_2 = draw_map(ax[1], lon, lat, draw_features=True)

    # For streamplot, we should use the projected lat and lon in the native
    # coordinate of projection of the map
    projected_lon_grid_on_map, projected_lat_grid_on_map = \
        map_1.makegrid(U_original.shape[1], U_original.shape[0],
                       returnxy=True)[2:4]

    # These are needed for Martha's dataset, but not needed for MontereyBay
    # projected_lon_grid_on_map = projected_lon_grid_on_map[::-1, :]
    # projected_lat_grid_on_map = projected_lat_grid_on_map[::-1, :]

    vel_magnitude_original = numpy.ma.sqrt(U_original**2 + V_original**2)
    vel_magnitude_inpainted = numpy.sqrt(U_inpainted**2 + V_inpainted**2)

    line_width_original = 3 * vel_magnitude_original / \
        vel_magnitude_original.max()
    line_width_inpainted = 3 * vel_magnitude_inpainted / \
        vel_magnitude_inpainted.max()

    min_value_original = numpy.min(vel_magnitude_original)
    max_value_original = numpy.max(vel_magnitude_original)
    min_value_inpainted = numpy.min(vel_magnitude_inpainted)
    max_value_inpainted = numpy.max(vel_magnitude_inpainted)

    # min_value_original -= \
    #         (max_value_original - min_value_original) * 0.2
    # min_value_inpainted -= \
    #         (max_value_inpainted - min_value_inpainted) * 0.2

    norm_original = matplotlib.colors.Normalize(vmin=min_value_original,
                                                vmax=max_value_original)
    norm_inpainted = matplotlib.colors.Normalize(vmin=min_value_inpainted,
                                                 vmax=max_value_inpainted)

    streamplot_original = map_1.streamplot(
            projected_lon_grid_on_map, projected_lat_grid_on_map,
            U_original, V_original, color=vel_magnitude_original,
            density=5, linewidth=line_width_original, cmap=plt.cm.ocean_r,
            norm=norm_original, zorder=-10)
    streamplot_inpainted = map_2.streamplot(
            projected_lon_grid_on_map, projected_lat_grid_on_map,
            U_inpainted, V_inpainted, color=vel_magnitude_inpainted,
            density=5, linewidth=line_width_inpainted,
            cmap=plt.cm.ocean_r, norm=norm_inpainted, zorder=-10)

    # Create axes for colorbar that is the same size as the plot axes
    divider_0 = make_axes_locatable(ax[0])
    cax_0 = divider_0.append_axes("right", size="5%", pad=0.07)
    cbar_0 = plt.colorbar(streamplot_original.lines, cax=cax_0)
    cbar_0.ax.tick_params(labelsize=label_fontsize)

    divider_1 = make_axes_locatable(ax[1])
    cax_1 = divider_1.append_axes("right", size="5%", pad=0.07)
    cbar_1 = plt.colorbar(streamplot_inpainted.lines, cax=cax_1)
    cbar_1.ax.tick_params(labelsize=label_fontsize)

    ax[0].set_title('(a) Original Velocity Streamlines',
                    fontsize=title_fontsize)
    ax[1].set_title('(b) Reconstructed Velocity Streamlines',
                    fontsize=title_fontsize)

    fig.set_tight_layout(True)
    fig.patch.set_alpha(0)

    # Save plot
    if save:
        filename = 'streamline'
        save_plot(filename, transparent_background=False, pdf=True,
                  bbox_extra_artists=None, verbose=True)
