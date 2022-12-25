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
from ._plot_utilities import save_plot, plt
from ._draw_map import draw_map

__all__ = ['plot_quiver']


# ===========
# plot quiver
# ===========

def plot_quiver(
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
    Plot velocity vector quiver.
    """

    # Config
    title_fontsize = 13

    # Plot quiver of u and v
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.6))

    map_1 = draw_map(ax[0], lon, lat, draw_features=True)
    map_2 = draw_map(ax[1], lon, lat, draw_features=True)

    lon_grid_on_map, lat_grid_on_map = map_1(lon_grid, lat_grid)
    vel_magnitude_original = numpy.ma.sqrt(U_original**2 + V_original**2)
    vel_magnitude_inpainted = numpy.sqrt(U_inpainted**2 + V_inpainted**2)
    map_1.quiver(lon_grid_on_map, lat_grid_on_map, U_original, V_original,
                 vel_magnitude_original, scale=1000, scale_units='inches')
    map_2.quiver(lon_grid_on_map, lat_grid_on_map, U_inpainted,
                 V_inpainted, vel_magnitude_inpainted, scale=1000,
                 scale_units='inches')
    ax[0].set_title('(a) Original Velocity Vector Field',
                    fontsize=title_fontsize)
    ax[1].set_title('(b) Reconstructed Velocity Vector Field',
                    fontsize=title_fontsize)

    fig.set_tight_layout(True)
    fig.patch.set_alpha(0)

    # Save plot
    if save:
        filename = 'quiver'
        save_plot(filename, transparent_background=False, pdf=True,
                  bbox_extra_artists=None, verbose=True)
