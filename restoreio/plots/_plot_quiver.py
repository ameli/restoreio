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
        V_inpainted):
    """
    Plot velocity vector quiver.
    """

    # Plot quiver of u and v
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    map_41 = draw_map(ax[0], lon, lat, draw_features=True)
    map_42 = draw_map(ax[1], lon, lat, draw_features=True)

    lon_grid_on_map, lat_grid_on_map = map_41(lon_grid, lat_grid)
    vel_magnitude_original = numpy.ma.sqrt(U_original**2 + V_original**2)
    vel_magnitude_inpainted = numpy.sqrt(U_inpainted**2 + V_inpainted**2)
    map_41.quiver(lon_grid_on_map, lat_grid_on_map, U_original, V_original,
                  vel_magnitude_original, scale=1000, scale_units='inches')
    map_42.quiver(lon_grid_on_map, lat_grid_on_map, U_inpainted,
                  V_inpainted, vel_magnitude_inpainted, scale=1000,
                  scale_units='inches')
    ax[0].set_title('Original velocity vector field')
    ax[1].set_title('Restored velocity vector field')

    fig.set_tight_layout(True)
