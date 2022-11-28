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
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ._draw_map import draw_map

# Change font family
plt.rc('font', family='serif')

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
        V_inpainted):
    """
    Streamplots
    """

    # Boundaries
    min_lon = numpy.min(lon)
    mid_lon = numpy.mean(lon)
    max_lon = numpy.max(lon)
    min_lat = numpy.min(lat)
    mid_lat = numpy.mean(lat)
    max_lat = numpy.max(lat)

    # plot streamlines
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    ax[0, 0].set_rasterization_zorder(0)
    ax[0, 1].set_rasterization_zorder(0)
    ax[1, 0].set_rasterization_zorder(0)
    ax[1, 1].set_rasterization_zorder(0)
    map_3_11 = draw_map(ax[0, 0], lon, lat, draw_features=True)
    map_3_12 = draw_map(ax[0, 1], lon, lat, draw_features=True)

    # -----------------------
    # Draw Map For StreamPlot
    # -----------------------

    def draw_map_for_stream_plot(axis):
        """
        This map does not plot coasts and ocean. This is for only plotting
        streamlines on a white background so that we can scale it and
        superimpose it on the previous plots in inkscape.
        """

        # Since we do not plot coasts, we use 'i' option for lowest
        # resolution.
        map = Basemap(
                ax=axis,
                projection='aeqd',
                llcrnrlon=min_lon,
                llcrnrlat=min_lat,
                urcrnrlon=max_lon,
                urcrnrlat=max_lat,
                area_thresh=0.1,
                lon_0=mid_lon,
                lat_0=mid_lat,
                resolution='i')

        return map

    # -------------------

    map_3_21 = draw_map_for_stream_plot(ax[1, 0])
    map_3_22 = draw_map_for_stream_plot(ax[1, 1])

    # For streamplot, we should use the projected lat and lon in the native
    # coordinate of projection of the map
    projected_lon_grid_on_map, projected_lat_grid_on_map = \
        map_3_21.makegrid(U_original.shape[1], U_original.shape[0],
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

    streamplot_original = map_3_21.streamplot(
            projected_lon_grid_on_map, projected_lat_grid_on_map,
            U_original, V_original, color=vel_magnitude_original,
            density=5, linewidth=line_width_original, cmap=plt.cm.ocean_r,
            norm=norm_original, zorder=-1)
    streamplot_inpainted = map_3_22.streamplot(
            projected_lon_grid_on_map, projected_lat_grid_on_map,
            U_inpainted, V_inpainted, color=vel_magnitude_inpainted,
            density=5, linewidth=line_width_inpainted,
            cmap=plt.cm.ocean_r, norm=norm_inpainted, zorder=-1)

    # Create axes for colorbar that is the same size as the plot axes
    divider_10 = make_axes_locatable(ax[1, 0])
    cax_10 = divider_10.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(streamplot_original.lines, cax=cax_10)

    divider_11 = make_axes_locatable(ax[1, 1])
    cax_11 = divider_11.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(streamplot_inpainted.lines, cax=cax_11)

    ax[0, 0].set_title('Original velocity streamlines')
    ax[0, 1].set_title('Restored velocity streamlines')

    # Draw Mapscale
    # Index = int(lat.size / 4)
    # x0, y0 = map_3_11(lon[0], lat[0])
    # x1, y1 = map_3_11(lon[Index], lat[0])
    # distance = (x1 - x0) / 1000 # Length of scale in Km
    distance = 40  # For Monterey Dataset
    # distance = 5  # For Martha Dataset
    map_3_11.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                          barstyle='simple', units='km',
                          labelstyle='simple', fontsize='7')
    map_3_12.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                          barstyle='simple', units='km',
                          labelstyle='simple', fontsize='7')
    map_3_21.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                          barstyle='simple', units='km',
                          labelstyle='simple', fontsize='7')
    map_3_22.drawmapscale(mid_lon, min_lat, mid_lon, mid_lat, distance,
                          barstyle='simple', units='km',
                          labelstyle='simple', fontsize='7')
