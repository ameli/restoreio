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
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Change font family
plt.rc('font', family='serif')

__all__ = ['plot_results']


# ============
# Plot Results
# ============

def plot_results(
        lon,
        lat,
        U_original,
        V_original,
        U_inpainted,
        V_inpainted,
        all_missing_indices,
        missing_indices_inside_hull,
        missing_indices_outside_hull,
        valid_indices,
        land_indices,
        hull_points_coord_list):
    """
    This function is called from the main() function, but is commented. To
    plot, uncomment this function in main(). You may disable iteration through
    all TimeIndex, and only plot for one TimeIndex inside the main().

    Note: Inside this function, there is a nested function "draw_map()", which
    calls Basemap. If the  attribute "resolution" of the basemap is set to 'f'
    (meaning full resolution), it takes alot time to generate the image. For
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
        land_point_coord = numpy.c_[land_lons, land_lats]
    else:
        land_point_coord = numpy.nan

    # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
    percent = 0.05   # For Monterey Dataset
    # percent = 0.1     # For Martha Dataset
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

    # --------
    # Draw map
    # --------

    def draw_map(axis):

        # Basemap (set resolution to 'i' for faster rasterization and 'f' for
        # full resolution but very slow.)
        map = Basemap(
                ax=axis,
                projection='aeqd',
                llcrnrlon=min_lon_with_offset,
                llcrnrlat=min_lat_with_offset,
                urcrnrlon=max_lon_with_offset,
                urcrnrlat=max_lat_with_offset,
                area_thresh=0.1,
                lon_0=mid_lon,
                lat_0=mid_lat,
                resolution='i')

        # Map features
        map.drawcoastlines()
        # map.drawstates()
        # map.drawcountries()
        # map.drawcounties()
        map.drawlsmask(land_color='Linen', ocean_color='#C7DCEF', lakes=True,
                       zorder=-2)
        # map.fillcontinents(color='red', lake_color='white', zorder=0)
        map.fillcontinents(color='moccasin', zorder=-1)

        # map.bluemarble()
        # map.shadedrelief()
        # map.etopo()

        # lat and lon lines
        lon_lines = numpy.linspace(numpy.min(lon), numpy.max(lon), 2)
        lat_lines = numpy.linspace(numpy.min(lat), numpy.max(lat), 2)
        map.drawparallels(lat_lines, labels=[1, 0, 0, 0], fontsize=10)
        map.drawmeridians(lon_lines, labels=[0, 0, 0, 1], fontsize=10)

        return map

    # ---------------------
    # Fig 1: Missing points
    # ---------------------

    def plot_1():
        """
        Missing points
        """

        fig_1, axes_1 = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        axes_1[0].set_aspect('equal')
        axes_1[1].set_aspect('equal')
        map_11 = draw_map(axes_1[0])
        map_12 = draw_map(axes_1[1])

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
        if numpy.any(numpy.isnan(land_point_coord)) is False:
            land_points_coord_X, land_points_coord_Y = map_11(
                    land_point_coord[:, 0], land_point_coord[:, 1])

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
        if numpy.any(numpy.isnan(land_point_coord)) is False:
            map_11.plot(land_points_coord_X, land_points_coord_Y, 'o',
                        markerfacecolor='red', markeredgecolor='red',
                        markersize=marker_size)
            # map_11.plot(land_points_coord_X, land_points_coord_Y, 'o',
            #             markerfacecolor='red', markersize=marker_size)
        plt.suptitle('All unavailable points')

        # Plot all hulls boundary polygons
        num_ghull_polygons = len(hull_points_coord_list)
        hull_polygons = [None] * num_ghull_polygons
        for i in range(num_ghull_polygons):
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
            plt.gca().add_patch(hull_polygons[i])

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
        if numpy.any(numpy.isnan(land_point_coord)) is False:
            map_12.plot(land_points_coord_X, land_points_coord_Y, 'o',
                        markerfacecolor='sandybrown',
                        markeredgecolor='sandybrown', markersize=marker_size)
            # map_12.plot(land_points_coord_X, land_points_coord_Y, 'o',
            #             markerfacecolor='sandybrown', markersize=marker_size)
        plt.suptitle('Missing data inside convex hull')

        # another hull polygon on the top layer of all plots without facecolor.
        hull_polygons_2 = [None] * num_ghull_polygons
        for i in range(num_ghull_polygons):
            hull_points_X, hull_points_Y = map_11(
                    hull_points_coord_list[i][:, 0],
                    hull_points_coord_list[i][:, 1])
            hull_points_XY = numpy.vstack(
                    (hull_points_X, hull_points_Y)).T.tolist()
            hull_polygons_2[i] = Polygon(hull_points_XY, facecolor='none',
                                         edgecolor='black', alpha=0.6,
                                         closed=True, linewidth=2)
            plt.gca().add_patch(hull_polygons_2[i])

    plot_1()

    # -----------------
    # Fig 2: Velocities
    # -----------------

    def plot_2():
        """
        U and V velocities
        """

        fig_2, axes_2 = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        axes_2[0, 0].set_rasterization_zorder(0)
        axes_2[0, 1].set_rasterization_zorder(0)
        axes_2[1, 0].set_rasterization_zorder(0)
        axes_2[1, 1].set_rasterization_zorder(0)
        map_2_11 = draw_map(axes_2[0, 0])
        map_2_12 = draw_map(axes_2[0, 1])
        map_2_21 = draw_map(axes_2[1, 0])
        map_2_22 = draw_map(axes_2[1, 1])

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
        divider_00 = make_axes_locatable(axes_2[0, 0])
        cax_00 = divider_00.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(contour_U_original, cax=cax_00)

        divider_01 = make_axes_locatable(axes_2[0, 1])
        cax_01 = divider_01.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(contour_U_inpainted, cax=cax_01)

        divider_10 = make_axes_locatable(axes_2[1, 0])
        cax_10 = divider_10.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(contour_V_original, cax=cax_10)

        divider_11 = make_axes_locatable(axes_2[1, 1])
        cax_11 = divider_11.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(contour_V_inpainted, cax=cax_11)

        axes_2[0, 0].set_title('Original East velocity')
        axes_2[0, 1].set_title('Restored East velocity')
        axes_2[1, 0].set_title('Original North velocity')
        axes_2[1, 1].set_title('Restored North velocity')

    plot_2()

    # -----------------
    # Fig 3: Streamplot
    # -----------------

    def plot_3():
        """
        Streamplots
        """

        # plot streamlines
        fig_3, axes_3 = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        axes_3[0, 0].set_rasterization_zorder(0)
        axes_3[0, 1].set_rasterization_zorder(0)
        axes_3[1, 0].set_rasterization_zorder(0)
        axes_3[1, 1].set_rasterization_zorder(0)
        map_3_11 = draw_map(axes_3[0, 0])
        map_3_12 = draw_map(axes_3[0, 1])

        # -----------------------
        # Draw Map For StreamPlot
        # -----------------------

        def draw_map_for_stream_plot(axis):
            """
            This map does not plot coasts and ocean. This is for only plotting
            streamlines on a white background so that we can scale it and
            superimpose it on the previous plots in inkscape.
            """

            min_lon = numpy.min(lon)
            mid_lon = numpy.mean(lon)
            max_lon = numpy.max(lon)
            min_lat = numpy.min(lat)
            mid_lat = numpy.mean(lat)
            max_lat = numpy.max(lat)

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

        map_3_21 = draw_map_for_stream_plot(axes_3[1, 0])
        map_3_22 = draw_map_for_stream_plot(axes_3[1, 1])

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
        divider_10 = make_axes_locatable(axes_3[1, 0])
        cax_10 = divider_10.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(streamplot_original.lines, cax=cax_10)

        divider_11 = make_axes_locatable(axes_3[1, 1])
        cax_11 = divider_11.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(streamplot_inpainted.lines, cax=cax_11)

        axes_3[0, 0].set_title('Original velocity streamlines')
        axes_3[0, 1].set_title('Restored velocity streamlines')

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

    plot_3()

    # -------------
    # Fig 4: Quiver
    # -------------

    def plot_4():
        """
        Quivers
        """

        # Plot quiver of u and v
        fig_4, axes_4 = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        map_41 = draw_map(axes_4[0])
        map_42 = draw_map(axes_4[1])
        lon_grid_on_map, lat_grid_on_map = map_41(lon_grid, lat_grid)
        vel_magnitude_original = numpy.ma.sqrt(U_original**2 + V_original**2)
        vel_magnitude_inpainted = numpy.sqrt(U_inpainted**2 + V_inpainted**2)
        map_41.quiver(lon_grid_on_map, lat_grid_on_map, U_original, V_original,
                      vel_magnitude_original, scale=1000, scale_units='inches')
        map_42.quiver(lon_grid_on_map, lat_grid_on_map, U_inpainted,
                      V_inpainted, vel_magnitude_inpainted, scale=1000,
                      scale_units='inches')
        axes_4[0].set_title('Original velocity vector field')
        axes_4[1].set_title('Restored velocity vector field')

    # plot_4()

    plt.show()
