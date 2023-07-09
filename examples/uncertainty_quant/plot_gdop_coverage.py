#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2016, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

"""
Plotting geometric location of two radars, gdop, etc.
"""

# =======
# Imports
# =======

import sys
import numpy
import netCDF4
from _utils._plot_utils._plot_utilities import plt, make_axes_locatable, \
        save_plot, load_plot_settings
from matplotlib.colors import ListedColormap
import matplotlib.ticker
from _utils._plot_utils._draw_map import draw_map
from _utils._load_variables import get_datetime_info
from _utils._subset import subset_domain, subset_datetime


# ================
# Compute Coverage
# ================

def _compute_coverage(u):
    """
    Computes the average of number of non-mask time frames at each location.
    The output is the percentage of coverage.
    """

    start_time = 0
    end_time = u.shape[0]-1
    counter = 0
    coverage = numpy.zeros((u.shape[1], u.shape[2]), dtype=float)
    for i in range(start_time, end_time+1):
        mask = numpy.isnan(u[i, :, :])
        coverage += 1.0-mask.astype(int)
        counter += 1
    coverage *= 100.0/float(counter)

    return coverage


# ============
# Compute GDOP
# ============

def _compute_gdop(xx, yy, site_x, site_y, site_normals1, site_normals2,
                  site_codes):
    """
    Computes east, north, and east/north component of the GDOP symmetric
    matrix, as well as the total GDOP.

    Note that the component of the GDOP matrix are in the form of squares
    (like variance). We take the square-root of the east and north component
    since they are always positive. However, for the east/north component,
    we output the GDOP as is in the matrix.
    """

    gdop = numpy.ma.masked_all(xx.shape+(2, 2), dtype=float)
    total_gdop = numpy.ma.masked_all(xx.shape, dtype=float)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):

            # Iterate over sites
            N_list = []
            for k in range(len(site_codes)):

                n1 = xx[i, j] - site_x[k]
                n2 = yy[i, j] - site_y[k]

                n = numpy.array([n1, n2])
                n = n / numpy.sqrt(n1**2+n2**2)

                N_list.append(n)

            # if len(N_list) < 2:
                # break

            N = numpy.array(N_list)

            NN_list = []
            for k in range(N.shape[0]):
                if (numpy.dot(N[k, :], site_normals1[k]) > 0.0) and \
                   (numpy.dot(N[k, :], site_normals2[k]) > 0.0):
                    NN_list.append(N[k, :])

            NN = numpy.array(NN_list)
            min_num_acceptable_sites = 2
            if NN.shape[0] >= min_num_acceptable_sites:

                # Use all sites without considering their angles of views
                gdop2 = numpy.linalg.inv(numpy.dot(N.T, N))

                # Use selected sites within their angles of views
                # gdop2 = numpy.linalg.inv(numpy.dot(NN.T, NN))

                gdop2[gdop2 > 200] = numpy.ma.masked

                gdop[i, j, 0, 0] = numpy.sqrt(gdop2[0, 0])  # Using square root
                gdop[i, j, 1, 1] = numpy.sqrt(gdop2[1, 1])  # Using square root
                gdop[i, j, 0, 1] = gdop2[0, 1]
                gdop[i, j, 1, 0] = gdop2[1, 0]
                total_gdop[i, j] = numpy.sqrt(gdop2[0, 0] + gdop2[1, 1])

    return gdop, total_gdop


# ================
# Normalize Vector
# ================

def _normalize_vector(nx, ny):
    """
    Normalizes the norm of a vector.
    """

    return numpy.array([nx, ny]) / numpy.sqrt(nx**2+ny**2)


# ==========
# Snap Array
# ==========

def _snap_array(min, max, snap):
    """
    Creates an array that increments every "snap" value, in between min and
    max. For instance, if min=0.05, max=0.62, and snap=0.2, the output array
    is [0.2, 0.4, 0.6].
    """

    snap_min = numpy.ceil(min / snap) * snap
    snap_max = numpy.floor(max / snap) * snap
    array = numpy.arange(snap_min, snap_max+1e-8, snap)

    return array


# ====================
# Plot Contour On Axis
# ====================

def _plot_contour_on_axis(ax, map, lons_grid_on_map, lats_grid_on_map,
                          scalar_field, title, colormap, threshold,
                          remove_contours=None):

    # Config
    title_fontsize = 11
    tick_fontsize = 10
    label_fontsize = 8

    # resterization. Anything with zorder less than 0 will be rasterized.
    ax.set_rasterization_zorder(0)

    scalar_field[numpy.isnan(scalar_field)] = threshold
    # scalar_field[scalar_field > threshold] = threshold

    # Plot contour
    contourf_level = numpy.linspace(
            numpy.min(scalar_field), threshold, 300).tolist()

    draw = map.contourf(lons_grid_on_map, lats_grid_on_map, scalar_field,
                        contourf_level, cmap=colormap, rasterized=True,
                        zorder=-1, extend='max')

    # Contour levels
    contour_level = numpy.linspace(-1, 0.85, 38).tolist() + [0.95, 1.05]
    if remove_contours is not None:
        for i in remove_contours:
            if i in contour_level:
                contour_level.remove(i)

    cs = map.contour(lons_grid_on_map, lats_grid_on_map, scalar_field,
                     contour_level, colors='black', linewidths=1, zorder=1)
    plt.clabel(cs, inline=1, fontsize=label_fontsize, zorder=1)
    for c in cs.collections:
        c.set_rasterized(False)
    draw.set_clim(vmin=numpy.min(scalar_field), vmax=threshold+1e-1)

    # Create ax for colorbar that is the same size as the plot ax
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)

    # Colorbar
    # cb = plt.colorbar(draw, cax=cax, ticks=numpy.linspace(-1, 2.4, 18))
    cb = plt.colorbar(draw, cax=cax)

    ticks = _snap_array(numpy.min(scalar_field), threshold, 0.15)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=tick_fontsize)
    cb.solids.set_rasterized(True)

    # ax labels
    ax.set_title(title, fontsize=title_fontsize)

    # Background blue for ocean
    ax.set_facecolor('#C7DCEF')


# =======================
# Plot Pcolormesh On Axis
# =======================

def _plot_pcolormesh_on_axis(ax, map, lons_grid_on_map, lats_grid_on_map,
                             scalar_field, title, colormap):

    # Config
    title_fontsize = 11
    tick_fontsize = 10

    # resterization. Anything with zorder less than 0 will be rasterized.
    # ax.set_rasterization_zorder(0)

    # Transparent colormap
    transparent_colormap = colormap(numpy.arange(colormap.N))
    transparent_colormap[:, -1] = numpy.linspace(0, 1, colormap.N)
    transparent_colormap = ListedColormap(transparent_colormap)

    # Plot pcolormesh
    draw = map.pcolormesh(lons_grid_on_map, lats_grid_on_map, scalar_field,
                          cmap=transparent_colormap, rasterized=True,
                          zorder=-1, vmin=0, vmax=100)

    # Create ax for colorbar that is the same size as the plot ax
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)

    # Colorbar
    cb = plt.colorbar(draw, cax=cax)
    # cb.ax.set_ylabel('Percent')
    cb.solids.set_edgecolor("face")

    ticks = numpy.linspace(0, 100, 6)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=tick_fontsize)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    cb.solids.set_rasterized(True)

    # ax labels
    ax.set_title(title, fontsize=title_fontsize)

    # Background blue for ocean
    ax.set_facecolor('#C7DCEF')


# ====
# Main
# ====

def main(argv):

    # Lon and Lat of radar sites from https://cordc.ucsd.edu/projects/hfrnet
    # and https://www.cencoos.org/observations/sensor-platforms/hf-radar.
    # Two normal vectors for each site are the view angles toward ocean.
    sites = {
        'GCYN': {
            'coord': (36.4395, -121.9223),
            'normal1': _normalize_vector(-6.2, -2.5),
            'normal2': _normalize_vector(-7.7, -0.8),
            'text_loc': (1, 2),
        },
        'NPGS': {
            'coord': (36.6031, -121.8724),
            'normal1': _normalize_vector(2.3, 2.9),
            'normal2': _normalize_vector(-4.3, 4.7),
            'text_loc': (2, -4),
        },
        'PPNS': {
            'coord': (36.6367, -121.9350),
            'normal1': _normalize_vector(-4.6, 3.4),
            'normal2': _normalize_vector(1, 7.2),
            'text_loc': (11, -1),
        },
        'MLML': {
            'coord': (36.8040, -121.7880),
            'normal1': _normalize_vector(-10.9, -5.5),
            'normal2': _normalize_vector(-7.2, 3.9),
            'text_loc': (-3, 3),
        },
        'SCRZ': {
            'coord': (36.9483, -122.0659),
            'normal1': _normalize_vector(-0.6, -3.9),
            'normal2': _normalize_vector(0, -1),
            'text_loc': (-6, 3.5),
        },
        'BIGC': {
            'coord': (37.0885, -122.2734),
            'normal1': _normalize_vector(-6.2, -2.8),
            'normal2': _normalize_vector(-2.2, -5.6),
            'text_loc': (1.5, 1.5),
        },
        'PESC': {
            'coord': (37.2524833, -122.4165667),
            'normal1': _normalize_vector(-4.3, 0.3),
            'normal2': _normalize_vector(-4.3, 0.3),
            'text_loc': (2.5, -3),
        },
        'MONT': {
            'coord': (37.5337, -122.5192),
            'normal1': _normalize_vector(-4.3, -0.1),
            'normal2': _normalize_vector(-4.3, -0.1),
            'text_loc': (0, 2),
        },
    }

    # Data is obtained from http://hfrnet-tds.ucsd.edu/thredds. In this thredds
    # website, navigate to "HF RADAR, US West Coast", click on
    # "HFRADAR US West Coast 2km Resolution Hourly RTV" and then
    # "Best Time Series". From "Access" methods, select "NetcdfSubset", then
    # extract a subset of data between -122.843 to -121.698 longitudes and
    # 36.3992 to 37.2802 latitudes, from 2017-01-01 00:00:00Z to 2017-02-01
    # 00:00:00Z. After download, rename to the file below.
    # filename = '../files/Monterey_Large_2km_Hourly_2017_01.nc'

    # OpenDap URL of the remote netCDF data
    url = 'http://hfrnet-tds.ucsd.edu/thredds/' + \
          'dodsC/HFR/USWC/2km/hourly/RTV/HFRAD' + \
          'AR_US_West_Coast_2km_Resolution_Hou' + \
          'rly_RTV_best.ncd'

    # nc = netCDF4.Dataset(filename)
    nc = netCDF4.Dataset(url)
    # site_lon = nc.variables['site_lon'][:]
    # site_lat = nc.variables['site_lat'][:]
    # site_code = nc.variables['site_code'][:]
    datetime_obj = nc.variables['time']
    lon_obj = nc.variables['lon']
    lat_obj = nc.variables['lat']
    east_vel_obj = nc.variables['u']

    # Get datetime info from datetime netcdf object
    datetime_info = get_datetime_info(datetime_obj)

    # Subset settings
    time = '2017-01-25T03:00:00'
    min_time = ""
    max_time = ""
    min_lon = -122.843
    max_lon = -121.698
    min_lat = 36.3992
    max_lat = 37.2802

    # Subset time
    min_datetime_index, max_datetime_index = subset_datetime(
        datetime_info, min_time, max_time, time)

    # Subset domain
    min_lon_index, max_lon_index, min_lat_index, max_lat_index = \
        subset_domain(lon_obj, lat_obj, min_lon, max_lon, min_lat, max_lat)
    data_lon = nc.variables['lon'][min_lon_index:max_lon_index+1]
    data_lat = nc.variables['lat'][min_lat_index:max_lat_index+1]

    # Subset velocity
    u = east_vel_obj[
            min_datetime_index:max_datetime_index+1,
            min_lat_index:max_lat_index+1,
            min_lon_index:max_lon_index+1]

    # Site code names
    # site_codes = []
    # for i in range(site_code.shape[0]):
    #     code = site_code[i].tostring().decode('ascii').strip('\x00').strip(
    #             ' ')
    #     site_codes.append(code)

    site_codes = []
    site_lons = []
    site_lats = []
    site_normals1 = []
    site_normals2 = []

    for site_code in sites.keys():
        site_codes.append(site_code)
        site_lats.append(sites[site_code]['coord'][0])
        site_lons.append(sites[site_code]['coord'][1])
        site_normals1.append(sites[site_code]['normal1'])
        site_normals2.append(sites[site_code]['normal2'])

    # lon and lat of input data
    min_data_lon = numpy.min(data_lon)
    min_data_lat = numpy.min(data_lat)
    max_data_lon = numpy.max(data_lon)
    max_data_lat = numpy.max(data_lat)

    # Create a higher resolution grid of lon and lat
    res = 400
    lon = numpy.linspace(min_data_lon, max_data_lon, res)
    lat = numpy.linspace(min_data_lat, max_data_lat, res)
    lons_grid, lats_grid = numpy.meshgrid(lon, lat)
    data_lons_grid, data_lats_grid = numpy.meshgrid(data_lon, data_lat)

    # Plot
    load_plot_settings()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9.2, 7.36))

    map_00 = draw_map(ax[0, 0], data_lon, data_lat, draw_features=True,
                      draw_coastlines=True, percent=0.05)
    map_01 = draw_map(ax[0, 1], data_lon, data_lat, draw_features=True,
                      draw_coastlines=True, percent=0.05)
    map_10 = draw_map(ax[1, 0], data_lon, data_lat, draw_features=True,
                      draw_coastlines=True, percent=0.05)
    map_11 = draw_map(ax[1, 1], data_lon, data_lat, draw_features=True,
                      draw_coastlines=True, percent=0.05)

    # Meshgrids
    xx, yy = map_00(lons_grid, lats_grid)
    data_xx, data_yy = map_00(data_lons_grid, data_lats_grid)

    # One percent of map window to be used to relocate text annotations
    x_min = numpy.min(numpy.min(data_xx))
    x_max = numpy.max(numpy.max(data_xx))
    y_min = numpy.min(numpy.min(data_yy))
    y_max = numpy.max(numpy.max(data_yy))
    dx = (x_max - x_min) / 100.0  # in meters
    dy = (y_max - y_min) / 100.0  # in meters

    site_x, site_y = map_00(site_lons[:], site_lats[:])
    for i in range(len(sites.keys())):

        # Check if the site is inside the domain
        if site_lons[i] >= min_data_lon and \
           site_lons[i] <= max_data_lon and \
           site_lats[i] >= min_data_lat and \
           site_lats[i] <= max_data_lat:

            # Circle of site code
            map_00.plot(site_x[i], site_y[i], 'o', color='red', markersize=5,
                        zorder=10)
            map_01.plot(site_x[i], site_y[i], 'o', color='red', markersize=5,
                        zorder=10)
            map_10.plot(site_x[i], site_y[i], 'o', color='red', markersize=5,
                        zorder=10)
            map_11.plot(site_x[i], site_y[i], 'o', color='red', markersize=5,
                        zorder=10)

            # Text of site code
            rx, ry = sites[site_codes[i]]['text_loc']
            text_fontsize = 9
            ax[0, 0].text(site_x[i]+rx*dx, site_y[i]+ry*dy, site_codes[i],
                          fontsize=text_fontsize, zorder=2)
            ax[1, 0].text(site_x[i]+rx*dx, site_y[i]+ry*dy, site_codes[i],
                          fontsize=text_fontsize, zorder=2)
            ax[0, 1].text(site_x[i]+rx*dx, site_y[i]+ry*dy, site_codes[i],
                          fontsize=text_fontsize, zorder=2)
            ax[1, 1].text(site_x[i]+rx*dx, site_y[i]+ry*dy, site_codes[i],
                          fontsize=text_fontsize, zorder=2)

    # Plot normal arrows on each site for test purposes
    # scale = (numpy.max(xx) - numpy.min(xx))*0.3
    # for i in range(4):
    #     ax[0, 0].arrow(site_x[i], site_y[i],
    #                    site_x[i] + scale*site_normals1[i, 0],
    #                    site_y[i] + scale*site_normals1[i, 1], linewidth=4,
    #                    head_width=10000, head_length=10000)
    # ax[0, 0].arrow(site_x[0], site_y[0], site_x[1], site_y[1], linewidth=4,
    #                head_width=10000, head_length=10000)

    # Compute GDOP
    gdop, total_gdop = _compute_gdop(xx, yy, site_x, site_y, site_normals1,
                                     site_normals2, site_codes)

    # Compute Coverage
    coverage = _compute_coverage(u)

    colormap_gdop = plt.cm.YlGnBu
    colormap_coverage = plt.cm.Purples
    _plot_contour_on_axis(ax[0, 0], map_00, xx, yy, gdop[:, :, 0, 0],
                          '(a) East GDOP', colormap_gdop, threshold=1.2,
                          remove_contours=[0.8, 0.9])
    _plot_contour_on_axis(ax[0, 1], map_01, xx, yy, gdop[:, :, 1, 1],
                          '(b) North GDOP', colormap_gdop, threshold=0.9,
                          remove_contours=[0.75, 0.8, 0.85])
    _plot_contour_on_axis(ax[1, 0], map_10, xx, yy, total_gdop[:, :],
                          '(c) Total GDOP', colormap_gdop, threshold=1.3)
    # _plot_contour_on_axis(ax[1, 1], map_11, xx, yy, gdop[:, :, 0, 1],
    #                       '(d) East/North GDOP', colormap_gdop,
    #                       threshold=1.2)
    _plot_pcolormesh_on_axis(ax[1, 1], map_11, data_xx, data_yy, coverage,
                             r'(d) Coverage (\%)', colormap_coverage)

    fig.subplots_adjust(hspace=0.15, wspace=0.03)

    # plt.show()

    fig.patch.set_alpha(0)
    filename = 'gdop_coverage'
    save_plot(filename, dpi=200, transparent_background=False, pdf=True,
              bbox_extra_artists=None, verbose=False)


# ===========
# System main
# ===========

if __name__ == "__main__":
    main(sys.argv)
