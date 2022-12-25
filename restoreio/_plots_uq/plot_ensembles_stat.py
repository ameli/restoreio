# SPDX-FileCopyrightText: Copyright 2016, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# ======
# Import
# ======

import numpy
import netCDF4

from ._refine_mask import refine_mask
from .._plots._plot_utilities import plt, matplotlib, make_axes_locatable, \
        load_plot_settings, save_plot, cm
from .._plots._draw_map import draw_map, draw_axis
from ._shifted_colormap import shifted_colormap
from ._statistical_distances import js_distance

__all__ = ['plot_ensembles_stat']


# =================
# Plot on Each axis
# =================

def _plot_on_each_axis(
        ax,
        map,
        lons_grid_on_map,
        lats_grid_on_map,
        scalar_field,
        title,
        colormap,
        clabel='',
        refined_mask_data={},
        log_norm=False,
        shift_colormap=False):
    """
    This plots in each of left or right axes.
    """

    # Config
    title_fontsize = 12
    label_fontsize = 10

    # Set axis to map
    map.ax = ax

    # Shift colormap
    if shift_colormap is True:
        min = numpy.ma.min(scalar_field)
        max = numpy.ma.max(scalar_field)
        if (min < 0) and (max > 0):
            mid_point = -min/(max-min)
            colormap = shifted_colormap(colormap, start=0.0,
                                        midpoint=mid_point, stop=1.0)

    # Pcolormesh
    if log_norm is True:
        # Plot in log scale
        min = numpy.max([numpy.min(scalar_field), 1e-6])
        draw = map.pcolormesh(
                lons_grid_on_map, lats_grid_on_map, scalar_field,
                cmap=colormap, rasterized=True, zorder=-1,
                norm=matplotlib.colors.LogNorm(vmin=min))
        # draw = map.contourf(
        #         lons_grid_on_map, lats_grid_on_map, scalar_field,
        #         200, cmap=colormap, corner_mask=False,
        #         rasterized=True, zorder=-1)
    else:
        # Do not plot in log scale
        draw = map.pcolormesh(lons_grid_on_map, lats_grid_on_map,
                              scalar_field, cmap=colormap,
                              rasterized=True, zorder=-1)

    # Draw edges lines around mask pixels
    if refined_mask_data is not {}:

        # Convert lon and lat degrees to length coordinates on map in meters
        refined_lons_grid_on_map, refined_lats_grid_on_map = map(
                refined_mask_data['refined_lons_grid'],
                refined_mask_data['refined_lats_grid'])

        # Plot a level-set of the mask at 0.5 (between True and False)
        ax.contour(refined_lons_grid_on_map, refined_lats_grid_on_map,
                   refined_mask_data['refined_mask'], levels=[0.5],
                   colors='black', zorder=-1, linewidths=1.5)

    # Create axes for colorbar that is the same size as the plot axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)

    # Colorbar
    cb = plt.colorbar(draw, cax=cax)
    cb.solids.set_rasterized(True)
    cb.set_label(clabel, fontsize=label_fontsize)
    cb.ax.tick_params(labelsize=label_fontsize)

    # ax labels
    ax.set_title(title, fontsize=title_fontsize)
    # ax.set_xlabel('lon (degrees)')
    # ax.set_ylabel('lat (degrees)')

    # Background blue for ocean
    ax.set_facecolor('#C7DCEF')


# =================
# Plot Scalar Field
# =================

def _plot_scalar_fields(
        lon,
        lat,
        map,
        lons_grid_on_map,
        lats_grid_on_map,
        scalar_field_1,
        scalar_field_2,
        colormap,
        title,
        title_prefix=('a', 'b'),
        vertical_axes=False,
        refined_mask_data={},
        shift_colormap=False,
        log_norm=False,
        save=True,
        filename='scalar_field',
        clabel=''):
    """
    This creates a figure with two axes.

    The quantity scalar_field_1 is related to the east velocity and will be
    plotted on the left axis. The quantity scalar_field_2 is related to the
    north velocity and will be plotted on the right axis.

    If ShiftColorMap is True, we set the zero to the center of the colormap
    range. This is useful if we use divergent colormaps like cm.bwr.
    """

    print('Plotting %s ...' % filename)

    # Default colormap
    if colormap is None:
        colormap = cm.jet()

    if vertical_axes:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 7.46))
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.73))
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    # Resterization. Anything with zorder less than 0 will be rasterized.
    # Do not use the below lies due to a bug in basemap.pcolormesh that in
    # backend mode (when plots are saved instead of being shown) an error will
    # be risen by: 'NoneType' object has no attribute 'buffer_rgba'
    # ax[0].set_rasterization_zorder(0)
    # ax[1].set_rasterization_zorder(0)

    # Left axis
    map.ax = ax[0]
    draw_axis(ax[0], lon, lat, map, draw_coastlines=True, draw_features=True)
    title_1 = '(' + title_prefix[0] + ') ' + title + ' (Eastward Component)'
    _plot_on_each_axis(ax[0], map, lons_grid_on_map, lats_grid_on_map,
                       scalar_field_1, title_1, colormap, clabel,
                       refined_mask_data, log_norm, shift_colormap)

    # Right axis
    map.ax = ax[1]
    draw_axis(ax[1], lon, lat, map, draw_coastlines=True, draw_features=True)
    title_2 = '(' + title_prefix[1] + ') ' + title + ' (Northward Component)'
    _plot_on_each_axis(ax[1], map, lons_grid_on_map, lats_grid_on_map,
                       scalar_field_2, title_2, colormap, clabel,
                       refined_mask_data, log_norm, shift_colormap)

    fig.set_tight_layout(True)
    fig.patch.set_alpha(0)

    # Save plot
    if save:
        save_plot(filename, transparent_background=False, pdf=True,
                  bbox_extra_artists=None, verbose=True)


# ===============================
# JS Distance Of Two Distribution
# ===============================

def _js_distance_of_two_distributions(
        filename_1,
        filename_2):
    """
    Reads two files, and computes the JS metric distance of their
    east/north velocities. The JS metric distance is the square root of JS
    distance. Log base 2 is used, hence the output is in range [0, 1].
    """

    nc_f = netCDF4.Dataset(filename_1)
    nc_t = netCDF4.Dataset(filename_2)

    east_mean_f = nc_f.variables['east_vel'][0, :]
    east_mean_t = nc_t.variables['east_vel'][0, :]
    east_sigma_f = nc_f.variables['east_err'][0, :]
    east_sigma_t = nc_t.variables['east_err'][0, :]
    east_jsd = js_distance(east_mean_t, east_mean_f, east_sigma_t,
                           east_sigma_f)

    north_mean_f = nc_f.variables['north_vel'][0, :]
    north_mean_t = nc_t.variables['north_vel'][0, :]
    north_sigma_f = nc_f.variables['north_err'][0, :]
    north_sigma_t = nc_t.variables['north_err'][0, :]
    north_jsd = js_distance(north_mean_t, north_mean_f, north_sigma_t,
                            north_sigma_f)

    return east_jsd, north_jsd


# ==========================
# Ratio of Truncation Energy
# ==========================

def _ratio_of_truncation_energy(
        filename_1,
        filename_2,
        missing_indices_in_ocean_inside_hull):
    """
    Ratio of StandardDeviation^2 for truncated and full KL expansion.
    """

    nc_f = netCDF4.Dataset(filename_1)
    nc_t = netCDF4.Dataset(filename_2)

    east_std_f = nc_f.variables['east_err'][0, :]
    east_std_t = nc_t.variables['east_err'][0, :]
    east_energy_ratio = numpy.ma.masked_all(east_std_f.shape, dtype=float)
    east_energy_ratio[:] = 1 - (east_std_t / east_std_f)**2

    north_std_f = nc_f.variables['north_err'][0, :]
    north_std_t = nc_t.variables['north_err'][0, :]
    north_energy_ratio = numpy.ma.masked_all(east_std_f.shape, dtype=float)
    north_energy_ratio[:] = 1 - (north_std_t / north_std_f)**2

    # mask
    for id in range(missing_indices_in_ocean_inside_hull.shape[0]):
        lat_index = missing_indices_in_ocean_inside_hull[id, 0]
        lon_index = missing_indices_in_ocean_inside_hull[id, 1]
        east_energy_ratio[lat_index, lon_index] = numpy.ma.masked
        north_energy_ratio[lat_index, lon_index] = numpy.ma.masked

    return east_energy_ratio, north_energy_ratio


# =========================
# Plot Ensembles Statistics
# =========================

def plot_ensembles_stat(
        lon,
        lat,
        valid_indices,
        missing_indices_in_ocean_inside_hull,
        U_one_time,
        V_one_time,
        error_U_one_time,
        error_V_one_time,
        U_all_ensembles_inpainted,
        V_all_ensembles_inpainted,
        U_all_ensembles_inpainted_stats,
        V_all_ensembles_inpainted_stats,
        save=True):
    """
    Plots of ensembles statistics.
    """

    load_plot_settings()

    # Mesh grid
    lons_grid, lats_grid = numpy.meshgrid(lon, lat)

    # Create a refined mask to plot contours around mask pixels.
    refined_mask_data = refine_mask(
        lon, lat, missing_indices_in_ocean_inside_hull)

    # # All Missing points coordinates
    # all_missing_lon = lons_grid[all_missing_indices[:, 0],
    #                             all_missing_indices[:, 1]]
    # all_missing_lat = lats_grid[all_missing_indices[:, 0],
    #                             all_missing_indices[:, 1]]
    # all_missing_points_coord = numpy.vstack(
    #         (all_missing_lon, all_missing_lat)).T
    #
    # # Missing points coordinates inside hull
    # missing_lon_inside_hull = lons_grid[missing_indices_inside_hull[:, 0],
    #                                     missing_indices_inside_hull[:, 1]]
    # missing_lat_inside_hull = lats_grid[missing_indices_inside_hull[:, 0],
    #                                     missing_indices_inside_hull[:, 1]]
    # missing_points_coord_inside_hull = numpy.vstack(
    #         (missing_lon_inside_hull, missing_lat_inside_hull)).T
    #
    # # Missing points coordinates outside hull
    # missing_lon_outside_hull = lons_grid[missing_indices_outside_hull[:, 0],
    #                                      missing_indices_outside_hull[:, 1]]
    # missing_lat_outside_hull = lats_grid[missing_indices_outside_hull[:, 0],
    #                                      missing_indices_outside_hull[:, 1/]
    # missing_points_coord_outside_hull = numpy.vstack((
    #     missing_lon_outside_hull, missing_lat_outside_hull)).T

    # # Valid points coordinates
    # valid_lon = lons_grid[valid_indices[:, 0], valid_indices[:, 1]]
    # valid_lat = lats_grid[valid_indices[:, 0], valid_indices[:, 1]]
    # valid_points_coord = numpy.c_[valid_lon, valid_lat]

    # # Land Point Coordinates
    # if numpy.any(numpy.isnan(land_indices)) == False:
    #     land_lon = lons_grid[land_indices[:, 0], land_indices[:, 1]]
    #     land_lat = lats_grid[land_indices[:, 0], land_indices[:, 1]]
    #     land_point_coord = numpy.c_[land_lon, land_lat]
    # else:
    #     land_point_coord = numpy.nan

    # Create a dummy ax for the map
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))

    # Map
    dummy_ax = None
    map = draw_map(dummy_ax, lon, lat, draw_features=True)
    lons_grid_on_map, lats_grid_on_map = map(lons_grid, lats_grid)

    # Original (Uninpainted) Data
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        U_one_time, V_one_time, cm.jet, 'Velocity',
                        title_prefix=('a', 'b'), vertical_axes=False,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=False, log_norm=False, save=save,
                        filename='orig_vel', clabel='m/s')
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        error_U_one_time, error_V_one_time, cm.Reds,
                        'Velocity Error', title_prefix=('c', 'd'),
                        vertical_axes=False,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=False, log_norm=False, save=save,
                        filename='orig_vel_error', clabel='m/s')

    # Central Ensemble
    central_ensemble_east_vel = \
        U_all_ensembles_inpainted_stats['CentralEnsemble'][0, :]
    central_ensemble_north_vel = \
        V_all_ensembles_inpainted_stats['CentralEnsemble'][0, :]
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        central_ensemble_east_vel, central_ensemble_north_vel,
                        cm.jet, 'Central Ensemble', title_prefix=('a', 'b'),
                        vertical_axes=False,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=False, log_norm=False, save=save,
                        filename='ensembles_central', clabel='m/s')

    # Mean Difference
    mean_diff_east_vel = numpy.ma.abs(
        (U_all_ensembles_inpainted_stats['Mean'][0, :] -
            U_all_ensembles_inpainted[0, :]) /
        U_all_ensembles_inpainted_stats['STD'][0, :])
    mean_diff_north_vel = numpy.ma.abs(
        (V_all_ensembles_inpainted_stats['Mean'][0, :] -
         V_all_ensembles_inpainted[0, :]) /
        V_all_ensembles_inpainted_stats['STD'][0, :])
    mean_diff_east_vel[numpy.ma.where(mean_diff_east_vel < 1e-8)] = \
        numpy.ma.masked
    mean_diff_north_vel[numpy.ma.where(mean_diff_north_vel < 1e-8)] = \
        numpy.ma.masked

    # Normalized first-order moment deviation w.r.t central ensemble
    # mean_diff_east_vel = numpy.ma.abs(
    #     (U_all_ensembles_inpainted_stats['Mean'][0, :] - \
    #         U_all_ensembles_inpainted_stats['central_ensemble'][0, :]) / \
    #         U_all_ensembles_inpainted_stats['central_ensemble'][0, :])
    # mean_diff_north_vel = numpy.ma.abs(
    #     (V_all_ensembles_inpainted_stats['Mean'][0, :] - \
    #         V_all_ensembles_inpainted_stats['central_ensemble'][0, :]) / \
    #         V_all_ensembles_inpainted_stats['central_ensemble'][0, :])
    # mean_diff_east_vel = numpy.ma.abs(
    #     (U_all_ensembles_inpainted_stats['Mean'][0, :] - \
    #             U_all_ensembles_inpainted_stats['central_ensemble'][0, :]))
    # mean_diff_north_vel = numpy.ma.abs(
    #     (V_all_ensembles_inpainted_stats['Mean'][0, :] - \
    #             V_all_ensembles_inpainted_stats['central_ensemble'][0, :]))
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        mean_diff_east_vel, mean_diff_north_vel, cm.Reds,
                        '1st Deviation', title_prefix=('a', 'e'),
                        vertical_axes=True,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=False, log_norm=True, save=save,
                        filename='deviation_1')

    # Mean
    mean_east_vel = U_all_ensembles_inpainted_stats['Mean'][0, :]
    mean_north_vel = V_all_ensembles_inpainted_stats['Mean'][0, :]
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        mean_east_vel, mean_north_vel, cm.jet,
                        'Ensembles Mean', title_prefix=('a', 'b'),
                        vertical_axes=False,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=False, log_norm=False, save=save,
                        filename='ensembles_mean', clabel='m/s')

    # STD
    # std_east_vel = numpy.log(U_all_ensembles_inpainted_stats['STD'][0, :])
    # std_north_vel = numpy.log(V_all_ensembles_inpainted_stats['STD'][0, :])
    std_east_vel = U_all_ensembles_inpainted_stats['STD'][0, :]
    std_north_vel = V_all_ensembles_inpainted_stats['STD'][0, :]
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        std_east_vel, std_north_vel, cm.Reds,
                        'Ensembles STD', title_prefix=('c', 'd'),
                        vertical_axes=False,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=False, log_norm=False, save=save,
                        filename='ensembles_std', clabel='m/s')

    # RMSD
    # rmsd_east_vel = \
    #    numpy.log(U_all_ensembles_inpainted_stats['RMSD'][0, :])
    # rmsd_north_vel = \
    #     numpy.log(V_all_ensembles_inpainted_stats['RMSD'][0, :])
    rmsd_east_vel = U_all_ensembles_inpainted_stats['RMSD'][0, :]
    rmsd_north_vel = V_all_ensembles_inpainted_stats['RMSD'][0, :]
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        rmsd_east_vel, rmsd_north_vel, cm.Reds, 'RMSD',
                        title_prefix=('a', 'b'), vertical_axes=False,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=False, log_norm=True, save=save,
                        filename='ensembles_rmsd', clabel='m/s')

    # NRMSD
    # nrmsd_east_vel = \
    #     numpy.log(U_all_ensembles_inpainted_stats['NRMSD'][0, :])
    # nrmsd_north_vel = \
    #     numpy.log(V_all_ensembles_inpainted_stats['NRMSD'][0, :])
    nrmsd_east_vel = U_all_ensembles_inpainted_stats['NRMSD'][0, :]
    nrmsd_north_vel = V_all_ensembles_inpainted_stats['NRMSD'][0, :]
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        nrmsd_east_vel, nrmsd_north_vel, cm.Reds, 'NRMSD',
                        title_prefix=('a', 'b'), vertical_axes=False,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=False, log_norm=True, save=save,
                        filename='ensembles_nrmsd')

    # Excess Normalized second moment deviation
    # ex_nmsd_east_vel = \
    #         numpy.log(U_all_ensembles_inpainted_stats['ExNMSD'][0, :])
    # ex_nmsd_north_vel = \
    #         numpy.log(V_all_ensembles_inpainted_stats['ExNMSD'][0, :])
    ex_nmsd_east_vel = U_all_ensembles_inpainted_stats['ExNMSD'][0, :]
    ex_nmsd_north_vel = V_all_ensembles_inpainted_stats['ExNMSD'][0, :]
    ex_nmsd_east_vel[numpy.ma.where(ex_nmsd_east_vel < 5e-5)] = \
        numpy.ma.masked
    ex_nmsd_north_vel[numpy.ma.where(ex_nmsd_north_vel < 5e-5)] = \
        numpy.ma.masked
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        ex_nmsd_east_vel, ex_nmsd_north_vel, cm.Reds,
                        '2nd Deviation', title_prefix=('b', 'f'),
                        vertical_axes=True,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=False, log_norm=True, save=save,
                        filename='deviation_2')

    # Skewness (Excess Normalized 3rd Moment Deviation w.r.t Central Ensemble)
    skewness_east_vel = U_all_ensembles_inpainted_stats['Skewness'][0, :]
    skewness_north_vel = V_all_ensembles_inpainted_stats['Skewness'][0, :]
    Trim = 1.2
    skewness_east_vel[numpy.ma.where(skewness_east_vel > Trim)] = Trim
    skewness_east_vel[numpy.ma.where(skewness_east_vel < -Trim)] = -Trim
    skewness_north_vel[numpy.ma.where(skewness_north_vel > Trim)] = Trim
    skewness_north_vel[numpy.ma.where(skewness_north_vel < -Trim)] = -Trim
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        skewness_east_vel, skewness_north_vel, cm.bwr,
                        '3rd Deviation',
                        title_prefix=('c', 'g'), vertical_axes=True,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=True, log_norm=False, save=save,
                        filename='deviation_3')

    # Excess Kurtosis
    ex_kurtosis_east_vel = \
        U_all_ensembles_inpainted_stats['ExKurtosis'][0, :]
    ex_kurtosis_north_vel = \
        V_all_ensembles_inpainted_stats['ExKurtosis'][0, :]
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        ex_kurtosis_east_vel, ex_kurtosis_north_vel, cm.bwr,
                        '4th Deviation',
                        title_prefix=('d', 'h'), vertical_axes=True,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=True, log_norm=False, save=save,
                        filename='deviation_4')

    # Entropy of ensembles
    entropy_east_vel = U_all_ensembles_inpainted_stats['Entropy'][0, :]
    entropy_north_vel = V_all_ensembles_inpainted_stats['Entropy'][0, :]
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        entropy_east_vel, entropy_north_vel, cm.RdBu_r,
                        'Entropy', title_prefix=('a', 'b'),
                        vertical_axes=False,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=True, log_norm=False, save=save,
                        filename='ensembles_entropy', clabel='nat')

    # Relative Entropy (KL Divergence) with respect to the normal distribution
    relative_entropy_east_vel = \
        U_all_ensembles_inpainted_stats['RelativeEntropy'][0, :]
    relative_entropy_north_vel = \
        V_all_ensembles_inpainted_stats['RelativeEntropy'][0, :]
    # Trim = 0.1
    relative_entropy_east_vel[
            numpy.ma.where(relative_entropy_east_vel > Trim)] = Trim
    relative_entropy_north_vel[
            numpy.ma.where(relative_entropy_north_vel > Trim)] = Trim
    _plot_scalar_fields(lon, lat, map, lons_grid_on_map, lats_grid_on_map,
                        relative_entropy_east_vel, relative_entropy_north_vel,
                        cm.YlOrRd, 'KL Divergence', title_prefix=('a', 'b'),
                        vertical_axes=False,
                        refined_mask_data=refined_mask_data,
                        shift_colormap=True, log_norm=False, save=save,
                        filename="ensembles_rel_entropy", clabel='nat')

    # Plotting additional entropies between two distributions
    # filename_1 = 'Output_Full_KL_Expansion.nc'
    # filename_2 = 'Output_Truncated_KL_Expansion.nc'
    # east_jsd, north_jsd = _js_distance_of_two_distributions(
    #         filename_1, filename_2)
    #
    # # Jensen-Shannon divergence between full and truncated KL expansion
    # _plot_scalar_fields(lon, lat, east_jsd, north_jsd, cm.Reds,
    #                     'JS Divergencpo', title_prefix=('a', 'b'),
    #                     vertical_axes=False,
    #                     refined_mask_data=refined_mask_data,
    #                     shift_colormap=False, log_norm=False, save=save,
    #                     filename='ensembles_js_divergence', clabel='bit')
    #
    # # Ratio of truncation error energy over total energy of KL expansion'
    # east_energy_ratio, north_energy_ratio = _ratio_of_truncation_energy(
    #         filename_1, filename_2, missing_indices_in_ocean_inside_hull)
    # _plot_scalar_fields(lon, lat, east_energy_ratio, north_energy_ratio,
    #                     cm.Reds, 'Rel. Truncation Error Energy',
    #                     title_prefix=('a', 'b'), vertical_axes=False,
    #                     refined_mask_data=refined_mask_data,
    #                     shift_colormap=False, log_norm=False, save=save,
    #                     filename='ensembles_rel_trunc_err_energy')

    if not save:
        plt.show()
