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
from .._plots._plot_utilities import save_plot, plt, make_axes_locatable, cm, \
        load_plot_settings
from ._refine_mask import refine_mask
from .._plots._draw_map import draw_map, draw_axis
from .._uncertainty_quant._image_utils import convert_valid_vector_to_image

__all__ = ['plot_kl_transform']


# ===============
# Snap To Decimal
# ===============

def _snap_to_decimal(x, snap=2):
    """
    Snaps a float number to its decimal precision. Snap=2, for instance, makes
    the numbers to snap to 0.2, 0.4, 0.6,  or to 2, 4, 6, etc.
    """

    fl_log_x = numpy.floor(numpy.log10(x) + 0.5)
    x_base = 10**fl_log_x
    x_rem = x / x_base
    x = x_base * numpy.floor(x_rem / snap + 0.5) * snap

    return x


# ================
# Plot eigenvalues
# ================

def _plot_eigenvalues(
        eigenvalues,
        vel_component,
        save=True):
    """
    Plots log log scale of eigenvalues
    """

    # eigenvalues
    fig, ax1 = plt.subplots(figsize=(5.7, 4.2))

    ax1.semilogy(eigenvalues, color='green',
                 label=r'$\lambda_i$ (%s Velocity Data)'
                 % vel_component.capitalize())
    ax1.set_xlabel(r'Modes Number ($i$ or $M$)')
    ax1.set_ylabel(r'$\lambda_i$', color='green')
    ax1.grid(True)
    ax1.set_xlim([1, eigenvalues.size])

    ax1.set_ylim([
        numpy.min([_snap_to_decimal(numpy.min(eigenvalues)), 1e-5]),
        numpy.max([_snap_to_decimal(numpy.max(eigenvalues)), 2e-1])])
    ax1.tick_params('y', colors='green')

    # Commutative eigenvalues
    eigenvalues_cum_sum = numpy.cumsum(eigenvalues)
    ax2 = ax1.twinx()
    ax2.plot(eigenvalues_cum_sum/eigenvalues_cum_sum[-1], color='blue',
             label=r'$\gamma_M$ (%s Velocity Data)'
             % vel_component.capitalize())

    # Horizontal lines at 0.6 and 0.9
    ax2.axhline(y=0.6, color='black', linestyle='--')
    ax2.axhline(y=0.9, color='black', linestyle='--')

    # Add 0.9 to the yticks
    yticks = numpy.sort(numpy.arange(0, 1.01, 0.2).tolist() + [0.9])
    yticks = numpy.array(yticks)
    # ax2.set_yticks(yticks)

    # Find where eig_cum_sum reaches 90%

    ax2.set_ylabel(r'Normalized Cumulative Sum $\gamma_{M}$', color='blue')
    ax2.set_xlim([1, eigenvalues.size])
    ax2.set_ylim([0, 1])
    ax2.tick_params('y', colors='blue')
    h1, l1 = ax1.get_legend_handles_labels()  # legend for both ax and twin
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower left', fontsize='small')

    title = 'Eigenvalues of KL Transform'
    ax1.set_title(title)

    # fig.set_tight_layout(True)

    # Save plot
    if save:
        filename = 'kl_eigenvalues_' + vel_component
        save_plot(filename, transparent_background=True, pdf=True,
                  bbox_extra_artists=None, verbose=True)


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
        refined_mask_data):
    """
    This plots in each of left or right axes.
    """

    # Config
    title_fontsize = 10
    label_fontsize = 8

    ax.set_aspect('equal')
    # ax.set_rasterization_zorder(0)
    ax.set_facecolor('#C7DCEF')

    # Pcolormesh
    draw = map.pcolormesh(lons_grid_on_map, lats_grid_on_map,
                          scalar_field, cmap=cm.jet, rasterized=True,
                          zorder=-1)
    # contour_levels = 200
    # draw = map.contourf(lons_grid_on_map, lats_grid_on_map,
    #                     scalar_field, contour_levels, cmap=cm.jet,
    #                     rasterized=True, zorder=-1,
    #                     corner_mask=False)

    # Draw edges lines around mask pixels
    if refined_mask_data is not {}:

        # Convert lon and lat degrees to length coordinates on map in meters
        refined_lons_grid_on_map, refined_lats_grid_on_map = map(
                refined_mask_data['refined_lons_grid'],
                refined_mask_data['refined_lats_grid'])

        # Plot a level-set of the mask at 0.5 (between True and False)
        ax.contour(refined_lons_grid_on_map, refined_lats_grid_on_map,
                   refined_mask_data['refined_mask'], levels=[0.5],
                   colors='black', zorder=-1, linewidths=1)

    # Create axes for colorbar that is the same size as the plot axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Colorbar
    cb = plt.colorbar(draw, cax=cax,
                      ticks=numpy.array(
                          [numpy.ma.min(scalar_field),
                           numpy.ma.max(scalar_field)]))
    cb.solids.set_rasterized(True)
    cb.ax.tick_params(labelsize=label_fontsize)
    for tick in cb.ax.get_yticklabels():
        tick.set_rotation(90)

    # Align top label to be bottom and align bottom label to be top.
    ticks = cb.ax.get_yticklabels()
    ticks[0].set_va('bottom')
    ticks[1].set_va('top')

    ax.set_title(title, fontdict={'fontsize': title_fontsize})


# =================
# Plot eigenvectors
# =================

def _plot_eigenvectors(
        lon,
        lat,
        valid_indices,
        missing_indices_in_ocean_inside_hull,
        image_shape,
        eigenvectors,
        vel_component,
        save=True):
    """
    Plot eigenvectors on the map.
    """

    # Mesh grid
    lons_grid, lats_grid = numpy.meshgrid(lon, lat)

    # Create a refined mask to plot contours around mask pixels.
    refined_mask_data = refine_mask(
        lon, lat, missing_indices_in_ocean_inside_hull)

    num_rows = 2
    num_columns = 6
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns,
                             figsize=(2.32*num_columns, 2.25*num_rows))

    title = 'Mercer Eigenvectors'
    if vel_component == 'east':
        title += ' (East Component)'
    else:
        title += ' (North Component)'
    # fig.suptitle(title, fontsize=12)

    # Map
    map = draw_map(None, lon, lat, percent=0.0, draw_coastlines=True,
                   draw_features=False)

    # Meshgrids on map
    lons_grid_on_map, lats_grid_on_map = map(lons_grid, lats_grid)

    counter = 0
    for ax in axes.flat:

        # Switch map to draw on the current ax
        map.ax = ax
        draw_axis(ax, lon, lat, map, percent=0.0, draw_coastlines=True,
                  draw_features=False)

        eigenvector_image = convert_valid_vector_to_image(
                eigenvectors[:, counter], valid_indices, image_shape)
        _plot_on_each_axis(ax, map, lons_grid_on_map, lats_grid_on_map,
                           eigenvector_image, 'Mode %d' % (counter+1),
                           refined_mask_data)
        counter += 1

    fig.subplots_adjust(hspace=0.14, wspace=0.15)

    # fig.set_tight_layout(True)
    fig.patch.set_alpha(0)

    # Save plot
    if save:
        filename = 'kl_eigenvectors_' + vel_component
        save_plot(filename, transparent_background=False, pdf=True,
                  bbox_extra_artists=None, verbose=True)


# =================
# Plot KL Transform
# =================

def plot_kl_transform(
        lon,
        lat,
        valid_indices,
        missing_indices_in_ocean_inside_hull,
        image_shape,
        eigenvalues,
        eigenvectors,
        vel_component,
        save=True):
    """
    Plots eigenvalues and eigenvectors of the KL transform.
    """

    load_plot_settings()

    _plot_eigenvalues(eigenvalues, vel_component, save)
    _plot_eigenvectors(lon, lat, valid_indices,
                       missing_indices_in_ocean_inside_hull, image_shape,
                       eigenvectors, vel_component, save)
