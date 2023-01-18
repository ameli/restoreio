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
from .._plots._plot_utilities import save_plot, plt, load_plot_settings

__all__ = ['plot_auto_correlation']


# =====================
# Plot Auto Correlation
# =====================

def plot_auto_correlation(
        acf_lon,
        acf_lat,
        acf_length_scale_lon,
        acf_length_scale_lat,
        vel_component,
        save=True):
    """
    Plots ACF.
    """

    load_plot_settings()

    fig, ax = plt.subplots(figsize=(5.7, 4))

    plot_size = 8
    x = numpy.arange(plot_size)
    y1 = numpy.exp(-x/acf_length_scale_lon)
    y2 = numpy.exp(-x/acf_length_scale_lat)

    # Plot
    # ax.plot(acf_lon, '-o', color='blue', label='Eastward autocorrelation')
    # ax.plot(acf_lat, '-o', color='green', label='Northward autocorrelation')
    # ax.plot(x, y1, '--s', color='blue',
    #         label='Eastward exponential kernel fit')
    # ax.plot(x, y2, '--s', color='green',
    #          label='Northward exponential kernel it')
    # ax.plot(numpy.array([x[0], x[-1]]),
    #          numpy.array([numpy.exp(-1), numpy.exp(-1)]), '--')

    ax.semilogy(acf_lon, 'o', color='blue', label='Eastward Autocorrelation')
    ax.semilogy(x, y1, '--', color='blue',
                label=r'Exponential Kernel Fit ($\alpha=%0.2f$)'
                      % acf_length_scale_lon)
    ax.semilogy(acf_lat, 'o', color='green',
                label='Northward Autocorrelation')
    ax.semilogy(x, y2, '--', color='green',
                label=r'Exponential Kernel Fit ($\alpha=%0.2f$)'
                      % acf_length_scale_lat)
    ax.semilogy(numpy.array([x[0], x[plot_size-1]]),
                numpy.array([numpy.exp(-1), numpy.exp(-1)]), '--')
    ax.semilogy(numpy.array([x[0], x[plot_size-1]]),
                numpy.array([numpy.exp(-3), numpy.exp(-3)]), '--')

    title = 'Autocorrelation Function'
    if vel_component == 'east':
        title += ' (East Velocity Component)'
    else:
        title += ' (North Velocity Component)'
    ax.set_title(title)
    ax.set_xlabel('Shift')
    ax.set_ylabel('ACF')
    ax.set_xlim(0, plot_size-1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc='lower left', fontsize='x-small')

    fig.set_tight_layout(True)

    # Save plot
    if save:
        filename = 'auto_correlation_' + vel_component
        save_plot(filename, transparent_background=True, pdf=True,
                  bbox_extra_artists=None, verbose=True)
