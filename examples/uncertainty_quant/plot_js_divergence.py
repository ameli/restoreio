#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import sys
import os
from os.path import join
import netCDF4
import numpy
import scipy.stats
from _plot_utilities import plt, load_plot_settings, save_plot, \
        PercentFormatter


# ===========
# JS Distance
# ===========

def js_distance(
        field_mean_1,
        field_mean_2,
        field_sigma_1,
        field_sigma_2):
    """
    JS distance of two normal distributions.
    """

    field_js_metric = numpy.ma.masked_all(
            field_mean_1.shape, dtype=float)

    for i in range(field_mean_1.shape[0]):
        for j in range(field_mean_1.shape[1]):

            if bool(field_mean_1.mask[i, j]) is False:
                mean_1 = field_mean_1[i, j]
                mean_2 = field_mean_2[i, j]
                sigma_1 = numpy.abs(field_sigma_1[i, j])
                sigma_2 = numpy.abs(field_sigma_2[i, j])

                x = numpy.linspace(numpy.min([mean_1-6*sigma_1,
                                             mean_2-6*sigma_2]),
                                   numpy.max([mean_1+6*sigma_1,
                                             mean_2+6*sigma_2]), 10000)

                norm_1 = scipy.stats.norm.pdf(x, loc=mean_1, scale=sigma_1)
                norm_2 = scipy.stats.norm.pdf(x, loc=mean_2, scale=sigma_2)
                norm_12 = 0.5*(norm_1+norm_2)

                jsd = 0.5 * (scipy.stats.entropy(norm_1, norm_12, base=2) +
                             scipy.stats.entropy(norm_2, norm_12, base=2))

                if jsd < 0.0:
                    if jsd > -1e-8:
                        jsd = 0.0
                    else:
                        print('WARNING: Negative JS distance: %f' % jsd)

                field_js_metric[i, j] = numpy.sqrt(jsd)

    return field_js_metric


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


# ====
# main
# ====

def main():
    """
    """

    # Get full number of modes, which is the number of valid points from the
    # original data points
    # data_dir =

    dir = "output_js_divergence"
    base_filename = "output"
    num_files = 200
    stride = 1

    # Initialize arrays
    files_id = numpy.arange(1, num_files+1, stride)
    js_div_east_mean = numpy.zeros((files_id.size+1,), dtype=float)
    js_div_north_mean = numpy.zeros((files_id.size+1,), dtype=float)
    js_div_east_std = numpy.zeros((files_id.size+1,), dtype=float)
    js_div_north_std = numpy.zeros((files_id.size+1,), dtype=float)

    # Corresponding to the 0-th mode (complete truncation of KL expansion)
    js_div_east_mean[0] = 1.0
    js_div_north_mean[0] = 1.0
    js_div_east_std[0] = 0.0
    js_div_north_std[0] = 0.0

    # The last file corresponds to the full KL expansion
    last_filename = join(dir, base_filename + "-%03d" % num_files + ".nc")

    if not os.path.isfile(last_filename):
        raise RuntimeError('File %s does not exists.' % last_filename)

    for i in range(files_id.size):
        filename = join(dir,
                        base_filename + "-%03d" % files_id[i] + ".nc")

        if not os.path.isfile(filename):
            raise RuntimeError('File %s does not exists.' % filename)
        print('Processing %s' % filename)

        js_div_east_field, js_div_north_field = \
            _js_distance_of_two_distributions(filename, last_filename)

        js_div_east_mean[i+1] = numpy.ma.mean(js_div_east_field)
        js_div_north_mean[i+1] = numpy.ma.mean(js_div_north_field)
        js_div_east_std[i+1] = numpy.ma.std(js_div_east_field)
        js_div_north_std[i+1] = numpy.ma.std(js_div_north_field)

    # Plot
    load_plot_settings()

    # Config
    title_fontsize = 12
    label_fontsize = 10

    fig, ax = plt.subplots(figsize=(5.7, 4.2))
    modes = numpy.zeros((files_id.size+1, ), dtype=float)
    modes[1:] = 100 * files_id / num_files

    ax.plot(modes, js_div_east_mean, color='darkgreen',
            label='East velocity data (average)')
    ax.plot(modes, js_div_north_mean, color='mediumblue',
            label='North velocity data (average)')
    ax.fill_between(modes, js_div_east_mean-js_div_east_std,
                    js_div_east_mean+js_div_east_std, color='lightgreen',
                    label='East velocity data (STD bound)', alpha=0.5)
    ax.fill_between(modes, js_div_north_mean-js_div_north_std,
                    js_div_north_mean+js_div_north_std, color='lightskyblue',
                    label='East velocity data (STD bound)', alpha=0.5)

    ax.set_title('JS Distance Between Truncated vs Complete KL Expansion',
                 fontsize=title_fontsize)
    ax.set_xlabel(r'Modes Ratio ($m / n$)', fontsize=label_fontsize)
    ax.set_ylabel(r'$\sqrt{\mathcal{D}_{JS}(p_m \Vert p_{n})}$',
                  fontsize=label_fontsize)
    ax.tick_params(labelsize=label_fontsize)
    ax.grid(True)
    ax.set_xlim([-0.25, 100])
    ax.set_ylim([0, 1])
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.legend(loc='upper right', fontsize='x-small')

    plt.tight_layout()

    # Save plot
    plot_filename = 'js_distance'
    save_plot(plot_filename, transparent_background=True, pdf=True,
              bbox_extra_artists=None, verbose=True)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
