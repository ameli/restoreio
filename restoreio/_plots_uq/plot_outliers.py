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
from .._plots._plot_utilities import save_plot, plt, load_plot_settings

__all__ = ['plot_outliers']


# =============
# Plot Outliers
# =============

def plot_outliers(
        X,
        y,
        dist_lin,
        dist_hub,
        dist_sl1,
        dist_chy,
        dist_atn,
        robust_noise,
        outliers,
        vel_component,
        save=True,
        verbose=False):
    """
    Plots Cook distance to find outliers. This function uses various loss
    function when solving a linear system to compare how each loss function
    affects the outliers.
    """

    # Mean of the Cook distance using linear loss function
    dist_lin_mean = numpy.mean(dist_lin)

    load_plot_settings()
    fig, ax = plt.subplots(figsize=(5.5, 3.66))

    ind = numpy.arange(X.shape[0])
    ax.plot(ind, dist_lin, color='black', label='Linear')
    ax.plot(ind, dist_sl1, color='red', label='Soft $L_1$')
    ax.plot(ind, dist_hub, color='blue', label='Huber')
    ax.plot(ind, dist_chy, color='green', label='Cauchy')
    ax.plot(ind, dist_atn, color='cyan', label='ArcTan')
    ax.axhline(3.0 * dist_lin_mean, linestyle='--', color='gray',
               label=r'3 $\bar{D}$ Threshold')
    ax.plot(ind[outliers], dist_lin[outliers], 'o', markersize=5,
            color='black', label='Outlier')

    ax.set_xlim([ind[0]-0.5, ind[-1]+0.5])
    ax.set_ylim(bottom=0)

    title_fontsize = 11
    label_fontsize = 10

    ax.tick_params(labelsize=label_fontsize)
    ax.set_xlabel('Index $i$', fontsize=label_fontsize)
    ax.set_ylabel('Cook Distance $D_i$', fontsize=label_fontsize)
    ax.set_title('Detection of Outliers',
                 fontdict={'fontsize': title_fontsize})

    # Legend
    lines = plt.gca().get_lines()
    include1 = [0, 1, 2, 3, 4]
    include2 = [5, 6]

    legend1 = plt.legend([lines[i] for i in include1],
                         [lines[i].get_label() for i in include1],
                         title='Loss Functions', title_fontsize=label_fontsize,
                         fontsize='x-small', bbox_to_anchor=(0.18, 1),
                         loc='upper left')

    legend2 = plt.legend([lines[i] for i in include2],             # noqa: F841
                         [lines[i].get_label() for i in include2],
                         title='Outlier Detection',
                         title_fontsize=label_fontsize, fontsize='x-small',
                         bbox_to_anchor=(.44, 1), loc='upper left')

    plt.gca().add_artist(legend1)

    # Save plot
    if save:
        filename = 'outliers_' + vel_component
        save_plot(filename, transparent_background=True, pdf=True,
                  bbox_extra_artists=None, verbose=verbose)
