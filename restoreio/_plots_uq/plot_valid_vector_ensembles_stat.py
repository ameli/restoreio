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
import scipy.stats
from .._plots._plot_utilities import save_plot, plt, load_plot_settings

__all__ = ['plot_valid_vector_ensembles_stat']


# ======================================
# Plot Valid Vector Ensembles Statistics
# ======================================

def plot_valid_vector_ensembles_stat(
        valid_vector,
        valid_vector_error,
        random_vectors,
        valid_vector_ensembles,
        vel_component,
        save=True):
    """
    Compare the mean, std, skewness, kurtosis of ensembles with the generated
    random vectors.

    valid vector is a 1D array of the size num_points, which is the number of
    valid points.

    valid_vector_ensembles is a 2D array of the size (num_points,
    num_ensembles).

    random_vectors is a 2D array of the size (num_modes, num_ensembles).
    """

    load_plot_settings()

    # Moments of valid vector ensembles
    m1 = numpy.mean(valid_vector_ensembles, axis=1) - valid_vector
    m2 = numpy.std(valid_vector_ensembles, axis=1) - valid_vector_error
    m3 = scipy.stats.skew(valid_vector_ensembles, axis=1)
    m4 = scipy.stats.kurtosis(valid_vector_ensembles, axis=1)

    # Moments of random vector
    r1 = numpy.mean(random_vectors, axis=1)
    r2 = numpy.std(random_vectors, axis=1) - 1.0
    r3 = scipy.stats.skew(random_vectors, axis=1)
    r4 = scipy.stats.kurtosis(random_vectors, axis=1)

    # Plot x limit
    num_points = valid_vector_ensembles.shape[0]
    num_modes = random_vectors.shape[0]
    xlim = numpy.max([num_points, num_modes]) - 1

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(11, 6))

    ax[0, 0].plot(m1, color='black', label='Diff. Mean of ensembles with' +
                  'central ensemble')
    ax[0, 0].plot(r1, color='red', label='Mean of random vectors')
    ax[0, 0].set_xlim([0, xlim])
    ax[0, 0].set_title('Mean Difference')
    plt.xlabel('Point')
    ax[0, 0].legend(fontsize='x-small')

    ax[0, 1].plot(m2, color='black',
                  label='Diff std ensembles with actual error')
    ax[0, 1].plot(r2, color='red', label='std of generate random vectors - 1')
    ax[0, 1].set_xlim([0, xlim])
    ax[0, 1].set_title('Standard Deviation Difference')
    ax[0, 1].set_xlabel('Point')
    ax[0, 1].legend(fontsize='x-small')

    ax[1, 0].plot(m3, color='black', label='Skewness of ensembles')
    ax[1, 0].plot(r3, color='red', label='Skewness of generated random ' +
                  'vectors')
    ax[1, 0].set_xlim([0, xlim])
    ax[1, 0].set_xlabel('Point')
    ax[1, 0].set_title('Skewness')
    ax[1, 0].legend(fontsize='x-small')

    ax[1, 1].plot(m4, color='black', label='Kurtosis of ensembles')
    ax[1, 1].plot(r4, color='red', label='Kurtosis of generated random ' +
                  'vectors')
    ax[1, 1].set_xlim([0, xlim])
    ax[1, 1].set_xlabel('Points')
    ax[1, 1].set_title('Excess Kurtosis')
    ax[1, 1].legend(fontsize='x-small')

    fig.set_tight_layout(True)

    # Save plot
    if save:
        filename = 'ensembles_stat_' + vel_component
        save_plot(filename, transparent_background=True, pdf=True,
                  bbox_extra_artists=None, verbose=True)
