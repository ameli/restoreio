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
import scipy

from .._plots_uq import plot_outliers

__all__ = ['robust_linear_solver']


# ===============
# Linear Function
# ===============

def _linear_function(beta, X_row, y_row):
    """
    Computes the residual of the equation X @ beta - y. Here, We only compute
    one row of X, called x. Thus, this function computes the residual of a
    linear system for only one par of data (X, y), where X_row is one row of
    the matrix X, and y_row here is the corresponding element of a label data
    y.
    """

    return numpy.dot(X_row, beta) - y_row


# =====
# Solve
# =====

def _solve(X, y, loss='linear', f_scale=1.0):
    """
    Solves a system based on a given loss function.

    The loss function can be one of 'linear', 'huber', 'soft_l1', 'cauchy', or
    'arctan'. Except the linear loss function, all other loss functions require
    a parameter 'f_scale' to be set. It is best to set this parameter to be the
    noise of the linear model (see compute_noise function).

    If the loss function is linear, the solution is equivalent to the ordinary
    least squares, where the solution beta is

    beta = inv(X.T @ X) # X.T @ y

    But for other loss function, the solution beta is not the same as the
    above.
    """

    # Initial guess of beta
    beta0 = numpy.ones(X.shape[1])

    # Using scipy's optimize function to solve least squares for a given loss
    # function.
    res_lsq = scipy.optimize.least_squares(_linear_function, beta0,
                                           args=(X, y), loss=loss,
                                           f_scale=f_scale)

    # Solution beta
    beta = res_lsq.x
    return beta


# ================
# Noise Estimation
# ================

def _noise_estimation(X, y):
    """
    Computes noise based on linear loss function (ordinary least square).
    """

    n, m = X.shape
    beta = numpy.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta
    error = y - y_hat
    noise2 = numpy.dot(error, error) / (n - m)
    noise = numpy.sqrt(noise2)

    return noise


# =======================
# Robust Noise Estimation
# =======================

def _robust_noise_estimation(X, y):
    """
    Computes noise based on linear loss function (ordinary least square), but
    the noise estimation does not include the outliers.
    """

    n = X.shape[0]

    # Compute Cook distance
    cook_dist = _cook_distance(X, y, loss='linear')
    cook_dist_mean = numpy.mean(cook_dist)

    # Flag points that have Cook distance larger than 3 * mean of cook distance
    cook_dist_mean = numpy.mean(cook_dist)
    outliers = numpy.zeros((n, ), dtype=bool)
    outliers[cook_dist > 3.0 * cook_dist_mean] = True
    not_outliers = numpy.logical_not(outliers)

    # Compute OLS noise but exclude outliers
    robust_noise = _noise_estimation(X[not_outliers, :], y[not_outliers])

    return robust_noise, outliers


# =============
# Cook Distance
# =============

def _cook_distance(X, y, loss='linear', f_scale=1.0):
    """
    Computes the Cook distance.

    The i-th Cook distance is competed by removing the i-th data from the data
    pair (X, y), which means removing the i-th row of X and y.

    """

    n, m = X.shape

    # Using all data
    beta_full = _solve(X, y, loss=loss, f_scale=f_scale)
    y_hat_full = X @ beta_full
    error_full = y - y_hat_full
    s2 = numpy.dot(error_full, error_full) / (n-m)

    # Cook distance for each data to be left out
    cook_dist = numpy.zeros((n, ), dtype=float)

    # Leave out each data once at a time (LOO)
    for i in range(n):
        ind = numpy.arange(n).tolist()
        ind.remove(i)

        # Compute beta when leaving one out (LOO)
        beta_loo = _solve(X[ind, :], y[ind], loss=loss, f_scale=f_scale)

        y_hat_loo = X @ beta_loo
        error = y_hat_full - y_hat_loo
        cook_dist[i] = numpy.dot(error, error) / (m * s2)

    return cook_dist


# ====================
# Robust Linear Solver
# ====================

def robust_linear_solver(
        X,
        y,
        vel_component,
        loss='huber',
        plot=False,
        save=True,
        verbose=False):
    """
    Solving linear system X @ beta = y using least squares with a given loss
    function.

    The plot argument plots the outliers using Cook distance using various loss
    functions.
    """

    robust_noise, outliers = _robust_noise_estimation(X, y)
    beta = _solve(X, y, loss=loss, f_scale=robust_noise)

    if plot:

        # Compute Cook distance using various loss functions
        dist_lin = _cook_distance(X, y, loss='linear')
        dist_hub = _cook_distance(X, y, loss='huber', f_scale=robust_noise)
        dist_sl1 = _cook_distance(X, y, loss='soft_l1', f_scale=robust_noise)
        dist_chy = _cook_distance(X, y, loss='cauchy', f_scale=robust_noise)
        dist_atn = _cook_distance(X, y, loss='arctan', f_scale=robust_noise)

        # Plot
        plot_outliers(X, y, dist_lin, dist_hub, dist_sl1, dist_chy, dist_atn,
                      robust_noise, outliers, vel_component, save=save,
                      verbose=verbose)

    return beta
