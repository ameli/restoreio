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

__all__ = ['compute_correlation_matrix']


# =======================================
# Auto Correlation ARD Exponential Kernel
# =======================================

def _autocorrelation_ard_exponential_kernel(
        id_1,
        id_2,
        valid_indices,
        acf_length_scale_lon,
        acf_length_scale_lat):
    """
    Finds the correlation between two points with id_1 and id_2.
    """

    index_J1 = valid_indices[id_1, 0]
    index_J2 = valid_indices[id_2, 0]
    index_I1 = valid_indices[id_1, 1]
    index_I2 = valid_indices[id_2, 1]

    # Autocorrelation
    X = (index_I1 - index_I2) / acf_length_scale_lon
    Y = (index_J1 - index_J2) / acf_length_scale_lat
    acf_id_1_id_2 = numpy.exp(-numpy.sqrt(X**2+Y**2))

    return acf_id_1_id_2


# ===========================
# Auto Correlation RBF Kernel
# ===========================

def _autocorrelation_rbf_kernel(id_1, id_2, valid_indices, quadratic_form):
    """
    RBF kernel with quadratic form.
    """

    index_J1 = valid_indices[id_1, 0]
    index_J2 = valid_indices[id_2, 0]
    index_I1 = valid_indices[id_1, 1]
    index_I2 = valid_indices[id_2, 1]
    distance_vector = numpy.array([index_I2-index_I1, index_J2-index_J1])
    quadrature = numpy.dot(distance_vector,
                           numpy.dot(quadratic_form, distance_vector.T))
    acf_id_1_id_2 = numpy.exp(-0.5*numpy.sqrt(quadrature))

    return acf_id_1_id_2


# ==========================
# Compute Correlation Matrix
# ==========================

def compute_correlation_matrix(
        valid_indices,
        acf_length_scale_lon,
        acf_length_scale_lat,
        quadratic_form):
    """
    valid_indices is array of size (num_valid, 2)
    covariance matrix Cor is of size (num_valid, num_valid).
    """

    num_valid = valid_indices.shape[0]
    cor = numpy.zeros((num_valid, num_valid), dtype=float)

    for i in range(num_valid):
        for j in range(i, num_valid):
            # cor[i, j] = _autocorrelation_ard_exponential_kernel(
            #         i, j, valid_indices, acf_length_scale_lon,
            #         acf_length_scale_lat)
            cor[i, j] = _autocorrelation_rbf_kernel(i, j, valid_indices,
                                                    quadratic_form)
            if i != j:
                cor[j, i] = cor[i, j]

    return cor
