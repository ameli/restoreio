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
from ._generate_valid_vector_ensembles import generate_valid_vector_ensembles
from .._plots_uq import plot_auto_correlation, plot_kl_transform, \
        plot_rbf_kernel
from ._image_utils import convert_valid_vector_to_image

__all__ = ['generate_image_ensembles']


# ==============================
# Get Valid Indices For All Axes
# ==============================

def _get_valid_indices_for_all_axes(array):
    """
    Generates a map between Ids and (TimeIndex, lat_index, lon_index)
    for only valid points. This is for the 3D array.
    """

    # Along lon
    valid_indices_list_lon = []
    ids_lon = numpy.ones(array.shape, dtype=int) * (-1)

    counter = 0
    for lat_index in range(array.shape[0]):
        for lon_index in range(array.shape[1]):
            if bool(array.mask[lat_index, lon_index]) is False:
                valid_indices_list_lon.append((lat_index, lon_index))
                ids_lon[lat_index, lon_index] = counter
                counter += 1

    valid_indices_lon = numpy.array(valid_indices_list_lon)

    # Along lat
    valid_indices_list_lat = []
    ids_lat = numpy.ones(array.shape, dtype=int) * (-1)

    counter = 0
    for lon_index in range(array.shape[1]):
        for lat_index in range(array.shape[0]):
            if bool(array.mask[lat_index, lon_index]) is False:
                valid_indices_list_lat.append((lat_index, lon_index))
                ids_lat[lat_index, lon_index] = counter
                counter += 1

    valid_indices_lat = numpy.array(valid_indices_list_lat)

    return valid_indices_lon, ids_lon, valid_indices_lat, ids_lat


# =======================================
# Compute AutoCorrelation Of Valid Vector
# =======================================

def _compute_autocorrelation_of_valid_vector(valid_vector):
    """
    acf is computed from a vectorized data. The N-dimensional data is
    vectorized to be like a time series. Data is assumed periodic, so the
    indices are rotating around the size of the vector. The acf is then a
    simple shift in the 1D data, like

    acf(shift) = sum_{i=1}^N time_series(i)*time_series(i+shift)

    acf is normalized, so that acf[0] = 1, by acf =/ acf[0].
    """

    num_valid = valid_vector.size
    time_series = valid_vector - numpy.mean(valid_vector)

    # Limit the size of acf shift
    acf_size = int(num_valid / 100)
    if acf_size < 10:
        acf_size = 10
    if acf_size > 100:
        acf_size = 100

    acf = numpy.zeros((acf_size, ), dtype=float)

    for i in range(acf_size):
        for j in range(num_valid):
            acf[i] += time_series[j]*time_series[(j+i) % num_valid]

    # Normalize
    acf /= acf[0]

    return acf


# ===================================
# Estimate Autocorrelation RBF Kernel
# ===================================

def _estimate_autocorrelation_rbf_kernel(
        masked_image_data,
        valid_indices,
        ids,
        window_lon,
        window_lat,
        vel_component,
        plot=False,
        save=True):
    """
    ---------
    Abstract:
    ---------

    The radial Basis Function (RBF) kernel is assumed in this function. We
    assume the 2D data has the following kernel:
        K(ii, jj) = exp(-0.5*sqrt(X.T * quadratic_form * X))
    where X is a vector:
        X = (i-ii, j-jj)
    That is, at the center point (i, j), the X is the distance vector with the
    shift (ii, jj). This function estimates the quadratic_form 2x2 symmetric
    matrix.

    -------
    Inputs:
    -------

    - masked_image_data:
        MxN masked data of the east/north velocity field.

    - valid_indices:
        (num_valid, 2) array. Each rwo is of the form [lat_index, lon_index].
        If there are NxM points in the grid, not all of these points have
        valid velocity data defined. Suppose there are only num_valid points
        with valid velocities. Each row of valid_indices is the latitude and
        longitudes of these points.

    - ids:
        (N, M) array of integers. If a point is non-valid, the value is -1, if
        a point is valid, the value on the array is the Id (row number in valid
        indices). This array is used to show how we mapped valid_indices to the
        grid ids.

    - window_lon:
        Scalar. This is the half window of stencil in lon direction for
        sweeping the kernel convolution. The rectangular area of the kernel is
        of size (2*window_lon_1, 2*window_lat+1).

    - window_lat:
        Scalar. This is the half window of stencil in lat direction for
        sweeping the kernel convolution. The rectangular area of the kernel is
        of size (2*window_lon_1, 2*window_lat+1).

    --------
    Outputs:
    --------

    - quadratic_form:
        2x2 symmetric matrix.

    ---------------------------
    How we estimate the kernel:
    ---------------------------

    1. We subtract mean of data form data. Lets call it data.

    2. For each valid point with id_1 (or lat and lon Indices), we look for all
       neighbor points within the central stencil of size window_lon,
       window_lat (This is a rectangle of size:

            2 * window_lon+1, 2*window_lat+1).

       If any of these points are valid itself, say point id_2, we compute the
       correlation:

            Correlation of id_1 and id_2 = data[id_1] * data[id_2].

       This value is stored in a 2D array of size of kernel, where the center
       of kernel corresponds to the point with id_1. We store this correlation
       value in the offset index of two points id_1 and id_2 in the Kernel
       array.

    2. Now we have num_valid 2D kernels for each valid points. We average them
       all to get a 2D kernel_average array. We normalize this array w.r.t to
       the center value of this array. So the center of the kernel_average is
       1.0. If we plot this array, it should be descending.

    3. We fit an exponential RBG function to the 2D kernel:

            z = exp(-0.5*sqrt(X.T * quadratic_form * X))

       where X = (i-ii, j-jj) is a distance vector form the center of the
       kernel array. Suppose the kernel array is of size

            (P=2*window_lon+1, Q=2*window_lat_1)

       To do so we take Z = 4.0*(log(z))^2 = X.T * quadratic_form * X. Suppose
       (ii, jj) is the center indices of the kernel array. For each point
       (i, j) in the kernel matrix, we compute

            A = [(i-ii)**2, 2*(i-ii)*(j-jj), (j-jj)**2] this is array of size
            (P*Q, 3) for each i, j on the Kernel.

            b = [Z] this is a vector of size (P*Q, ) for each i, j in the
            kernel.

       Now we find the least square AX=b.
       The quadratic form is

        quadratic_form = [ X[0], X[1] ]
                        [ X[1], X[2] ]

    -----
    Note:
    -----

    - Let Lambda_1 and Lambda_2 be the eigenvalues of Quadratic Form. Then
      L1 = 1/sqrt(Lambda_1) and L2=1/sqrt(Lambda_2) are the characteristic
      lengths of the RBF kernel. If the quadratic kernel is diagonal, this is
      essentially the ARM kernel.

    - The eigenvalues should be non-negative. But if they are negative, this is
      because we chose a large window_lon or window_lat, hence the 2D Kernel
      function is not strictly descending everywhere. To fix this choose
      smaller window sizes.
    """

    # Subtract mean of masked Image Data
    data = numpy.ma.copy(masked_image_data) - numpy.ma.mean(masked_image_data)

    # 3D array that the first index is for each valid point, and the 2 and 3
    # index creates a 2D matrix that is the autocorrelation of that valid point
    # (i, j) with the nearby point (i+ii, j+jj)
    kernel_for_all_valid_points = numpy.ma.masked_all(
            (valid_indices.shape[0], 2*window_lat+1, 2*window_lon+1),
            dtype=float)

    # Iterate over valid points
    for id_1 in range(valid_indices.shape[0]):
        lat_index_1, lon_index_1 = valid_indices[id_1, :]

        # Sweep the kernel rectangular area to find nearby points to the center
        # point.
        for lat_offset in range(-window_lat, window_lat+1):
            for lon_offset in range(-window_lon, window_lon+1):

                lat_index_2 = lat_index_1 + lat_offset
                lon_index_2 = lon_index_1 + lon_offset

                if (lat_index_2 >= 0) and \
                        (lat_index_2 < masked_image_data.shape[0]):
                    if (lon_index_2 >= 0) and \
                            (lon_index_2 < masked_image_data.shape[1]):
                        id_2 = ids[lat_index_2, lon_index_2]

                        if id_2 >= 0:
                            # The nearby point is a valid point. Compute the
                            # correlation of points id_1 and id_2 and store in
                            # the Kernel
                            kernel_for_all_valid_points[
                                    id_1, lat_offset+window_lat,
                                    lon_offset+window_lon] = \
                                            data[lat_index_2, lon_index_2] * \
                                            data[lat_index_1, lon_index_1]

    # Average all 2D kernels over the valid points (over the first index)
    kernel_average = numpy.ma.mean(kernel_for_all_valid_points, axis=0)

    # Normalize the kernel w.r.t the center of the kernel. So the center is 1.0
    # and all other correlations are less that 1.0.
    kernel_average = kernel_average / kernel_average[window_lat, window_lon]

    # Get the gradient of the kernel to find up to where the kernel is
    # descending. We only use kernel in areas there it is descending.
    gradient_kernel_average = numpy.gradient(kernel_average)

    # Find where on the grid the data is descending (in order to avoid
    # ascending in acf)
    descending = numpy.zeros((2*window_lat+1, 2*window_lon+1), dtype=bool)
    for lat_offset in range(-window_lat, window_lat+1):
        for lon_offset in range(-window_lon, window_lon+1):
            radial = numpy.array([lat_offset, lon_offset])
            norm = numpy.linalg.norm(radial)
            if norm > 0:
                radial = radial / norm
            elif norm == 0:
                descending[lat_offset+window_lat, lon_offset+window_lon] = True
            Grad = numpy.array(
                    [gradient_kernel_average[0][
                        lat_offset+window_lat, lon_offset+window_lon],
                     gradient_kernel_average[1][
                        lat_offset+window_lat, lon_offset+window_lon]])

            if numpy.dot(Grad, radial) < 0.0:
                descending[lat_offset+window_lat, lon_offset+window_lon] = True

    # Construct a Least square matrices
    A_list = []
    b_list = []

    for lat_offset in range(-window_lat, window_lat+1):
        for lon_offset in range(-window_lon, window_lon+1):
            Value = kernel_average[lat_offset+window_lat,
                                   lon_offset+window_lon]
            if Value <= 0.05:
                continue
            elif bool(descending[lat_offset+window_lat,
                                 lon_offset+window_lon]) is False:
                continue

            a = numpy.zeros((3, ), dtype=float)
            a[0] = lon_offset**2
            a[1] = 2.0*lon_offset*lat_offset
            a[2] = lat_offset**2
            A_list.append(a)
            b_list.append(4.0*(numpy.log(Value))**2)

    # Check length
    if len(b_list) < 3:
        raise RuntimeError('Insufficient number of kernel points. Can not ' +
                           'perform least square to estimate kernel.')

    # List to array
    A = numpy.array(A_list)
    b = numpy.array(b_list)

    # Least square
    AtA = numpy.dot(A.T, A)
    Atb = numpy.dot(A.T, b)
    X = numpy.linalg.solve(AtA, Atb)
    quadratic_form = numpy.array([[X[0], X[1]], [X[1], X[2]]])

    # Plot RBF kernel function
    if plot:
        plot_rbf_kernel(quadratic_form, kernel_average, X, window_lon,
                        window_lat, vel_component, save)

    return quadratic_form


# =====================================
# Estimate AutoCorrelation Length Scale
# =====================================

def _estimate_autocorrelaton_length_scale(acf):
    """
    Assuming a Markov-1 Stationary process (mean and std do not change over
    time), the autocorrelation function is acf = rho**(d), where d is spatial
    distance between two points.
    """

    # Find up to where the acf is positive
    window = 0
    while (acf[window] > 0.0) and (acf[window] > acf[window+1]):
        window += 1
        if window >= acf.size-1:
            break

    if window < 1:
        raise RuntimeError('window of positive acf is not enough to ' +
                           'estimate parameter.')

    x = numpy.arange(1, window)
    y = numpy.log(acf[1:window])
    length_scale = -numpy.mean(x/y)

    return length_scale


# ========================
# Generate Image Ensembles
# ========================

def generate_image_ensembles(
        lon,
        lat,
        masked_image_data,
        masked_image_data_error,
        valid_indices,
        missing_indices_in_ocean_inside_hull,
        num_ensembles,
        num_modes,
        kernel_window,
        vel_component,
        plot=False,
        save=True):
    """
    Note: The lon and lat is NOT needed for the computation of this function.
    However, if we want to plot the eigenvectors on the map, we need the lons
    and lats.

    Input:
    - masked_image_data:
        (n, m) Image array that are partially masked. This is the original
        velocity data (either u or v velocity)

    - masked_image_data_error:
        (n, m) Image array that is partially masked. This is the error of
        original velocity data (either u or v velocity)

    - valid_indices:
        (num_valid, 2) 1D array. First column are latitudes (i indices) and
        second column are longitude (j indices) of valid data on velocity
        arrays and their errors.

    - num_ensembles:
        The number of output array of ensembles is actually num_ensembles+1,
        since the first ensemble that we output is the original data itself in
        order to have the central ensemble in the data.

    Note:
        The first ensemble is the central ensemble, which is the ensemble
        without perturbation of variables and corresponds to the mean of each
        variable.
    """

    # Convert Image data to vector
    valid_vector = masked_image_data[valid_indices[:, 0], valid_indices[:, 1]]
    valid_vector_error = \
        masked_image_data_error[valid_indices[:, 0], valid_indices[:, 1]]

    # Compute Autocorrelation of data for each axis
    valid_indices_lon, ids_lon, valid_indices_lat, ids_lat = \
        _get_valid_indices_for_all_axes(masked_image_data)
    valid_vector_lon = \
        masked_image_data[valid_indices_lon[:, 0], valid_indices_lon[:, 1]]
    valid_vector_lat = \
        masked_image_data[valid_indices_lat[:, 0], valid_indices_lat[:, 1]]
    acf_lon = _compute_autocorrelation_of_valid_vector(valid_vector_lon)
    acf_lat = _compute_autocorrelation_of_valid_vector(valid_vector_lat)
    acf_length_scale_lon = _estimate_autocorrelaton_length_scale(acf_lon)
    acf_length_scale_lat = _estimate_autocorrelaton_length_scale(acf_lat)

    print('length_scales: Lon: %f, Lat: %f'
          % (acf_length_scale_lon, acf_length_scale_lat))

    # Plot acf
    if plot:
        plot_auto_correlation(acf_lon, acf_lat, acf_length_scale_lon,
                              acf_length_scale_lat, vel_component, save)

    # window of kernel
    window_lon = kernel_window
    window_lat = kernel_window
    quadratic_form = _estimate_autocorrelation_rbf_kernel(
            masked_image_data, valid_indices_lon, ids_lon, window_lon,
            window_lat, vel_component, plot)

    # Generate ensembles for vector (Note: eigenvalues and eigenvectors are
    # only needed for plotting them)
    valid_vector_ensembles, eigenvalues, eigenvectors = \
        generate_valid_vector_ensembles(
                valid_vector, valid_vector_error, valid_indices, num_ensembles,
                num_modes, acf_length_scale_lon, acf_length_scale_lat,
                quadratic_form, vel_component, plot=plot, save=save)
    num_ensembles = valid_vector_ensembles.shape[1]

    # Convert back vector to image
    masked_image_data_ensembles = numpy.ma.masked_all(
            (num_ensembles+1, )+masked_image_data.shape, dtype=float)

    # Add the original data to the first ensemble as the central ensemble
    masked_image_data_ensembles[0, :, :] = masked_image_data

    # Add ensembles that are produced by KL expansion with perturbation of
    # variables
    for ensemble_id in range(num_ensembles):
        masked_image_data_ensembles[ensemble_id+1, :, :] = \
                convert_valid_vector_to_image(
                        valid_vector_ensembles[:, ensemble_id], valid_indices,
                        masked_image_data.shape)

    # Plot eigenvalues and eigenvectors (Uncomment to plot)
    if plot:
        plot_kl_transform(lon, lat, valid_indices,
                          missing_indices_in_ocean_inside_hull,
                          masked_image_data.shape, eigenvalues, eigenvectors,
                          vel_component, save)

    return masked_image_data_ensembles
