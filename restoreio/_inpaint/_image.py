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
from ._plot_image import plot_color_and_grayscale_images
from ._cast_types import cast_float_array_to_uint8_array

__all__ = ['convert_velocities_to_color_image']


# =================================
# Convert Velocities To Color Image
# =================================

def convert_velocities_to_color_image(
        all_missing_indices_in_ocean,
        land_indices,
        valid_indices,
        U_original,
        V_original):
    """
    Takes two arrays of U and V (each 2D numpy array), converts them into
    grayscale 8-bit array, and then add them into a 3-channel color image. The
    third channel is filled with zeros.

    Note: U and V are assumed to be 2D arrays, not 3D array.

    The both U and V arrays are set to zero on land area. NOTE that this does
    not result in black image on land. This is because zero on array is not
    mapped to zero on image. Zero in image (black) is corresponding to the min
    value of the U or V arrays.

    To plot the image results, set plot_images=True.
    """

    # Get mean value of U
    valid_U = U_original[valid_indices[:, 0], valid_indices[:, 1]]
    mean_U = numpy.mean(valid_U)

    # Fill U with mean values
    filled_U = numpy.zeros(U_original.shape, dtype=float)

    # Use original U for valid points
    for i in range(valid_indices.shape[0]):
        filled_U[valid_indices[i, 0], valid_indices[i, 1]] = \
                U_original[valid_indices[i, 0], valid_indices[i, 1]]

    # Use U_mean for missing points in ocean
    for i in range(all_missing_indices_in_ocean.shape[0]):
        filled_U[all_missing_indices_in_ocean[i, 0],
                 all_missing_indices_in_ocean[i, 1]] = mean_U

    # Zero out the land indices
    if numpy.any(numpy.isnan(land_indices)) is False:
        for i in range(land_indices.shape[0]):
            filled_U[land_indices[i, 0], land_indices[i, 1]] = 0.0

    # Get mean values of V
    valid_V = V_original[valid_indices[:, 0], valid_indices[:, 1]]
    mean_V = numpy.mean(valid_V)

    # Fill V with mean values
    filled_V = numpy.zeros(U_original.shape, dtype=float)

    # Use original V for valid points
    for i in range(valid_indices.shape[0]):
        filled_V[valid_indices[i, 0], valid_indices[i, 1]] = \
                V_original[valid_indices[i, 0], valid_indices[i, 1]]

    # Use mean V for missing points in ocean
    for i in range(all_missing_indices_in_ocean.shape[0]):
        filled_V[all_missing_indices_in_ocean[i, 0],
                 all_missing_indices_in_ocean[i, 1]] = mean_V

    # Zero out the land indices
    if numpy.any(numpy.isnan(land_indices)) is False:
        for i in range(land_indices.shape[0]):
            filled_V[land_indices[i, 0], land_indices[i, 1]] = 0.0

    # Create gray scale image for each U and V
    gray_scale_image_U = cast_float_array_to_uint8_array(filled_U)
    gray_scale_image_V = cast_float_array_to_uint8_array(filled_V)

    # Create color image from both gray scales U and V
    color_image = numpy.zeros((U_original.shape[0], U_original.shape[1], 3),
                              dtype=numpy.uint8)
    color_image[:, :, 0] = gray_scale_image_U
    color_image[:, :, 1] = gray_scale_image_V

    # Plot images. To plot, set plot_images to True.
    plot_images = False
    if plot_images:
        plot_color_and_grayscale_images(
                gray_scale_image_U, gray_scale_image_V, color_image)

    return color_image
