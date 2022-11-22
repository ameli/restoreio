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

import cv2
import numpy
from ._plot_image import plot_color_and_grayscale_images

__all__ = ['inpaint_all_missing_points',
           'restore_missing_points_inside_domain']


# ===============================
# Cast Float Array to UInt8 Array
# ===============================

def _cast_float_array_to_uint8_array(float_array):
    """
    Casts float array to UInt8. Here the float range of data is mapped to the
    integer range 0-255 linearly.
    """

    min_array = numpy.min(float_array)
    max_array = numpy.max(float_array)

    # Scale array into the rage of 0 to 255
    scaled_float_array = 255.0 * (float_array - min_array) / \
        (max_array - min_array)

    # Cast float array to uint8 array
    uint8_array = (scaled_float_array + 0.5).astype(numpy.uint8)

    return uint8_array


# ===============================
# Cast UInt8 Array To Float Array
# ===============================

def _cast_uint8_array_to_float_array(uint8_array, original_float_array):
    """
    Casts UInt8 array to float array. Here, the second argument
    "OriginalFLoarArray" is used to find the range of data. So that the range
    0-255 to mapped to the range of data linearly.
    """

    min_float = numpy.min(original_float_array)
    max_float = numpy.max(original_float_array)

    scaled_float_array = uint8_array.astype(float)
    float_array = min_float + scaled_float_array * (max_float - min_float) / \
        255.0

    return float_array


# =================================
# Convert Velocities To Color Image
# =================================

def _convert_velocities_to_color_image(
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
    gray_scale_image_U = _cast_float_array_to_uint8_array(filled_U)
    gray_scale_image_V = _cast_float_array_to_uint8_array(filled_V)

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


# ==========================
# Inpaint All Missing Points
# ==========================

def inpaint_all_missing_points(
        all_missing_indices_in_ocean,
        land_indices,
        valid_indices,
        U_original,
        V_original,
        difusivity,
        sweep_all_directions):
    """
    This function uses opencv.inpaint to restore the colored images.
    The colored images are obtained from adding 2 grayscale images from
    velocities U and V and a 0 (null) channel. In this method, ALL missing
    points in ocean, including those inside and outside the hull.

    There are two parameters:

    Diffusivity: This is the same as Reynolds number in NS.
    sweep_all_directions: If set to True, the image is inpainted 4 times as
    follow:
        1. Original orientation of image
        2. Flipped left/right orientation of image
        3. Filled up/down orientation of image
        4. Again the original orientation og image

    Note: If in this function we set inpaint_land=True, it inpaints land area
    as well. If not, the land is already zero to zero and it considers it as a
    known value on the image.
    """

    # Create 8-bit 3-channel image from U and V
    color_image = _convert_velocities_to_color_image(
            all_missing_indices_in_ocean, land_indices, valid_indices,
            U_original, V_original)

    # Create mask (these are missing points inside and outside hull)
    mask = numpy.zeros(U_original.shape, dtype=numpy.uint8)
    for i in range(all_missing_indices_in_ocean.shape[0]):
        mask[all_missing_indices_in_ocean[i, 0],
             all_missing_indices_in_ocean[i, 1]] = 1

    # Inpaint land as well as missing points. This overrides the zero values
    # that are assigned to land area.
    if numpy.any(numpy.isnan(land_indices)) is False:
        inpaint_land = False
        if inpaint_land is True:
            for i in range(land_indices.shape[0]):
                mask[land_indices[i, 0], land_indices[i, 1]] = 1

    # Inpaint
    inpainted_color_image = cv2.inpaint(color_image, mask, difusivity,
                                        cv2.INPAINT_NS)

    # Sweep the image in all directions, this flips the image left/right and
    # up/down
    if sweep_all_directions is True:

        # Flip image left/right
        inpainted_color_image = cv2.inpaint(inpainted_color_image[::-1, :, :],
                                            mask[::-1, :], difusivity,
                                            cv2.INPAINT_NS)

        # Flip left/right again to retrieve back the image
        inpainted_color_image = inpainted_color_image[::-1, :, :]

        # Flip image up/down
        inpainted_color_image = cv2.inpaint(inpainted_color_image[:, ::-1, :],
                                            mask[:, ::-1], difusivity,
                                            cv2.INPAINT_NS)

        # Flip left/right again to retrieve back the image
        inpainted_color_image = inpainted_color_image[:, ::-1, :]

        # Inpaint with no flip again
        inpainted_color_image = cv2.inpaint(inpainted_color_image, mask,
                                            difusivity, cv2.INPAINT_NS)

    # Retrieve velocities arrays
    U_inpainted_all_missing_points = _cast_uint8_array_to_float_array(
            inpainted_color_image[:, :, 0], U_original)
    V_inpainted_all_missing_points = _cast_uint8_array_to_float_array(
            inpainted_color_image[:, :, 1], V_original)

    return U_inpainted_all_missing_points, V_inpainted_all_missing_points


# ====================================
# Restore Missing Points Inside Domain
# ====================================

def restore_missing_points_inside_domain(
        missing_indices_in_ocean_inside_hull,
        missing_indices_in_ocean_outside_hull,
        land_indices,
        U_original,
        V_original,
        U_inpainted_all_missing_points,
        V_inpainted_all_missing_points):
    """
    This function takes the inpainted image, and retains only the inpainted
    points that are inside the convex hull.

    The function "InpaintAllMissingPoints" inpaints all points including inside
    and outside the convex hull. However this function discards the missing
    points that are outside the convex hull.

    masked points:
        -Points on land
        -All missing points in ocean outside hull

    Numeric points:
        -Valid points from original dataset (This does not include lan points)
        -Missing points in ocean inside hull that are inpainted.
    """

    fill_value = 999

    # Create mask of the array
    mask = numpy.zeros(U_original.shape, dtype=bool)

    # mask missing points in ocean outside hull
    for i in range(missing_indices_in_ocean_outside_hull.shape[0]):
        mask[missing_indices_in_ocean_outside_hull[i, 0],
             missing_indices_in_ocean_outside_hull[i, 1]] = True

    # mask missing/valid points on land
    if numpy.any(numpy.isnan(land_indices)) is False:
        for i in range(land_indices.shape[0]):
            mask[land_indices[i, 0], land_indices[i, 1]] = True

    # TEMPORARY: This is just for plotting in order to get PNG file.
    if numpy.any(numpy.isnan(land_indices)) is False:
        for i in range(land_indices.shape[0]):
            U_original[land_indices[i, 0], land_indices[i, 1]] = \
                numpy.ma.masked
            V_original[land_indices[i, 0], land_indices[i, 1]] = \
                numpy.ma.masked

    # Restore U
    U_inpainted_masked = numpy.ma.masked_array(U_original, mask=mask,
                                               fill_value=fill_value)
    for i in range(missing_indices_in_ocean_inside_hull.shape[0]):
        U_inpainted_masked[missing_indices_in_ocean_inside_hull[i, 0],
                           missing_indices_in_ocean_inside_hull[i, 1]] = \
                U_inpainted_all_missing_points[
                        missing_indices_in_ocean_inside_hull[i, 0],
                        missing_indices_in_ocean_inside_hull[i, 1]]

    # Restore V
    V_inpainted_masked = numpy.ma.masked_array(V_original, mask=mask,
                                               fill_value=fill_value)
    for i in range(missing_indices_in_ocean_inside_hull.shape[0]):
        V_inpainted_masked[missing_indices_in_ocean_inside_hull[i, 0],
                           missing_indices_in_ocean_inside_hull[i, 1]] = \
                V_inpainted_all_missing_points[
                        missing_indices_in_ocean_inside_hull[i, 0],
                        missing_indices_in_ocean_inside_hull[i, 1]]

    return U_inpainted_masked, V_inpainted_masked
