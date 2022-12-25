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
from ._image import convert_velocities_to_color_image
from ._cast_types import cast_uint8_array_to_float_array

__all__ = ['inpaint_missing_points_inside_domain']


# ==========================
# Inpaint All Missing Points
# ==========================

def _inpaint_all_missing_points(
        all_missing_indices_in_ocean,
        land_indices,
        valid_indices,
        U_original,
        V_original,
        diffusivity,
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
    color_image = convert_velocities_to_color_image(
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
    inpainted_color_image = cv2.inpaint(color_image, mask, diffusivity,
                                        cv2.INPAINT_NS)

    # Sweep the image in all directions, this flips the image left/right and
    # up/down
    if sweep_all_directions is True:

        # Flip image left/right
        inpainted_color_image = cv2.inpaint(inpainted_color_image[::-1, :, :],
                                            mask[::-1, :], diffusivity,
                                            cv2.INPAINT_NS)

        # Flip left/right again to retrieve back the image
        inpainted_color_image = inpainted_color_image[::-1, :, :]

        # Flip image up/down
        inpainted_color_image = cv2.inpaint(inpainted_color_image[:, ::-1, :],
                                            mask[:, ::-1], diffusivity,
                                            cv2.INPAINT_NS)

        # Flip left/right again to retrieve back the image
        inpainted_color_image = inpainted_color_image[:, ::-1, :]

        # Inpaint with no flip again
        inpainted_color_image = cv2.inpaint(inpainted_color_image, mask,
                                            diffusivity, cv2.INPAINT_NS)

    # Retrieve velocities arrays
    U_inpainted_all_missing_points = cast_uint8_array_to_float_array(
            inpainted_color_image[:, :, 0], U_original)
    V_inpainted_all_missing_points = cast_uint8_array_to_float_array(
            inpainted_color_image[:, :, 1], V_original)

    return U_inpainted_all_missing_points, V_inpainted_all_missing_points


# ===================
# Mask Outside Domain
# ===================

def _mask_outside_domain(
        missing_indices_in_ocean_inside_hull,
        missing_indices_in_ocean_outside_hull,
        land_indices,
        U_original,
        V_original,
        U_inpainted_all_missing_points,
        V_inpainted_all_missing_points):
    """
    The restored data is inpainted both inside and outside domain. However, we
    only need the inpainted data inside the domain. This function cleans the
    extra inpainted data outside the hull domain by making it masked.
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

    # Keep inpainted data inside domain for variable U
    U_inpainted_masked = numpy.ma.masked_array(U_original, mask=mask,
                                               fill_value=fill_value)
    for i in range(missing_indices_in_ocean_inside_hull.shape[0]):
        U_inpainted_masked[missing_indices_in_ocean_inside_hull[i, 0],
                           missing_indices_in_ocean_inside_hull[i, 1]] = \
                U_inpainted_all_missing_points[
                        missing_indices_in_ocean_inside_hull[i, 0],
                        missing_indices_in_ocean_inside_hull[i, 1]]

    # Keep inpainted data inside domain for variable V
    V_inpainted_masked = numpy.ma.masked_array(V_original, mask=mask,
                                               fill_value=fill_value)
    for i in range(missing_indices_in_ocean_inside_hull.shape[0]):
        V_inpainted_masked[missing_indices_in_ocean_inside_hull[i, 0],
                           missing_indices_in_ocean_inside_hull[i, 1]] = \
                V_inpainted_all_missing_points[
                        missing_indices_in_ocean_inside_hull[i, 0],
                        missing_indices_in_ocean_inside_hull[i, 1]]

    return U_inpainted_masked, V_inpainted_masked


# ====================================
# Restore Missing Points Inside Domain
# ====================================

def inpaint_missing_points_inside_domain(
        all_missing_indices_in_ocean,
        missing_indices_in_ocean_inside_hull,
        missing_indices_in_ocean_outside_hull,
        land_indices,
        valid_indices,
        U_original,
        V_original,
        diffusivity,
        sweep_all_directions):
    """
    This function takes the inpainted image, and retains only the inpainted
    points that are inside the convex hull.

    The function "inpaint_all_missing_points" inpaints all points including
    inside and outside the convex hull. However this function discards the
    missing points that are outside the convex hull.

    masked points:
        -Points on land
        -All missing points in ocean outside hull

    Numeric points:
        -Valid points from original dataset (This does not include lan points)
        -Missing points in ocean inside hull that are inpainted.
    """

    # Inpaint all missing points including inside and outside the hull domain
    U_inpainted_all_missing_points, V_inpainted_all_missing_points = \
        _inpaint_all_missing_points(
                all_missing_indices_in_ocean,
                land_indices,
                valid_indices,
                U_original,
                V_original,
                diffusivity,
                sweep_all_directions)

    # Remove (mask) those inpainted points that are outside of the hull domain
    U_inpainted_masked, V_inpainted_masked = _mask_outside_domain(
            missing_indices_in_ocean_inside_hull,
            missing_indices_in_ocean_outside_hull,
            land_indices,
            U_original,
            V_original,
            U_inpainted_all_missing_points,
            V_inpainted_all_missing_points)

    return U_inpainted_masked, V_inpainted_masked
