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

# ===============================
# Cast Float Array to UInt8 Array
# ===============================

def CastFloatArrayToUInt8Array(FloatArray):
    """
    Casts float array to UInt8. Here the float range of data is mapped to the integer range 0-255 linearly.
    """

    MinArray = numpy.min(FloatArray)
    MaxArray = numpy.max(FloatArray)

    # Scale array into the rage of 0 to 255
    ScaledFloatArray = 255.0 * (FloatArray - MinArray) / (MaxArray - MinArray)

    # Cast float array to uint8 array
    UInt8Array = (ScaledFloatArray + 0.5).astype(numpy.uint8)

    return UInt8Array

# ===============================
# Cast UInt8 Array To Float Array
# ===============================

def CastUInt8ArrayToFloatArray(UInt8Array, OriginalFloatArray):
    """
    Casts UInt8 array to float array. Here, the second argument ""OriginalFLoarArray" is used to 
    find the range of data. So that the range 0-255 to mapped to the range of data linearly.
    """

    MinFloat = numpy.min(OriginalFloatArray)
    MaxFloat = numpy.max(OriginalFloatArray)

    ScaledFloatArray = UInt8Array.astype(float)
    FloatArray = MinFloat + ScaledFloatArray * (MaxFloat - MinFloat) / 255.0

    return FloatArray

# =================================
# Convert Velocities To Color Image
# =================================

def ConvertVelocitiesToColorImage( \
        AllMissingIndicesInOcean, \
        LandIndices, \
        ValidIndices, \
        U_Original, \
        V_Original):
    """
    Takes two arrays of U and V (each 2D numpy array), converts them into grayscale 8-bit array, and then 
    add them into a 3-channel color image. The third channel is filled with zeros.

    Note: U and V are assumed to be 2D arrays, not 3D array.
    
    The both U and V arrays are set to zero on land area. NOTE that this does not result in black image on land.
    This is becaseu zero on array is not mapped to zero on image. Zero im image (black) is corresponding to the min 
    value of the U or V arrays.

    To plot the image results, set PlotImages=True.
    """

    # Get mean value of U
    Valid_U = U_Original[ValidIndices[:, 0], ValidIndices[:, 1]]
    Mean_U = numpy.mean(Valid_U)

    # Fill U with mean values
    Filled_U = numpy.zeros(U_Original.shape, dtype=float)

    # Use original U for valid points
    for i in range(ValidIndices.shape[0]):
        Filled_U[ValidIndices[i, 0], ValidIndices[i, 1]] = U_Original[ValidIndices[i, 0], ValidIndices[i, 1]]

    # Use U_mean for missing points in ocean
    for i in range(AllMissingIndicesInOcean.shape[0]):
        Filled_U[AllMissingIndicesInOcean[i, 0], AllMissingIndicesInOcean[i, 1]] = Mean_U

    # Zero out the land indices
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):
            Filled_U[LandIndices[i, 0], LandIndices[i, 1]] = 0.0

    # Get mean values of V
    Valid_V = V_Original[ValidIndices[:, 0], ValidIndices[:, 1]]
    Mean_V = numpy.mean(Valid_V)

    # Fill V with mean values
    Filled_V = numpy.zeros(U_Original.shape, dtype=float)

    # Use original V for valid points
    for i in range(ValidIndices.shape[0]):
        Filled_V[ValidIndices[i, 0], ValidIndices[i, 1]] = V_Original[ValidIndices[i, 0], ValidIndices[i, 1]]

    # Use mean V for missing points in ocean
    for i in range(AllMissingIndicesInOcean.shape[0]):
        Filled_V[AllMissingIndicesInOcean[i, 0], AllMissingIndicesInOcean[i, 1]] = Mean_V

    # Zero out the land indices
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):
            Filled_V[LandIndices[i, 0], LandIndices[i, 1]] = 0.0

    # Create gray scale image for each U and V
    GrayScaleImage_U = CastFloatArrayToUInt8Array(Filled_U)
    GrayScaleImage_V = CastFloatArrayToUInt8Array(Filled_V)

    # Create color image from both gray scales U and V
    ColorImage = numpy.zeros((U_Original.shape[0], U_Original.shape[1], 3), dtype=numpy.uint8)
    ColorImage[:, :, 0] = GrayScaleImage_U
    ColorImage[:, :, 1] = GrayScaleImage_V

    # Plot images. To plot, set PlotImages to True.
    PlotImages = False
    if PlotImages:
        PlotColorAndGrayscaleImages(GrayscaleImage_U, GrayScaleImage_V, ColorImage)

    return ColorImage

# ==========================
# Inpaint All Missing Points
# ==========================

def InpaintAllMissingPoints( \
        AllMissingIndicesInOcean, \
        LandIndices, \
        ValidIndices, \
        U_Original, \
        V_Original, \
        Difusivity, \
        SweepAllDirections):
    """
    This function uses opencv.inpaint to restore the colored images.
    The colored images are obtained from adding 2 grayscale images from velocities U and V and a 0 (null) channel.
    In this method, ALL missing points in ocean, inclusing those inside and outside the hull.

    There are two parameters:

    Difusivity: This is the same as Reynolds number in NS.
    SweepAllDirections: If set to True, the image is inpainted 4 times as follow:
        1. Original orientation of image
        2. Flipped left/right orientation of image
        3. Filled up/down orientation of image
        4. Again the original orientation og image

    Note: If in this function we set InpaintLand=True, it inpaints land area as well. If not, the land is already zer to zero
    and it considers it as a known value on the image.
    """

    # Create 8-bit 3-channel image from U and V
    ColorImage = ConvertVelocitiesToColorImage(AllMissingIndicesInOcean, LandIndices, ValidIndices, U_Original, V_Original)

    # Create Mask (these are missing points inside and outside hull)
    Mask = numpy.zeros(U_Original.shape, dtype=numpy.uint8)
    for i in range(AllMissingIndicesInOcean.shape[0]):
        Mask[AllMissingIndicesInOcean[i, 0], AllMissingIndicesInOcean[i, 1]] = 1

    # Inpaint land as well as missing points. This overrides tha zero values that are assigned to land area.
    if numpy.any(numpy.isnan(LandIndices)) == False:
        InpaintLand = False
        if InpaintLand == True:
            for i in range(LandIndices.shape[0]):
                Mask[LandIndices[i, 0], LandIndices[i, 1]] = 1

    # Inpaint
    InpaintedColorImage = cv2.inpaint(ColorImage, Mask, Difusivity, cv2.INPAINT_NS)

    # Sweep the image in all directions, this flips the image left/right and up/down
    if SweepAllDirections == True:

        # Flip image left/right
        InpaintedColorImage = cv2.inpaint(InpaintedColorImage[::-1, :, :], Mask[::-1, :], Difusivity, cv2.INPAINT_NS)

        # Flip left/right again to retrive back the image
        InpaintedColorImage = InpaintedColorImage[::-1, :, :]

        # Flip image up/down
        InpaintedColorImage = cv2.inpaint(InpaintedColorImage[:, ::-1, :], Mask[:, ::-1], Difusivity, cv2.INPAINT_NS)

        # Flip left/right again to retrive back the image
        InpaintedColorImage = InpaintedColorImage[:, ::-1, :]

        # Inpaint with no flip again
        InpaintedColorImage = cv2.inpaint(InpaintedColorImage, Mask, Difusivity, cv2.INPAINT_NS)

    # Retrieve velocities arrays
    U_InpaintedAllMissingPoints = CastUInt8ArrayToFloatArray(InpaintedColorImage[:, :, 0], U_Original)
    V_InpaintedAllMissingPoints = CastUInt8ArrayToFloatArray(InpaintedColorImage[:, :, 1], V_Original)

    return U_InpaintedAllMissingPoints, V_InpaintedAllMissingPoints

# ====================================
# Restore Missing Points Inside Domain
# ====================================

def RestoreMissingPointsInsideDomain( \
        MissingIndicesInOceanInsideHull, \
        MissingIndicesInOceanOutsideHull, \
        LandIndices, \
        U_Original, \
        V_Original, \
        U_InpaintedAllMissingPoints, \
        V_InpaintedAllMissingPoints):
    """
    This function takes the inpainted image, and retains only the inpainted points that are inside the convex hull.

    The function "InpaintAllMissingPoints" inpaints all points including inside and outside the convex hull. However
    this function discards the missing points that are outside the convex hull.

    Masked points:
        -Points on land
        -All missing points in ocean outside hull

    Numeric points:
        -Valid points from original dataset (This does not include lan points)
        -Missing points in ocean inside hull that are inpainted.
    """

    FillValue = 999

    # Create mask of the array
    Mask = numpy.zeros(U_Original.shape, dtype=bool)

    # Mask missing points in ocean outside hull
    for i in range(MissingIndicesInOceanOutsideHull.shape[0]):
        Mask[MissingIndicesInOceanOutsideHull[i, 0], MissingIndicesInOceanOutsideHull[i, 1]] = True

    # Mask missing/valid points on land
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):
            Mask[LandIndices[i, 0], LandIndices[i, 1]] = True

    # TEMPORARY: This is just for plotting in order to get PNG file.
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):
            U_Original[LandIndices[i, 0], LandIndices[i, 1]] = numpy.ma.masked
            V_Original[LandIndices[i, 0], LandIndices[i, 1]] = numpy.ma.masked

    # Restore U
    U_Inpainted_Masked = numpy.ma.masked_array(U_Original, mask=Mask, fill_value=FillValue)
    for i in range(MissingIndicesInOceanInsideHull.shape[0]):
        U_Inpainted_Masked[MissingIndicesInOceanInsideHull[i, 0], MissingIndicesInOceanInsideHull[i, 1]] = \
                U_InpaintedAllMissingPoints[MissingIndicesInOceanInsideHull[i, 0], MissingIndicesInOceanInsideHull[i, 1]]

    # Restore V
    V_Inpainted_Masked = numpy.ma.masked_array(V_Original, mask=Mask, fill_value=FillValue)
    for i in range(MissingIndicesInOceanInsideHull.shape[0]):
        V_Inpainted_Masked[MissingIndicesInOceanInsideHull[i, 0], MissingIndicesInOceanInsideHull[i, 1]] = \
                V_InpaintedAllMissingPoints[MissingIndicesInOceanInsideHull[i, 0], MissingIndicesInOceanInsideHull[i, 1]]

    return U_Inpainted_Masked, V_Inpainted_Masked
