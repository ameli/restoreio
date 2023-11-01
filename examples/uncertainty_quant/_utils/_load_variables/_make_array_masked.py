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

__all__ = ['make_array_masked']


# =================
# Make Array masked
# =================

def make_array_masked(array, fill_value):
    """
    Often the array is not masked, but has nan or inf values. This function
    creates a masked array and mask nan and inf.

    Input:
        - array: is a 2D numpy array.
    Output:
        - array: is a 2D numpy.ma array.

    Note: array should be numpy object not netCDF object. So if you have a
          netCDF object, pass its numpy array with array[:] into this function.
    """

    if (not hasattr(array, 'mask')) or (numpy.isscalar(array.mask)):

        # Mask based on the fill values
        mask = (array >= fill_value - 1.0)

        # Mask based on nan values
        mask_nan = numpy.isnan(array)
        if mask_nan.any():
            mask = numpy.logical_or(mask, mask_nan)

        # Mask based on inf values
        mask_inf = numpy.isinf(array)
        if mask_inf.any():
            mask = numpy.logical_or(mask, mask_inf)

        array = numpy.ma.masked_array(array, mask=mask)

    else:
        # This array is masked. But check if any non-masked value is nan or inf
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    if bool(array.mask[i, j, k]) is False:
                        if numpy.isnan(array[i, j, k]) or \
                                numpy.isinf(array[i, j, k]):
                            array.mask[i, j, k] = True
                            array.data[i, j, k] = numpy.nan

    return array
