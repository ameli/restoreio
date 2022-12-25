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

__all__ = ['cast_float_array_to_uint8_array',
           'cast_uint8_array_to_float_array']


# ===============================
# Cast Float Array to UInt8 Array
# ===============================

def cast_float_array_to_uint8_array(float_array):
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

def cast_uint8_array_to_float_array(uint8_array, original_float_array):
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
