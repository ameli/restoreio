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
from .._server_utils import terminate_with_error

__all__ = ['check_monotonicity', 'find_closest_index']


# ==================
# Check Monotonicity
# ==================

def check_monotonicity(array, array_name):
    """
    Checks if an array elements are monotonically increasing.
    """

    if array.size < 2:
        terminate_with_error(
            '%s' % array_name + 'size should be at least two.')

    # Direction of array (either increasing or decreasing)
    direction = +1
    if array[1] < array[0]:
        direction = -1

    # Check if the rest of array is monotonic with the direction of the first
    # two elements of the array.
    for i in range(1, array.size-1):
        if direction * (array[i] - array[i+1]) >= 0.0:
            terminate_with_error(
                '%s' % array_name + ' array is not monotonic.')


# ==================
# Find Closest Index
# ==================

def find_closest_index(array, value):
    """
    Find the index of an array which the array element  is closets to the
    given value. This assumes the array is monotonically increasing.
    """

    diff = numpy.fabs(array - value)
    index = numpy.argmin(diff)

    return index
