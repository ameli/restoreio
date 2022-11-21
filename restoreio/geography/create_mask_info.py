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

__all__ = ['create_mask_info']


# ================
# Create Mask Info
# ================

def create_mask_info(
            U_one_time,
            land_indices,
            missing_indices_in_ocean_inside_hull,
            missing_indices_in_ocean_outside_hull,
            valid_indices):
    """
    Create a masked array.

    0:  Valid Indices
    1:  missing_indices_in_ocean_inside_hull
    2:  missing_indices_in_ocean_outside_hull
    -1: land_indices
    """

    # zero for all valid indices
    mask_info = numpy.zeros(U_one_time.shape, dtype=int)

    # Missing indices in ocean inside hull
    for i in range(missing_indices_in_ocean_inside_hull.shape[0]):
        mask_info[missing_indices_in_ocean_inside_hull[i, 0],
                  missing_indices_in_ocean_inside_hull[i, 1]] = 1

    # Missing indices in ocean outside hull
    for i in range(missing_indices_in_ocean_outside_hull.shape[0]):
        mask_info[missing_indices_in_ocean_outside_hull[i, 0],
                  missing_indices_in_ocean_outside_hull[i, 1]] = 2

    # Land indices
    if numpy.any(numpy.isnan(land_indices)) is False:
        for i in range(land_indices.shape[0]):
            mask_info[land_indices[i, 0], land_indices[i, 1]] = -1

    return mask_info
