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

__all__ = ['refine_grid']


# ===========
# Refine Grid
# ===========

def refine_grid(
        refinement_level,
        lon,
        lat,
        U_all_times,
        V_all_times):
    """
    Increases the size of grid by the factor of refinement_level.
    The extra points on the grid will be numpy.ma.mask.

    Note that this does NOT refine the data. Rather, this just increases the
    size of grid. That is, between each two points we introduce a few grid
    points and we mask them. By masking these new points we will tend to
    restore them later.
    """

    # No refinement for level 1
    if refinement_level == 1:
        return lon, lat, U_all_times, V_all_times

    # lon
    lon_refined = numpy.zeros(refinement_level*(lon.size-1)+1,
                              dtype=float)

    for i in range(lon.size):
        # Data points
        lon_refined[refinement_level*i] = lon[i]

        # Fill in extra points
        if i < lon.size - 1:
            for j in range(1, refinement_level):
                weight = float(j)/float(refinement_level)
                lon_refined[refinement_level*i+j] = \
                    ((1.0-weight) * lon[i]) + (weight * lon[i+1])

    # lat
    lat_refined = numpy.zeros(refinement_level*(lat.size-1)+1,
                              dtype=float)

    for i in range(lat.size):
        # Data points
        lat_refined[refinement_level*i] = lat[i]

        # Fill in extra points
        if i < lat.size - 1:
            for j in range(1, refinement_level):
                weight = float(j)/float(refinement_level)
                lat_refined[refinement_level*i+j] = \
                    ((1.0-weight) * lat[i]) + (weight * lat[i+1])

    # East Velocity
    U_all_times_refined = numpy.ma.masked_all(
            (U_all_times.shape[0],
             refinement_level*(U_all_times.shape[1]-1)+1,
             refinement_level*(U_all_times.shape[2]-1)+1),
            dtype=numpy.float64)

    U_all_times_refined[:, ::refinement_level, ::refinement_level] = \
        U_all_times[:, :, :]

    # North Velocity
    V_all_times_refined = numpy.ma.masked_all(
            (V_all_times.shape[0],
             refinement_level*(V_all_times.shape[1]-1)+1,
             refinement_level*(V_all_times.shape[2]-1)+1),
            dtype=numpy.float64)

    V_all_times_refined[:, ::refinement_level, ::refinement_level] = \
        V_all_times[:, :, :]

    return lon_refined, lat_refined, U_all_times_refined, V_all_times_refined
