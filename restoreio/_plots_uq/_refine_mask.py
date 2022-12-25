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
from scipy.interpolate import NearestNDInterpolator

__all__ = ['refine_mask']


# ===========
# Refine Mask
# ===========

def refine_mask(
        lon,
        lat,
        missing_indices_in_ocean_inside_hull):
    """
    Refines the mask grid. This is used for the purpose of plotting a contour
    on the edges of mask pixels. To do so, a mask array of 0 and 1 is created
    where the locations of missing points "inside the hull" are 1, and zero
    elsewhere. The contour line with the value 0.5 is then plotted to draw the
    boundary of the missing point pixels. Unfortunately, the contour on such
    mask array is not following the sharp square boundaries of the pixels. To
    fix this, we create a refined mask array with higher resolution. We use
    resolutions with factors of multiples of 5 (such as 5, 10, 15, etc) since
    they create best results with contour value 0.5.

    Note that the mask is defined based on the missing points "inside the
    hull", not all missing points.
    """

    refined_mask_data = {}
    num_missing_points = missing_indices_in_ocean_inside_hull.shape[0]

    # Draw edges lines around mask pixels
    if num_missing_points > 1:

        # Bounds and size of the domain
        lon_min = numpy.min(lon)
        lon_max = numpy.max(lon)
        lat_min = numpy.min(lat)
        lat_max = numpy.max(lat)

        # Grid size
        lon_size = lon.size
        lat_size = lat.size

        # Grid
        lons_grid, lats_grid = numpy.meshgrid(lon, lat)

        # Create mask of missing indices in ocean inside hull
        mask = numpy.zeros((lon_size, lat_size), dtype=float)
        for point_id in range(num_missing_points):
            i = missing_indices_in_ocean_inside_hull[point_id, 0]
            j = missing_indices_in_ocean_inside_hull[point_id, 1]
            mask[i, j] = 1

        # Interpolated grid with 5 times more points on each ax
        lon_lat = zip(lons_grid.ravel(), lats_grid.ravel())
        interp = NearestNDInterpolator(list(lon_lat), mask.ravel())

        # Use multiples of 5 as they make sharp contour edges when the
        # level-set with the value 0.5 is sought.
        if lon_size < 50 and lat_size < 50:
            refine = 25
        elif lon_size < 100 and lat_size < 100:
            refine = 15
        else:
            refine = 5

        refined_lon = numpy.linspace(lon_min, lon_max, refine*(lon_size-1)+1)
        refined_lat = numpy.linspace(lat_min, lat_max, refine*(lat_size-1)+1)
        refined_lons_grid, refined_lats_grid = numpy.meshgrid(
                refined_lon, refined_lat)

        # Interpolate refined mask from the nearest neighbors of original mask
        refined_mask = interp(refined_lons_grid, refined_lats_grid)

        refined_mask_data = {
            'refined_lons_grid': refined_lons_grid,
            'refined_lats_grid': refined_lats_grid,
            'refined_mask': refined_mask
        }

    return refined_mask_data
