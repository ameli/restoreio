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
from ._array_utilities import check_monotonicity, find_closest_index, \
    terminate_with_error

__all__ = ['subset_domain']


# =============
# Subset Domain
# =============

def subset_domain(
        lon,
        lat,
        min_lon,
        max_lon,
        min_lat,
        max_lat):
    """
    Find min and max indices to subset the processing domain.
    """

    # Check longitude and latitude arrays are monotonic
    check_monotonicity(lon[:], 'Longitudes')
    check_monotonicity(lat[:], 'Latitudes')

    # min lon
    if min_lon is None:
        min_lon_index = 0
    else:
        # Check bound
        if min_lon < numpy.min(lon[:]):
            terminate_with_error(
                'The given min longitude %0.4f ' % min_lon + 'is smaller ' +
                'then the min longitude of the data, which is %f.' % lon[0])

        # Find min lon index
        min_lon_index = find_closest_index(lon[:], min_lon)

    # max lon
    if max_lon is None:
        max_lon_index = lon.size-1
    else:
        # Check bound
        if max_lon > numpy.max(lon[:]):
            terminate_with_error(
                'The given max longitude %0.4f ' % max_lon + 'is greater ' +
                'than the max longitude of the data, which is %f.' % lon[-1])

        # Find max lon index
        max_lon_index = find_closest_index(lon[:], max_lon)

    # min lat
    if min_lat is None:
        min_lat_index = 0
    else:
        # Check bound
        if min_lat < numpy.min(lat[:]):
            terminate_with_error(
                'The given min latitude %0.4f ' % min_lat + 'is smaller ' +
                'than the min latitude of the data, which is %f.' % lat[0])

        # Find min lat index
        min_lat_index = find_closest_index(lat[:], min_lat)

    # max lat
    if max_lat is None:
        max_lat_index = lat.size-1
    else:
        # Check bound
        if max_lat > numpy.max(lat[:]):
            terminate_with_error(
                'The given max latitude %0.4f ' % max_lat + 'is greater ' +
                'than the max latitude of the data, which is %f.' % lat[-1])

        # Find max lat index
        max_lat_index = find_closest_index(lat[:], max_lat)

    # In some dataset, the direction of lon or lat arrays are reversed, meaning
    # they are decreasing monotonic. In such a case, the min or max indices
    # should be swapped.

    # Swap min and max lon indices
    if min_lon_index > max_lon_index:
        temp_min_lon_index = min_lon_index
        min_lon_index = max_lon_index
        max_lon_index = temp_min_lon_index

    # Swap min and max lat indices
    if min_lat_index > max_lat_index:
        temp_min_lat_index = min_lat_index
        min_lat_index = max_lat_index
        max_lat_index = temp_min_lat_index

    return min_lon_index, max_lon_index, min_lat_index, max_lat_index
