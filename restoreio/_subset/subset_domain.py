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
import math
from ._array_utilities import check_monotonicity, find_closest_index
from .._server_utils import terminate_with_error

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

    # Dataset bound
    dataset_min_lon = numpy.min(lon[:])
    dataset_max_lon = numpy.max(lon[:])
    dataset_min_lat = numpy.min(lat[:])
    dataset_max_lat = numpy.max(lat[:])

    # Check consistency of the bounds
    if (not math.isnan(min_lon)) and (not math.isnan(max_lon)):
        if min_lon >= max_lon:
            terminate_with_error(
                'The given min longitude (%0.4f) ' % min_lon +
                'should be smaller than the given max longitude ' +
                '(%0.4f).' % max_lon)

    if (not math.isnan(min_lat)) and (not math.isnan(max_lat)):
        if min_lat >= max_lat:
            terminate_with_error(
                'The given min latitude (%0.4f) ' % min_lat +
                'should be smaller than the given max latitude ' +
                '(%0.4f).' % max_lat)

    # min lon
    if math.isnan(min_lon):
        min_lon_index = 0
    else:
        # Check bound
        if min_lon < dataset_min_lon:
            terminate_with_error(
                'The given min longitude %0.4f ' % min_lon + 'is smaller ' +
                'then the min longitude of the data, which is %f.'
                % dataset_min_lon)

        if min_lon > dataset_max_lon:
            terminate_with_error(
                'The given min longitude %0.4f ' % min_lon + 'is larger ' +
                'then the max longitude of the data, which is %f.'
                % dataset_max_lon)

        # Find min lon index
        min_lon_index = find_closest_index(lon[:], min_lon)

    # max lon
    if math.isnan(max_lon):
        max_lon_index = lon.size-1
    else:
        # Check bound
        if max_lon > dataset_max_lon:
            terminate_with_error(
                'The given max longitude %0.4f ' % max_lon + 'is greater ' +
                'than the max longitude of the data, which is %f.'
                % dataset_max_lon)

        if max_lon < dataset_min_lon:
            terminate_with_error(
                'The given max longitude %0.4f ' % max_lon + 'is smaller ' +
                'than the min longitude of the data, which is %f.'
                % dataset_min_lon)

        # Find max lon index
        max_lon_index = find_closest_index(lon[:], max_lon)

    # min lat
    if math.isnan(min_lat):
        min_lat_index = 0
    else:
        # Check bound
        if min_lat < dataset_min_lat:
            terminate_with_error(
                'The given min latitude %0.4f ' % min_lat + 'is smaller ' +
                'than the min latitude of the data, which is %f.'
                % dataset_min_lat)

        if min_lat > dataset_max_lat:
            terminate_with_error(
                'The given min latitude %0.4f ' % min_lat + 'is larger ' +
                'than the max latitude of the data, which is %f.'
                % dataset_max_lat)

        # Find min lat index
        min_lat_index = find_closest_index(lat[:], min_lat)

    # max lat
    if math.isnan(max_lat):
        max_lat_index = lat.size-1
    else:
        # Check bound
        if max_lat > dataset_max_lat:
            terminate_with_error(
                'The given max latitude %0.4f ' % max_lat + 'is greater ' +
                'than the max latitude of the data, which is %f.'
                % dataset_max_lat)

        if max_lat < dataset_min_lat:
            terminate_with_error(
                'The given max latitude %0.4f ' % max_lat + 'is smaller ' +
                'than the min latitude of the data, which is %f.'
                % dataset_min_lat)

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
