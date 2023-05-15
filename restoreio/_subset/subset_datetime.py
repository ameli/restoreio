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

from ._array_utilities import check_monotonicity, find_closest_index
from datetime import datetime
import netCDF4

__all__ = ['subset_datetime']


# ==================
# Get Datetime Index
# ==================

def _get_datetime_index(datetime_string, datetime_info):
    """
    Returns time index compared to an array of datetimes.
    """

    # datetime string format
    datetime_format = '%Y-%m-%dT%H:%M:%S'

    # Time object
    time_obj = datetime.strptime(datetime_string, datetime_format)

    # Convert object to number since an epoch given by the unit of the dataset
    # for example in the unit of days since 1970-01-01 00:00:00
    time_value = netCDF4.date2num(time_obj, units=datetime_info['unit'],
                                  calendar=datetime_info['calendar'])

    # Find index of time
    datetime_index = find_closest_index(datetime_info['array'],
                                        float(time_value))

    return datetime_index


# ===========
# Subset Time
# ===========

def subset_datetime(
        datetime_info,
        min_time,
        max_time,
        time):
    """
    Find min and max indices to subset the processing time interval.
    """

    datetime_array = datetime_info['array']

    # Check longitude and latitude arrays are monotonic
    check_monotonicity(datetime_array, 'Time')

    if (min_time == '') and (max_time == '') and (time == ''):

        # When nothing is specified, use the whole data
        min_datetime_index = 0
        max_datetime_index = datetime_array.size - 1

    elif time != '':

        # This is when a single time is specified (such as in uncertainty
        # quantification or plotting where only time time point is used).

        # When single time point is specified, a time interval should not also
        # be specified.
        if (min_time != '') or (max_time != ''):
            raise ValueError(
                'When "time" argument is specified, the other time ' +
                'arguments "min_time" and "max_time" cannot be specified ' +
                'and vice versa.')

        # Get time index of the given time string
        time_index = _get_datetime_index(time, datetime_info)

        # Make interval of length zero with the same time for the start and end
        min_datetime_index = time_index
        max_datetime_index = time_index

    else:

        # Process min time
        if min_time == '':
            min_datetime_index = 0
        else:
            min_datetime_index = _get_datetime_index(min_time, datetime_info)

        # Process max time
        if max_time == '':
            max_datetime_index = datetime_array.size - 1
        else:
            max_datetime_index = _get_datetime_index(max_time, datetime_info)

    return min_datetime_index, max_datetime_index
