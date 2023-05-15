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

import netCDF4
import datetime
import numpy

__all__ = ['get_datetime_info']


# =================
# Get Datetime Info
# =================

def get_datetime_info(datetimes_obj):
    """
    This is used in writer function. Converts date char format to datetime
    numeric format. This parses the times chars and converts them to date
    times.
    """

    # Datetimes units
    if (hasattr(datetimes_obj, 'units')) and \
       (datetimes_obj.units != ''):
        datetimes_unit = datetimes_obj.units
    else:
        datetimes_unit = 'days since 1970-01-01 00:00:00 UTC'

    # Datetimes calendar
    if (hasattr(datetimes_obj, 'calendar')) and \
       (datetimes_obj.calendar != ''):
        datetimes_calendar = datetimes_obj.calendar
    else:
        datetimes_calendar = 'gregorian'

    # Datetimes
    days_list = []
    original_datetimes_array = datetimes_obj[:]

    if original_datetimes_array.ndim == 1:

        # Datetimes in original dataset is already suitable to use
        datetimes_array = original_datetimes_array[:].astype(numpy.double)

    elif original_datetimes_array.ndim == 2:

        # Datetime in original dataset is in the form of string. They
        # should be converted to numerics
        for i in range(original_datetimes_array.shape[0]):

            # Get row as string (often it is already a string, or a byte
            # type)
            char_time = numpy.chararray(original_datetimes_array.shape[1])
            for j in range(original_datetimes_array.shape[1]):
                char_time[j] = original_datetimes_array[i, j].astype('str')

            # Parse chars to integers

            # Year
            if char_time.size >= 4:
                year = int(char_time[0] + char_time[1] + char_time[2] +
                           char_time[3])
            else:
                year = 1970

            # Month
            if char_time.size >= 6:
                month = int(char_time[5] + char_time[6])
            else:
                month = 1

            # Day
            if char_time.size >= 9:
                day = int(char_time[8] + char_time[9])
            else:
                day = 1

            # Hour
            if char_time.size >= 13:
                hour = int(char_time[11] + char_time[12])
            else:
                hour = 0

            # Minute
            if char_time.size >= 15:
                minute = int(char_time[14] + char_time[15])
            else:
                minute = 0

            # Second
            if char_time.size >= 19:
                second = int(char_time[17] + char_time[18])
            else:
                second = 0

            # Create day object
            days_list.append(datetime.datetime(year, month, day, hour,
                                               minute, second))

        # Convert dates to numbers
        datetimes_array = netCDF4.date2num(
            days_list, units=datetimes_unit,
            calendar=datetimes_calendar).astype(numpy.double)
    else:
        raise RuntimeError("Datetime ndim is more than 2.")

    # Create a Time dictionary
    datetime_info = {
        'array': datetimes_array,
        'unit': datetimes_unit,
        'calendar': datetimes_calendar
    }

    return datetime_info
