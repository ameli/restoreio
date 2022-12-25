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

import os
import sys
import netCDF4
import numpy
import time
import datetime

__all__ = ['write_output_file']


# =================
# Prepare datetimes
# =================

def prepare_datetimes(time_indices, datetime_object):
    """
    This is used in writer function.
    Converts date char format to datetime numeric format.
    This parses the times chars and converts them to date times.
    """

    # datetimes units
    if (hasattr(datetime_object, 'units')) and (datetime_object.units != ''):
        datetimes_unit = datetime_object.units
    else:
        datetimes_unit = 'days since 1970-01-01 00:00:00 UTC'

    # datetimes calendar
    if ((hasattr(datetime_object, 'calendar')) and
            (datetime_object.calendar != '')):
        datetimes_calendar = datetime_object.calendar
    else:
        datetimes_calendar = 'gregorian'

    # datetimes
    days_list = []
    original_datetimes = datetime_object[:]

    if original_datetimes.ndim == 1:

        # datetimes in original dataset is already suitable to use
        datetimes = original_datetimes[time_indices]

    elif original_datetimes.ndim == 2:

        # Datetime in original dataset is in the form of string. They should be
        # converted to numerics
        # for i in range(original_datetimes.shape[0]):

        # If time_indices is just a single number, convert it to list.
        if type(time_indices) == int:
            time_indices = [time_indices]

        # Iteration over time indices
        for i in time_indices:

            # Get row as string (often it is already a string, or a byte type)
            char_time = numpy.chararray(original_datetimes.shape[1])
            for j in range(original_datetimes.shape[1]):
                char_time[j] = original_datetimes[i, j].astype('str')

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
            days_list.append(datetime.datetime(
                year, month, day, hour, minute, second))

        # Convert dates to numbers
        datetimes = netCDF4.date2num(days_list, units=datetimes_unit,
                                     calendar=datetimes_calendar)
    else:
        raise RuntimeError("Datetime ndim is more than 2.")

    return datetimes, datetimes_unit, datetimes_calendar


# ====================
# remove existing file
# ====================

def remove_existing_file(filename):
    """
    Removes existing output file if exists.
    """

    if os.path.exists(filename):
        os.remove(filename)


# =================
# Write Output File
# =================

def write_output_file(
        time_indices,
        datetime_object,
        longitude,
        latitude,
        mask_info,
        u_all_times_inpainted,
        v_all_times_inpainted,
        u_all_times_inpainted_error,
        v_all_times_inpainted_error,
        fill_value,
        output_filename):
    """
    Writes the inpainted array to an output netcdf file.
    """

    print("Message: Writing to NetCDF file ...")
    sys.stdout.flush()

    # Remove output file if exists
    remove_existing_file(output_filename)

    output_file = netCDF4.Dataset(output_filename, 'w',
                                  format='NETCDF4_CLASSIC')

    # Dimensions
    output_file.createDimension('time', None)
    output_file.createDimension('lon', len(longitude))
    output_file.createDimension('lat', len(latitude))

    # Prepare times from file
    datetimes, datetime_unit, datetime_calendar = prepare_datetimes(
            time_indices, datetime_object)

    # Datetime
    output_datetime = output_file.createVariable(
            'time', numpy.dtype('float64').char, ('time', ))
    output_datetime[:] = datetimes
    output_datetime.units = datetime_unit
    output_datetime.calendar = datetime_calendar
    output_datetime.standard_name = 'time'
    output_datetime._CoordinateAxisType = 'Time'
    output_datetime.axis = 'T'

    # longitude
    output_longitude = output_file.createVariable(
            'lon', numpy.dtype('float64').char, ('lon', ))
    output_longitude[:] = longitude
    output_longitude.units = 'degree_east'
    output_longitude.standard_name = 'longitude'
    output_longitude.positive = 'east'
    output_longitude._CoordinateAxisType = 'Lon'
    output_longitude.axis = 'X'
    output_longitude.coordsys = 'geographic'

    # latitude
    output_latitude = output_file.createVariable(
            'lat', numpy.dtype('float64').char, ('lat', ))
    output_latitude[:] = latitude
    output_latitude.units = 'degree_north'
    output_latitude.standard_name = 'latitude'
    output_latitude.positive = 'up'
    output_latitude._CoordinateAxisType = 'Lat'
    output_latitude.axis = 'Y'
    output_latitude.coordsys = 'geographic'

    # mask Info
    mask = output_file.createVariable(
            'mask', numpy.dtype('float64').char, ('time', 'lat', 'lon', ),
            fill_value=fill_value, zlib=True)
    mask[:] = mask_info
    mask.long_name = "Integer values at each points. \n \
            -1: Indicates points on land. These points are not used. \n \
             0: Indicates points in ocean with valid velocity data. \n \
                These points are used for restoration. \n \
             1: Indicates points in ocean inside convex/concave hull of \n \
                data domain but with missing velocity data. These points \n \
                are restored. \n \
             2: Indicates points in ocean outside convex/concave hull of \n \
                data domain but with missing velocity data. These points \n \
                are not used."
    mask.coordinates = 'longitude latitude datetime'
    mask.missing_value = fill_value
    mask.coordsys = "geographic"

    # Velocity U
    output_u = output_file.createVariable(
            'East_vel', numpy.dtype('float64').char, ('time', 'lat', 'lon', ),
            fill_value=fill_value, zlib=True)
    output_u[:] = u_all_times_inpainted
    output_u.units = 'm s-1'
    output_u.standard_name = 'surface_eastward_sea_water_velocity'
    output_u.positive = 'toward east'
    output_u.coordinates = 'longitude latitude datetime'
    output_u.missing_value = fill_value
    output_u.coordsys = "geographic"

    # Velocity V
    output_v = output_file.createVariable(
            'North_vel', numpy.dtype('float64').char, ('time', 'lat', 'lon', ),
            fill_value=fill_value, zlib=True)
    output_v[:] = v_all_times_inpainted
    output_v.units = 'm s-1'
    output_v.standard_name = 'surface_northward_sea_water_velocity'
    output_v.positive = 'toward north'
    output_v.coordinates = 'longitude latitude datetime'
    output_v.missing_value = fill_value
    output_v.coordsys = "geographic"

    # Velocity U Error
    if u_all_times_inpainted_error is not None:
        output_u_error = output_file.createVariable(
                'East_err', numpy.dtype('float64').char,
                ('time', 'lat', 'lon', ), fill_value=fill_value, zlib=True)
        output_u_error[:] = u_all_times_inpainted_error
        output_u_error.units = 'm s-1'
        output_u_error.positive = 'toward east'
        output_u_error.coordinates = 'longitude latitude datetime'
        output_u_error.missing_value = fill_value
        output_u_error.coordsys = "geographic"

    # Velocity V Error
    if v_all_times_inpainted_error is not None:
        output_v_error = output_file.createVariable(
                'North_err', numpy.dtype('float64').char,
                ('time', 'lat', 'lon', ), fill_value=fill_value, zlib=True)
        output_v_error[:] = v_all_times_inpainted_error
        output_v_error.units = 'm s-1'
        output_v_error.positive = 'toward north'
        output_v_error.coordinates = 'longitude latitude datetime'
        output_v_error.missing_value = fill_value
        output_v_error.coordsys = "geographic"

    # Global Attributes
    output_file.Conventions = 'CF-1.6'
    output_file.COORD_SYSTEM = 'GEOGRAPHIC'
    output_file.contributor_name = 'Siavash Ameli'
    output_file.contributor_email = 'sameli@berkeley.edu'
    output_file.contributor_role = 'Post process data to fill missing points.'
    output_file.institution = 'University of California, Berkeley'
    output_file.date_modified = time.strftime("%x")
    output_file.title = 'Restored missing data inside the data domain'
    output_file.source = 'Surface observation using high frequency radar.'
    output_file.summary = """The HFR original data contain missing data points
            both inside and outside the computational domain. The missing
            points that are inside a convex hull around the domain of available
            valid data points are filled. This technique uses a PDE based video
            restoration."""
    output_file.project = 'Advanced Lagrangian Predictions for Hazards' + \
        'Assessments (NSF-ALPHA)'
    output_file.acknowledgement = 'This material is based upon work ' + \
        'supported by the National Science Foundation Graduate ' + \
        'Research Fellowship under Grant No. 1520825.'
    output_file.geospatial_lat_min = "%f" % (numpy.min(latitude[:]))
    output_file.geospatial_lat_max = "%f" % (numpy.max(latitude[:]))
    output_file.geospatial_lat_units = 'degree_north'
    output_file.geospatial_lon_min = "%f" % (numpy.min(longitude[:]))
    output_file.geospatial_lon_max = "%f" % (numpy.max(longitude[:]))
    output_file.geospatial_lon_units = 'degree_east'
    output_file.geospatial_vertical_min = '0'
    output_file.geospatial_vertical_max = '0'

    output_file.time_coverage_start = \
        "%s" % (netCDF4.num2date(output_datetime[0],
                units=output_datetime.units,
                calendar=output_datetime.calendar))
    output_file.time_coverage_end = \
        "%s" % (netCDF4.num2date(output_datetime[-1],
                units=output_datetime.units,
                calendar=output_datetime.calendar))
    output_file.cdm_data_type = 'grid'

    # Close streams
    output_file.close()

    print("Wrote to: %s." % output_filename)
    print("Message: Writing to NetCDF file ... Done.")
    sys.stdout.flush()
