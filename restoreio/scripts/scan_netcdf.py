#! /usr/bin/env python

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
import sys
import warnings
import netCDF4
import pyncml
import os.path
import getopt
import datetime
import json

try:
    # For python 3
    from urllib.parse import urlparse
except ImportError:
    # For python 2
    from urlparse import urlparse


# ====================
# Terminate With Error
# ====================

def _terminate_with_error(message):
    """
    Returns an incomplete Json object with ScanStatus = False, and an error
    message.
    """

    # Fill output with defaults
    dataset_info_dict = {
        "Scan": {
            "ScanStatus": False,
            "Message": message
        }
    }

    # Print out and exit gracefully
    dataset_info_json = json.dumps(dataset_info_dict, indent=4)
    print(dataset_info_json)
    sys.stdout.flush()
    sys.exit()


# ===============
# Parse Arguments
# ===============

def _parse_arguments(argv):
    """
    Parses the argument of the executable and obtains the filename.

    Input file is netcdf nc file or ncml file.
    Output is a strigified Json.
    """

    # -------------
    # Print Version
    # -------------

    def _print_version():

        version_string = \
                """
Version 0.0.1
Siavash Ameli
University of California, Berkeley
                """

        print(version_string)

    # -----------
    # Print Usage
    # -----------

    def _print_usage(exec_name):
        usage_string = "Usage: " + exec_name + \
                " -i <input_filename.{nc, ncml}>"
        options_string = \
            """
Required arguments:

    -i --input          Input filename. This should be full path. Input file
                        extension can be *.nc or *.ncml.

Optional arguments:

    -h --help           Prints this help message.
    -v --version        Prints the version and author info.
    -V --Velocity       Scans Velocities in the dataset.
                """
        example_string = \
            """
Examples:

    1. Using a url on a remote machine. This will not scan the velocities.
       $ %s -i <url>

    2. Using a filename on the local machine. This will not scan the velocities
       $ %s -i <filename>

    3. Scan velocities form a url or filename:
       $ %s -V -i <url or filename>
                """ % (exec_name, exec_name, exec_name)

        print(usage_string)
        print(options_string)
        print(example_string)

    # -----------------

    # Initialize variables (defaults)
    input_filename = ''
    scan_velocity_status = False

    try:
        opts, args = getopt.getopt(argv[1:], "hvVi:",
                                   ["help", "version", "Velocity", "input="])
    except getopt.GetoptError:
        _print_usage(argv[0])
        sys.exit(2)

    # Assign options
    for opt, arg in opts:

        if opt in ('-h', '--help'):
            _print_usage(argv[0])
            sys.exit()
        elif opt in ('-v', '--version'):
            _print_version()
            sys.exit()
        elif opt in ("-i", "--input"):
            input_filename = arg
        elif opt in ("-V", "--Velocity"):
            scan_velocity_status = True

    # Check Arguments
    if len(argv) < 2:
        _print_usage(argv[0])
        sys.exit(0)

    # Check input_filename
    if (input_filename == ''):
        _print_usage(argv[0])
        raise RuntimeError("input_filename is empty.")

    return input_filename, scan_velocity_status


# ==================
# Load Local Dataset
# ==================

def _load_local_dataset(filename):
    """
    Opens either ncml or nc file and returns the aggregation file object.
    """

    # Check file extension
    file_extension = os.path.splitext(filename)[1]
    if file_extension == '.ncml':

        # Change directory
        data_directory = os.path.dirname(filename)
        current_directory = os.getcwd()
        os.chdir(data_directory)

        # NCML
        ncml_string = open(filename, 'r').read()
        ncml_string = ncml_string.encode('ascii')
        ncml = pyncml.etree.fromstring(ncml_string)
        nc = pyncml.scan(ncml=ncml)

        # Get nc files list
        files_list = [f.path for f in nc.members]
        os.chdir(current_directory)

        # Aggregate
        agg = netCDF4.MFDataset(files_list, aggdim='t')
        return agg

    elif file_extension == '.nc':

        nc = netCDF4.Dataset(filename)
        return nc

    else:
        raise RuntimeError("File should be either ncml or nc. filename: " +
                           filename)


# ===================
# Load Remote Dataset
# ===================

def _load_remote_dataset(url):
    """
    URL can be point to a *.nc or *.ncml file.
    """

    # Check URL is opendap
    if (url.startswith('http://') is False) and \
       (url.startswith('https://') is False):
        _terminate_with_error('Input data URL does not seem to be a URL. ' +
                              'A URL should start with <code>http://</code> ' +
                              'or <code>https://</code>.')
    elif ("/thredds/dodsC/" not in url) and ("opendap" not in url):
        _terminate_with_error('Input data URL is not an <b>OpenDap</b> URL ' +
                              'or is not hosted on a THREDDs server. Check ' +
                              'if your data URL contains ' +
                              '<code>/thredds/dodsC/</code> or ' +
                              '<code>/opendap/</code>.')
    elif url.endswith(('.nc', '.nc.gz', '.ncd', '.ncml', '.ncml.gz')) is False:
        _terminate_with_error('The input data URL is not a <i>netcdf</i> ' +
                              'file. The URL should end with ' +
                              '<code>.nc</code>, <code>.nc.gz</code>, ' +
                              '<code>.ncml</code> or <code>.ncml.gz</code>.')

    # Check file extension
    file_extension = os.path.splitext(url)[1]

    # Case of zipped files (get the correct file extension before the '.gz')
    if file_extension == ".gz":
        file_extension = os.path.splitext(url[:-3])[1]

    if file_extension == '.ncml':

        # nc = open_url(url)
        nc = netCDF4.Dataset(url)

    elif (file_extension == '.nc') or (file_extension == '.ncd'):

        nc = netCDF4.Dataset(url)

    else:
        _terminate_with_error('File extension in the data URL should be ' +
                              'either <code>.nc</code>, ' +
                              '<code>.nc.gz</code>, <code>.ncml</code> or ' +
                              '<code>.ncml.gz</code>.')

    return nc


# ============
# Load Dataset
# ============

def _load_dataset(input_filename):
    """
    Dispatches the execution to either of the following two functions:
    1. LoadMultiFileDataset: For files where the input_filename is a path on
       the local machine.
    2. _load_remote_dataset: For files remotely where input_filename is a URL.
    """

    # Check input filename
    if input_filename == '':
        _terminate_with_error('Input data URL is empty. You should provide ' +
                              'an OpenDap URL.')

    # Check if the input_filename has a "host" name
    if bool(urlparse(input_filename).netloc):
        # input_filename is a URL
        return _load_remote_dataset(input_filename)
    else:
        # input_filename is a path
        return _load_local_dataset(input_filename)


# ===============
# Search Variable
# ===============

def _search_variable(agg, names_list, standard_names_list):
    """
    This function searches for a list of names and standard names to match a
    variable.

    Note: All strings are compared with their lowercase form.
    """

    variable_found = False
    obj_name = ''
    obj_standard_name = ''

    # Search among standard names list
    for standard_name in standard_names_list:
        for key in agg.variables.keys():
            variable = agg.variables[key]
            if hasattr(variable, 'standard_name'):
                standard_name_in_agg = variable.standard_name
                if standard_name.lower() == standard_name_in_agg.lower():
                    obj = agg.variables[key]
                    obj_name = obj.name
                    obj_standard_name = obj.standard_name
                    variable_found = True
                    break
        if variable_found is True:
            break

    # Search among names list
    if variable_found is False:
        for name in names_list + standard_names_list:
            for key in agg.variables.keys():
                if name.lower() == key.lower():
                    obj = agg.variables[key]
                    obj_name = obj.name
                    if hasattr(obj, 'standard_name'):
                        obj_standard_name = obj.standard_name
                    variable_found = True
                    break
            if variable_found is True:
                break

    # Last check to see if the variable is found
    if variable_found is False:
        return None

    return obj, obj_name, obj_standard_name


# =============================
# Load Time And Space Variables
# =============================

def _load_time_and_space_variables(agg):
    """
    Finds the following variables from the aggregation object agg.

    - Time
    - Longitude
    - Latitude
    """

    # Time
    time_names_list = ['time', 'datetime', 't']
    time_standard_names_list = ['time']
    datetime_obj, datetime_name, datetime_standard_name = _search_variable(
            agg, time_names_list, time_standard_names_list)

    # Check time variable
    if datetime_obj is None:
        _terminate_with_error('Can not find the <i>time</i> variable in ' +
                              'the netcdf file.')
    elif hasattr(datetime_obj, 'units') is False:
        _terminate_with_error('The <t>time</i> variable does not have ' +
                              '<i>units</i> attribute.')
    # elif hasattr(datetime_obj, 'calendar') is False:
    #     _terminate_with_error('The <t>time</i> variable does not have ' +
    #                           '<i>calendar</i> attribute.')
    elif datetime_obj.size < 2:
        _terminate_with_error('The <i>time</i> variable size should be at ' +
                              'least <tt>2</tt>.')

    # Longitude
    longitude_names_list = ['longitude', 'lon', 'long']
    longitude_standard_names_list = ['longitude']
    longitude_obj, longitude_name, longitude_standard_name = _search_variable(
            agg, longitude_names_list, longitude_standard_names_list)

    # Check longitude variable
    if longitude_obj is None:
        _terminate_with_error('Can not find the <i>longitude</i> variable ' +
                              'in the netcdf file.')
    elif len(longitude_obj.shape) != 1:
        _terminate_with_error('The <t>longitude</i> variable dimension ' +
                              'should be <tt>1<//t>.')
    elif longitude_obj.size < 2:
        _terminate_with_error('The <i>longitude</i> variable size should ' +
                              'be at least <tt>2</tt>.')

    # Latitude
    latitude_names_list = ['latitude', 'lat']
    latitude_standard_names_list = ['latitude']
    latitude_obj, latitude_name, latitude_standard_name = _search_variable(
            agg, latitude_names_list, latitude_standard_names_list)

    # Check latitude variable
    if latitude_obj is None:
        _terminate_with_error('Can not find the <i>latitude</i> variable in ' +
                              'the netcdf file.')
    elif len(latitude_obj.shape) != 1:
        _terminate_with_error('The <t>latitude</i> variable dimension ' +
                              'should be <tt>1<//t>.')
    elif latitude_obj.size < 2:
        _terminate_with_error('The <i>latitude</i> variable size should ' +
                              'be at least <tt>2</tt>.')

    return datetime_obj, longitude_obj, latitude_obj


# =======================
# Load Velocity Variables
# =======================

def _load_velocity_variables(agg):
    """
    Finds the following variables from the aggregation object agg.

    - Eastward velocity U
    - Northward velocity V
    """
    # East Velocity
    east_velocity_names_list = ['east_vel', 'eastward_vel', 'u',
                                'east_velocity', 'eastward_velocity']
    east_velocity_standard_names_list = ['surface_eastward_sea_water_velocity',
                                         'eastward_sea_water_velocity']
    east_velocity_obj, east_velocity_name, east_velocity_standard_name = \
        _search_variable(agg, east_velocity_names_list,
                         east_velocity_standard_names_list)

    # North Velocity
    north_velocity_names_list = ['north_vel', 'northward_vel', 'v',
                                 'north_velocity', 'northward_velocity']
    north_velocity_standard_names_list = [
            'surface_northward_sea_water_velocity',
            'northward_sea_water_velocity']
    north_velocity_obj, north_velocity_name, north_velocity_standard_name = \
        _search_variable(agg, north_velocity_names_list,
                         north_velocity_standard_names_list)

    return east_velocity_obj, north_velocity_obj, east_velocity_name, \
        north_velocity_name, east_velocity_standard_name, \
        north_velocity_standard_name


# =================
# Prepare datetimes
# =================

def _prepare_datetimes(datetime_obj):
    """
    This is used in writer function.
    Converts date char format to datetime numeric format.
    This parses the times chars and converts them to date times.
    """

    # datetimes units
    if (hasattr(datetime_obj, 'units')) and (datetime_obj.units != ''):
        datetimes_unit = datetime_obj.units
    else:
        datetimes_unit = 'days since 1970-01-01 00:00:00 UTC'

    # datetimes calendar
    if (hasattr(datetime_obj, 'calendar')) and \
            (datetime_obj.calendar != ''):
        datetimes_calendar = datetime_obj.calendar
    else:
        datetimes_calendar = 'gregorian'

    # datetimes
    days_list = []
    original_datetimes = datetime_obj[:]

    if original_datetimes.ndim == 1:

        # datetimes in original dataset is already suitable to use
        datetimes = original_datetimes

    elif original_datetimes.ndim == 2:

        # Datetime in original dataset is in the form of string. They should be
        # converted to numerics
        for i in range(original_datetimes.shape[0]):

            # Get row as string (often it is already a string, or a byte type)
            char_time = numpy.chararray(original_datetimes.shape[1])
            for j in range(original_datetimes.shape[1]):
                char_time[j] = original_datetimes[i, j].astype('str')

            # Parse chars to integers
            year = int(char_time[0] + char_time[1] + char_time[2] +
                       char_time[3])
            month = int(char_time[5] + char_time[6])
            day = int(char_time[8] + char_time[9])
            hour = int(char_time[11] + char_time[12])
            minute = int(char_time[14] + char_time[15])
            second = int(char_time[17] + char_time[18])

            # Create Day object
            days_list.append(datetime.datetime(
                year, month, day, hour, minute, second))

        # Convert dates to numbers
        datetimes = netCDF4.date2num(days_list, units=datetimes_unit,
                                     calendar=datetimes_calendar)
    else:
        raise RuntimeError("Datetime ndim is more than 2.")

    return datetimes, datetimes_unit, datetimes_calendar


# =============
# Get Time Info
# =============

def _get_time_info(datetime_obj):
    """
    Get the initial time info and time duration.
    """

    datetimes, datetimes_unit, datetimes_calendar = \
        _prepare_datetimes(datetime_obj)

    # Initial time
    initial_time = datetimes[0]
    initial_datetime_obj = netCDF4.num2date(
            initial_time, units=datetimes_unit, calendar=datetimes_calendar)

    initial_time_dict = {
        "Year": str(initial_datetime_obj.year).zfill(4),
        "Month": str(initial_datetime_obj.month).zfill(2),
        "Day": str(initial_datetime_obj.day).zfill(2),
        "Hour": str(initial_datetime_obj.hour).zfill(2),
        "Minute": str(initial_datetime_obj.minute).zfill(2),
        "Second": str(initial_datetime_obj.second).zfill(2),
        "Microsecond": str(initial_datetime_obj.microsecond).zfill(6)
    }

    # Round off with microsecond
    if int(initial_time_dict['Microsecond']) > 500000:
        initial_time_dict['Microsecond'] = '000000'
        initial_time_dict['Second'] = str(int(initial_time_dict['Second'])+1)

    # Round off with second
    if int(initial_time_dict['Second']) >= 60:
        excess_second = int(initial_time_dict['Second']) - 60
        initial_time_dict['Second'] = '00'
        initial_time_dict['Minute'] = \
            str(int(initial_time_dict['Minute']) + excess_second + 1)

    # Round off with minute
    if int(initial_time_dict['Minute']) >= 60:
        excess_minute = int(initial_time_dict['Minute']) - 60
        initial_time_dict['Minute'] = '00'
        initial_time_dict['Hour'] = \
            str(int(initial_time_dict['Hour']) + excess_minute + 1)

    # Round off with hour
    if int(initial_time_dict['Hour']) >= 24:
        excess_hour = int(initial_time_dict['Hour']) - 24
        initial_time_dict['Hour'] = '00'
        initial_time_dict['Day'] = \
            str(int(initial_time_dict['Day']) + excess_hour + 1)

    # Final time
    final_time = datetimes[-1]
    final_datetime_obj = netCDF4.num2date(
            final_time, units=datetimes_unit, calendar=datetimes_calendar)

    final_time_dict = {
        "Year": str(final_datetime_obj.year).zfill(4),
        "Month": str(final_datetime_obj.month).zfill(2),
        "Day": str(final_datetime_obj.day).zfill(2),
        "Hour": str(final_datetime_obj.hour).zfill(2),
        "Minute": str(final_datetime_obj.minute).zfill(2),
        "Second": str(final_datetime_obj.second).zfill(2),
        "Microsecond": str(final_datetime_obj.microsecond).zfill(6)
    }

    # Round off with microsecond
    if int(final_time_dict['Microsecond']) > 500000:
        final_time_dict['Microsecond'] = '000000'
        # # Do not increase the second for final time
        # final_time_dict['Second'] = str(int(initial_time_dict['Second'])+1)

    # Round off with second
    if int(final_time_dict['Second']) >= 60:
        excess_second = int(final_time_dict['Second']) - 60
        final_time_dict['Second'] = '00'
        final_time_dict['Minute'] = \
            str(int(final_time_dict['Minute']) + excess_second + 1)

    # Round off with minute
    if int(final_time_dict['Minute']) >= 60:
        excess_minute = int(final_time_dict['Minute']) - 60
        final_time_dict['Minute'] = '00'
        final_time_dict['Hour'] = \
            str(int(final_time_dict['Hour']) + excess_minute + 1)

    # Round off with hour
    if int(final_time_dict['Hour']) >= 24:
        excess_hour = int(final_time_dict['Hour']) - 24
        final_time_dict['Hour'] = '00'
        final_time_dict['Day'] = \
            str(int(final_time_dict['Day']) + excess_hour + 1)

    # Find time unit
    datetimes_unit_string = \
        datetimes_unit[:datetimes_unit.find('since')].replace(' ', '')

    # Find time unit conversion to make times in unit of day
    if 'microsecond' in datetimes_unit_string:
        time_unit_conversion = 1.0 / 1000000.0
    elif 'millisecond' in datetimes_unit_string:
        time_unit_conversion = 1.0 / 1000.0
    elif 'second' in datetimes_unit_string:
        time_unit_conversion = 1.0
    elif 'minute' in datetimes_unit_string:
        time_unit_conversion = 60.0
    elif 'hour' in datetimes_unit_string:
        time_unit_conversion = 3600.0
    elif 'day' in datetimes_unit_string:
        time_unit_conversion = 24.0 * 3600.0

    # Time duration (in seconds)
    time_duration = numpy.fabs(datetimes[-1] - datetimes[0]) * \
        time_unit_conversion

    # Round off with microsecond
    # time_duration = numpy.floor(time_duration + 0.5)
    time_duration = numpy.floor(time_duration)

    # Day
    residue = 0.0
    time_duration_day = int(numpy.floor(time_duration/(24.0*3600.0)))
    residue = time_duration - float(time_duration_day) * (24.0*3600.0)

    # Hour
    time_duration_hour = int(numpy.floor(residue / 3600.0))
    residue -= float(time_duration_hour) * 3600.0

    # Minute
    time_duration_minute = int(numpy.floor(residue / 60.0))
    residue -= float(time_duration_minute) * 60.0

    # Second
    time_duration_second = int(numpy.floor(residue))

    time_duration_dict = {
        "Day": str(time_duration_day),
        "Hour": str(time_duration_hour).zfill(2),
        "Minute": str(time_duration_minute).zfill(2),
        "Second": str(time_duration_second).zfill(2)
    }

    # Datetime Size
    datetime_size = datetime_obj.size

    # Create time info dictionary
    time_info = {
        "InitialTime": initial_time_dict,
        "FinalTime": final_time_dict,
        "TimeDuration": time_duration_dict,
        "TimeDurationInSeconds": str(time_duration),
        "DatetimeSize": str(datetime_size)
    }

    return time_info


# ==============
# Get Space Info
# ==============

def _get_space_info(longitude_obj, latitude_obj):
    """
    Get the dictionary of data bounds and camera bounds.
    """

    # Bounds for dataset
    min_latitude = numpy.min(latitude_obj[:])
    max_latitude = numpy.max(latitude_obj[:])
    min_longitude = numpy.min(longitude_obj[:])
    max_longitude = numpy.max(longitude_obj[:])

    # Cut longitude for overlapping longitudes
    if max_longitude - min_longitude > 360:
        max_longitude = min_longitude + 360

    # Center
    mid_longitude = 0.5 * (min_longitude + max_longitude)
    mid_latitude = 0.5 * (min_latitude + max_latitude)

    # Range (in meters)
    earth_radius = 6378.1370e+3   # meters
    latitude_range = (max_latitude - min_latitude) * (numpy.pi / 180.0) * \
        earth_radius
    longitude_range = (max_longitude - min_longitude) * (numpy.pi / 180.0) * \
        earth_radius * numpy.cos(mid_latitude * numpy.pi / 180.0)

    # Resolutions
    longitude_resolution = longitude_obj.size
    latitude_resolution = latitude_obj.size

    # View Range
    # This is used to show a bit larger area from left to right
    view_scale = 1.4
    view_range = numpy.clip(longitude_range * view_scale, 0.0,
                            2.0 * earth_radius * view_scale)

    # Pitch Angle, measured from horizon downward. 45 degrees for small ranges,
    # approaches 90 degrees for large ranges.
    pitch_angle = 45.0 + 45.0 * numpy.max(
            [numpy.fabs(max_longitude-min_longitude)/360.0,
             numpy.fabs(max_latitude-min_latitude)/180.0])

    # Bounds for Camera
    latitude_ratio = 0.2
    latitude_span = max_latitude - min_latitude

    # On very large latitude spans, the latitude_ratio becomes ineffective.
    camera_min_latitude = numpy.clip(
            min_latitude - latitude_ratio * latitude_span *
            (1.0 - latitude_span/180.0), -90.0, 90.0)
    camera_max_latitude = numpy.clip(
            max_latitude + latitude_ratio * latitude_span *
            (1.0 - latitude_span/180.0), -90.0, 90.0)

    longitude_ratio = 0.2
    longitude_span = max_longitude - min_longitude

    # On very large longitude spans, the longitude_ratio becomes ineffective.
    camera_min_longitude = mid_longitude - numpy.clip(
        longitude_span/2.0 + longitude_ratio * longitude_span *
        (1.0 - longitude_span / 360.0), 0.0, 90.0)
    camera_max_longitude = mid_longitude + numpy.clip(
        longitude_span/2.0 + longitude_ratio * longitude_span *
        (1.0 - longitude_span / 360.0), 0.0, 90.0)

    data_bounds_dict = {
        "MinLatitude": str(min_latitude),
        "MidLatitude": str(mid_latitude),
        "MaxLatitude": str(max_latitude),
        "MinLongitude": str(min_longitude),
        "MidLongitude": str(mid_longitude),
        "MaxLongitude": str(max_longitude)
    }

    data_range_dict = {
        "LongitudeRange": str(longitude_range),
        "LatitudeRange": str(latitude_range),
        "ViewRange": str(view_range),
        "PitchAngle": str(pitch_angle)
    }

    data_resolution_dict = {
        "LongitudeResolution": str(longitude_resolution),
        "LatitudeResolution": str(latitude_resolution)
    }

    camera_bounds_dict = {
        "MinLatitude": str(camera_min_latitude),
        "MaxLatitude": str(camera_max_latitude),
        "MinLongitude": str(camera_min_longitude),
        "MaxLongitude": str(camera_max_longitude),
    }

    # Space Info
    space_info_dict = {
        "DataResolution": data_resolution_dict,
        "DataBounds": data_bounds_dict,
        "DataRange": data_range_dict,
        "CameraBounds": camera_bounds_dict
    }

    return space_info_dict


# =================
# Get Velocity Name
# =================

def _get_velocity_name(east_name, north_name):
    """
    Given two names: east_name and north_name, it finds the intersection of the
    names. Also it removes the redundant slashes, etc.

    Example:
    east_name:  surface_eastward_sea_water_velocity
    north_name: surface_northward_sea_water_velocity

    Output velocity_name: surface_sea_water_velocity
    """

    # Split names based on a delimiter
    delimiter = '_'
    east_name_splitted = east_name.split(delimiter)
    north_name_splitted = north_name.split(delimiter)

    velocity_name_list = []
    num_words = numpy.min([len(east_name_splitted), len(north_name_splitted)])
    for i in range(num_words):
        if east_name_splitted[i] == north_name_splitted[i]:
            velocity_name_list.append(east_name_splitted[i])

    # Convert set to string
    if len(velocity_name_list) > 0:
        velocity_name = '_'.join(str(s) for s in velocity_name_list)
    else:
        velocity_name = ''

    return velocity_name


# =================
# Get Velocity Info
# =================

def _get_velocity_info(
        east_velocity_obj,
        north_velocity_obj,
        east_velocity_name,
        north_velocity_name,
        east_velocity_standard_name,
        north_velocity_standard_name):
    """
    Get dictionary of velocities.
    """

    # Get the number of indices to be selected for finding min and max.
    num_times = east_velocity_obj.shape[0]
    num_time_indices = 10  # Selecting 10 samples
    if num_time_indices > num_times:
        num_time_indices = num_times

    # The selection of random time indices to be used for finding min and max
    numpy.random.seed(0)
    times_indices = numpy.random.randint(0, num_times-1, num_time_indices)

    # Min/Max velocities for each time frame
    east_velocities_mean = numpy.zeros(len(times_indices), dtype=float)
    east_velocities_std = numpy.zeros(len(times_indices), dtype=float)
    north_velocities_mean = numpy.zeros(len(times_indices), dtype=float)
    north_velocities_std = numpy.zeros(len(times_indices), dtype=float)

    # Find Min and Max of each time frame
    for k in range(len(times_indices)):

        time_index = times_indices[k]

        with numpy.errstate(invalid='ignore'):
            east_velocities_mean[k] = \
                numpy.nanmean(east_velocity_obj[time_index, :, :])
            east_velocities_std[k] = \
                numpy.nanstd(east_velocity_obj[time_index, :, :])
            north_velocities_mean[k] = \
                numpy.nanmean(north_velocity_obj[time_index, :, :])
            north_velocities_std[k] = \
                numpy.nanstd(north_velocity_obj[time_index, :, :])

    # Mean and STD of Velocities among all time frames
    east_velocity_mean = numpy.nanmean(east_velocities_mean)
    east_velocity_std = numpy.nanmean(east_velocities_std)
    north_velocity_mean = numpy.nanmean(north_velocities_mean)
    north_velocity_std = numpy.nanmean(north_velocities_std)

    # Min/Max of Velocities, assuming u and v have Gaussian distributions
    scale = 4.0
    min_east_velocity = east_velocity_mean - scale * east_velocity_std
    max_east_velocity = east_velocity_mean + scale * east_velocity_std
    min_north_velocity = north_velocity_mean - scale * north_velocity_std
    max_north_velocity = north_velocity_mean + scale * north_velocity_std

    # An estimate for max velocity speed. If u and v has Gaussian
    # distributions, the velocity speed has Chi distribution
    velocity_mean = numpy.sqrt(east_velocity_mean**2 + east_velocity_std**2 +
                               north_velocity_mean**2 + north_velocity_std**2)
    typical_velocity_speed = 4.0 * velocity_mean

    # Get the velocity name from east and north names
    if (east_velocity_standard_name != '') and \
            (north_velocity_standard_name != ''):
        velocity_standard_name = _get_velocity_name(
                east_velocity_standard_name, north_velocity_standard_name)
    else:
        velocity_standard_name = ''

    # Create a Velocity Info Dict
    velocity_info_dict = {
        "EastVelocityName": east_velocity_name,
        "NorthVelocityName": north_velocity_name,
        "EastVelocityStandardName": east_velocity_standard_name,
        "NorthVelocityStandardName": north_velocity_standard_name,
        "VelocityStandardName": velocity_standard_name,
        "MinEastVelocity": str(min_east_velocity),
        "MaxEastVelocity": str(max_east_velocity),
        "MinNorthVelocity": str(min_north_velocity),
        "MaxNorthVelocity": str(max_north_velocity),
        "TypicalVelocitySpeed": str(typical_velocity_speed)
    }

    return velocity_info_dict


# ====
# scan
# ====

def scan(argv):
    """
    Reads a netcdf file and returns data info.

    Notes:

        - If the option -V is used to scan min and max of velocities,
          we do not find the min and max of velocity for all time frames. This
          is because if the nc file is large, it takes a long time. Also we do
          not load the whole velocities like U[:] or V[:] because it the data
          is large, the netCDF4 package raises an error.
    """

    # Fill output with defaults
    dataset_info_dict = {
        "Scan": {
            "ScanStatus": True,
            "Message": ""
        },
        "TimeInfo": None,
        "SpaceInfo": None,
        "VelocityInfo": None
    }

    # Parse arguments
    input_filename, scan_velocity_status = _parse_arguments(argv)

    # Open file
    agg = _load_dataset(input_filename)

    # Load variables
    datetime_obj, longitude_obj, latitude_obj = \
        _load_time_and_space_variables(agg)

    # Get Time Info
    time_info_dict = _get_time_info(datetime_obj)
    dataset_info_dict['TimeInfo'] = time_info_dict

    # Get Space Info
    space_info_dict = _get_space_info(longitude_obj, latitude_obj)
    dataset_info_dict['SpaceInfo'] = space_info_dict

    # Velocities
    if scan_velocity_status is True:

        # Get velocity objects
        east_velocity_obj, north_velocity_obj, east_velocity_name, \
                north_velocity_name, east_velocity_standard_name, \
                north_velocity_standard_name = _load_velocity_variables(agg)
        velocity_info_dict = _get_velocity_info(
                east_velocity_obj, north_velocity_obj, east_velocity_name,
                north_velocity_name, east_velocity_standard_name,
                north_velocity_standard_name)
        dataset_info_dict['VelocityInfo'] = velocity_info_dict

    agg.close()

    dataset_info_json = json.dumps(dataset_info_dict, indent=4)
    print(dataset_info_json)
    sys.stdout.flush()


# ====
# Main
# ====

def main():
    """
    Main function to be called when this script is called as an executable.
    """

    # Converting all warnings to error
    # warnings.simplefilter('error', UserWarning)
    warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Main function
    scan(sys.argv)


# ===========
# Script Main
# ===========

if __name__ == "__main__":
    main()
