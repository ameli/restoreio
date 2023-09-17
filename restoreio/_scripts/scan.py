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
import datetime
import json

import argparse
from .._parser.formatter import WrappedNewlineFormatter
from .examples import examples
from ..__version__ import __version__

try:
    # For python 3
    from urllib.parse import urlparse
except ImportError:
    # For python 2
    from urlparse import urlparse

__all__ = ['scan']


# ====================
# Terminate With Error
# ====================

def _terminate_with_error(message, terminate=False):
    """
    Returns an incomplete Json object with ScanStatus = False, and an error
    message.

    If terminate is True, the python program is exited with code 1. If False,
    a valueError is raised without terminating the program.
    """

    if terminate:

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
        sys.exit(1)

    else:
        raise ValueError(message)


# =============
# Create Parser
# ============

def create_parser():
    """
    Creates parser object.

    The parser object can be used in both parsing arguments and to generate
    Sphinx documentation for the command line interface (CLI).
    """

    # Instantiate the parser
    description = 'Scan dataset and return data information.'
    epilog = examples
    # formatter_class = argparse.RawTextHelpFormatter
    # formatter_class = argparse.ArgumentDefaultsHelpFormatter
    # formatter_class = DescriptionWrappedNewlineFormatter
    formatter_class = WrappedNewlineFormatter

    parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                     formatter_class=formatter_class,
                                     add_help=False)

    # Manually create two groups of required and optional arguments
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Add back help
    optional.add_argument('-h', '--help', action='help',
                          default=argparse.SUPPRESS,
                          help='show this help message and exit')

    # Input filename
    help_input = """
    Input filename. This can be either the path to a local file or the URL to
    a remote dataset. The file or URL may or may not have a file extension.
    However, if the file does have an extension, the file extension should be
    either ``.nc``, ``.nc4``, ``.ncd``, ``.nc.gz``, ``.ncml``, or ``*.ncml.gz``
    only.
    """
    required.add_argument('-i', type=str, help=help_input, metavar='INPUT',
                          required=True)

    # Scan velocity
    help_scan_velocity = """
    Scans the velocity arrays in the file. This is useful to find the min and
    max range of the velocity data to adjust the color bar for plotting.
    """
    optional.add_argument('-V', action='store_true', help=help_scan_velocity)

    # Scan other variables
    help_scan_other_var = """
    Scans other arrays in the file. This is useful to find the min and max
    range of the other arrays data to adjust the color bar for plotting. This
    argument should be comma-separated name of variables.
    """
    optional.add_argument('-O', type=str, help=help_scan_other_var,
                          default='', metavar='OTHER')

    # Terminate
    help_terminate = """
    If `True`, the program exists with code `1`. This is useful when this
    package is executed on a server to pass exit signals to a Node application.
    On the downside, this option causes an interactive python environment to
    both terminate the script and the python environment itself. To avoid this,
    set this option to `False`. In this case, upon an error, the ``ValueError``
    is raised, which cases the script to terminate, however, an interactive
    python environment will not be exited.
    """
    optional.add_argument('-T', action='store_true', help=help_terminate)

    # Version
    help_version = """
    Prints version.
    """
    version = '%(prog)s {version}'.format(version=__version__)
    parser.add_argument('-v', '--version', action='version', version=version,
                        help=help_version)

    return parser


# ===============
# Parse Arguments
# ===============

def _parse_arguments():
    """
    Parses the argument of the executable and obtains the filename.
    """

    # Create parser object
    parser = create_parser()

    # Parse arguments. Here args is a namespace
    args = parser.parse_args()

    # Convert namespace to dictionary
    # args = vars(args)

    # Output dictionary
    arguments = {
        'input': args.i,
        'scan_velocity': args.V,
        'scan_other_var': args.O,
        'terminate': args.T,
    }

    return arguments


# ==================
# Load Local Dataset
# ==================

def _load_local_dataset(filename, terminate):
    """
    Opens either ncml or nc file and returns the aggregation file object.
    """

    # Check file extension
    file_extension = os.path.splitext(filename)[1]
    if file_extension in ['.ncml', '.ncml.gz']:

        # Change directory
        data_directory = os.path.dirname(filename)
        current_directory = os.getcwd()
        os.chdir(data_directory)

        # NCML
        try:
            ncml_string = open(filename, 'r').read()
            ncml_string = ncml_string.encode('ascii')
            ncml = pyncml.etree.fromstring(ncml_string)
            nc = pyncml.scan(ncml=ncml)

            # Get nc files list
            files_list = [f.path for f in nc.members]
            os.chdir(current_directory)

            # Aggregate
            agg = netCDF4.MFDataset(files_list, aggdim='t')

        except BaseException as error:
            print('ERROR: Can not read local multifile ncml dataset: '
                  '<tt>' + filename + "</tt>.")
            raise error

        return agg

    elif file_extension in ['.nc', '.nc4', '.ncd', '.nc.gz']:

        try:
            nc = netCDF4.Dataset(filename)
        except BaseException as error:
            print('ERROR: Can not read local dataset: '
                  '<tt>' + filename + "</tt>.")
            sys.stdout.flush()
            raise error

        return nc

    else:
        _terminate_with_error(
            'File extension in the data should be either '
            '<code>.nc</code>, <code>.nc4</code>, <code>.ncd</code>, '
            '<code>.nc.gz</code>, <code>.ncml</code>, or '
            '<code>.ncml.gz</code>.')


# ===================
# Load Remote Dataset
# ===================

def _load_remote_dataset(url, terminate):
    """
    URL can be point to a *.nc or *.ncml file.
    """

    # Check URL is opendap
    if (url.startswith('http://') is False) and \
       (url.startswith('https://') is False):
        _terminate_with_error('Input data URL does not seem to be a URL. ' +
                              'A URL should start with <code>http://</code> ' +
                              'or <code>https://</code>.',
                              terminate)

    elif ("/thredds/dodsC/" not in url) and ("opendap" not in url):
        _terminate_with_error('Input data URL is not an <b>OpenDap</b> URL ' +
                              'or is not hosted on a THREDDs server. Check ' +
                              'if your data URL contains ' +
                              '<code>/thredds/dodsC/</code> or ' +
                              '<code>/opendap/</code>.',
                              terminate)

    # Check file extension
    file_extension = os.path.splitext(url)[1]

    # Case of zipped files (get the correct file extension before the '.gz')
    if file_extension == ".gz":
        file_extension = os.path.splitext(url[:-3])[1]

    # Note that some opendap urls do not even have a file extension
    if file_extension != "":

        # If a file extension exists, check if it is a standard netcdf file
        if file_extension not in \
                ['.nc', '.nc4', '.ncd', '.nc.gz', '.ncml', '.ncml.gz']:
            _terminate_with_error(
                'The input data URL is not an <i>netcdf</i> file. The URL ' +
                'should end with <code>.nc</code>, <code>.nc4</code>, ' +
                '<code>.ncd</code>, <code>.nc.gz</code>, ' +
                '<code>.ncml</code>, <code>.ncml.gz</code>, or without file ' +
                'extension.', terminate)

    try:
        nc = netCDF4.Dataset(url)

    except OSError:
        _terminate_with_error('Unable to read %s.' % url, terminate)

    return nc


# ============
# Load Dataset
# ============

def _load_dataset(input, terminate):
    """
    Dispatches the execution to either of the following two functions:
    1. LoadMultiFileDataset: For files where the input is a path on
       the local machine.
    2. _load_remote_dataset: For files remotely where input is a URL.
    """

    # Check input filename
    if input == '':
        _terminate_with_error('Input data URL is empty. You should provide ' +
                              'an OpenDap URL.',
                              terminate)

    # Check if the input has a "host" name
    if bool(urlparse(input).netloc):
        # input is a URL
        return _load_remote_dataset(input, terminate)
    else:
        # input is a path
        return _load_local_dataset(input, terminate)


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

def _load_time_and_space_variables(agg, terminate):
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
                              'the netcdf file.', terminate)
    elif hasattr(datetime_obj, 'units') is False:
        _terminate_with_error('The <t>time</i> variable does not have ' +
                              '<i>units</i> attribute.', terminate)
    # elif hasattr(datetime_obj, 'calendar') is False:
    #     _terminate_with_error('The <t>time</i> variable does not have ' +
    #                           '<i>calendar</i> attribute.', terminate)
    # elif datetime_obj.size < 2:
    #     _terminate_with_error('The <i>time</i> variable size should be at ' +
    #                           'least <tt>2</tt>.', terminate)

    # Longitude
    longitude_names_list = ['longitude', 'lon', 'long']
    longitude_standard_names_list = ['longitude']
    longitude_obj, longitude_name, longitude_standard_name = _search_variable(
        agg, longitude_names_list, longitude_standard_names_list)

    # Check longitude variable
    if longitude_obj is None:
        _terminate_with_error('Can not find the <i>longitude</i> variable ' +
                              'in the netcdf file.', terminate)
    elif len(longitude_obj.shape) != 1:
        _terminate_with_error('The <t>longitude</i> variable dimension ' +
                              'should be <tt>1<//t>.', terminate)
    elif longitude_obj.size < 2:
        _terminate_with_error('The <i>longitude</i> variable size should ' +
                              'be at least <tt>2</tt>.', terminate)

    # Latitude
    latitude_names_list = ['latitude', 'lat']
    latitude_standard_names_list = ['latitude']
    latitude_obj, latitude_name, latitude_standard_name = _search_variable(
        agg, latitude_names_list, latitude_standard_names_list)

    # Check latitude variable
    if latitude_obj is None:
        _terminate_with_error('Can not find the <i>latitude</i> variable in ' +
                              'the netcdf file.', terminate)
    elif len(latitude_obj.shape) != 1:
        _terminate_with_error('The <t>latitude</i> variable dimension ' +
                              'should be <tt>1<//t>.', terminate)
    elif latitude_obj.size < 2:
        _terminate_with_error('The <i>latitude</i> variable size should ' +
                              'be at least <tt>2</tt>.', terminate)

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
    east_velocity_names_list = ['east_vel', 'eastward_vel', 'u', 'ugos',
                                'east_velocity', 'eastward_velocity']
    east_velocity_standard_names_list = [
        'surface_eastward_sea_water_velocity',
        'eastward_sea_water_velocity',
        'surface_geostrophic_eastward_sea_water_velocity',
        'surface_geostrophic_sea_water_x_velocity',
        'surface_geostrophic_eastward_sea_water_velocity_assuming_sea_' +
        'level_for_geoid',
        'surface_eastward_geostrophic_sea_water_velocity_assuming_sea_' +
        'level_for_geoid',
        'surface_geostrophic_sea_water_x_velocity_assuming_mean_sea_level_' +
        'for_geoid',
        'surface_geostrophic_sea_water_x_velocity_assuming_sea_level_for_' +
        'geoid',
        'surface_geostrophic_eastward_sea_water_velocity_assuming_mean_sea_' +
        'level_for_geoid',
        'sea_water_x_velocity',
        'x_sea_water_velocity']
    east_velocity_obj, east_velocity_name, east_velocity_standard_name = \
        _search_variable(agg, east_velocity_names_list,
                         east_velocity_standard_names_list)

    # North Velocity
    north_velocity_names_list = ['north_vel', 'northward_vel', 'v', 'vgos',
                                 'north_velocity', 'northward_velocity']
    north_velocity_standard_names_list = [
        'surface_northward_sea_water_velocity',
        'northward_sea_water_velocity',
        'surface_geostrophic_northward_sea_water_velocity',
        'surface_geostrophic_sea_water_y_velocity',
        'surface_geostrophic_northward_sea_water_velocity_assuming_sea_' +
        'level_for_geoid',
        'surface_northward_geostrophic_sea_water_velocity_assuming_sea_' +
        'level_for_geoid',
        'surface_geostrophic_sea_water_y_velocity_assuming_mean_sea_level_' +
        'for_geoid',
        'surface_geostrophic_sea_water_y_velocity_assuming_sea_level_for_' +
        'geoid',
        'surface_geostrophic_northward_sea_water_velocity_assuming_mean_' +
        'sea_level_for_geoid',
        'sea_water_y_velocity',
        'y_sea_water_velocity']
    north_velocity_obj, north_velocity_name, north_velocity_standard_name = \
        _search_variable(agg, north_velocity_names_list,
                         north_velocity_standard_names_list)

    return east_velocity_obj, north_velocity_obj, east_velocity_name, \
        north_velocity_name, east_velocity_standard_name, \
        north_velocity_standard_name


# =============
# Load Variable
# =============

def _load_variable(agg, var_name):
    """
    Finds a variable with given name from the aggregation object agg.
    """

    # East Velocity
    var_names_list = [var_name]
    var_standard_names_list = []
    var_obj = _search_variable(agg, var_names_list, var_standard_names_list)[0]

    return var_obj


# =================
# Prepare datetimes
# =================

def _prepare_datetimes(datetime_obj, terminate):
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
        _terminate_with_error("Datetime ndim is more than 2.", terminate)

    return datetimes, datetimes_unit, datetimes_calendar


# =============
# Get Time Info
# =============

def _get_time_info(datetime_obj, terminate):
    """
    Get the initial time info and time duration.
    """

    # Datetime Size
    datetime_size = datetime_obj.size

    datetimes, datetimes_unit, datetimes_calendar = \
        _prepare_datetimes(datetime_obj, terminate)

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
    if (int(initial_time_dict['Microsecond']) > 500000):
        initial_time_dict['Microsecond'] = '000000'
        initial_time_dict['Second'] = str(int(initial_time_dict['Second']) + 1)

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
    if datetime_size == 1:
        final_time_dict = initial_time_dict
    else:
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
            # final_time_dict['Second'] = str(int(final_time_dict['Second'])+1)

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
    if datetime_size == 1:
        time_duration = 0.0
    else:
        time_duration = numpy.fabs(datetimes[-1] - datetimes[0]) * \
            time_unit_conversion

        # Round off with microsecond
        # time_duration = numpy.floor(time_duration + 0.5)
        time_duration = numpy.floor(time_duration)

    # Day
    residue = 0.0
    time_duration_day = int(numpy.floor(time_duration / (24.0 * 3600.0)))
    residue = time_duration - float(time_duration_day) * (24.0 * 3600.0)

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
        [numpy.fabs(max_longitude - min_longitude) / 360.0,
         numpy.fabs(max_latitude - min_latitude) / 180.0])

    # Bounds for Camera
    latitude_ratio = 0.2
    latitude_span = max_latitude - min_latitude

    # On very large latitude spans, the latitude_ratio becomes ineffective.
    camera_min_latitude = numpy.clip(
        min_latitude - latitude_ratio * latitude_span *
        (1.0 - latitude_span / 180.0), -90.0, 90.0)
    camera_max_latitude = numpy.clip(
        max_latitude + latitude_ratio * latitude_span *
        (1.0 - latitude_span / 180.0), -90.0, 90.0)

    longitude_ratio = 0.2
    longitude_span = max_longitude - min_longitude

    # On very large longitude spans, the longitude_ratio becomes ineffective.
    camera_min_longitude = mid_longitude - numpy.clip(
        longitude_span / 2.0 + longitude_ratio * longitude_span *
        (1.0 - longitude_span / 360.0), 0.0, 90.0)
    camera_max_longitude = mid_longitude + numpy.clip(
        longitude_span / 2.0 + longitude_ratio * longitude_span *
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


# =====================
# Get Array Memory Size
# =====================

def _get_array_memory_size(array, terminate):
    """
    If array ndim is three, such as (time, lat, lon), this function returns
    the size of array(0, :, :).

    If array ndim is four, such as (time, depth, lat, lon), this function
    returns the size of array(0, 0, :, :).
    """

    # Depending on ndim, exclude time and depth dimensions as they wont be read
    if array.ndim == 3:
        shape = array.shape[1:]
        itemsize = array[0, 0, :0].itemsize
    elif array.ndim == 4:
        shape = array.shape[2:]
        itemsize = array[0, 0, 0, :0].itemsize
    else:
        _terminate_with_error('Array ndim should be three or four.',
                              terminate)

    # Size of array (excluding time and depth dimensions)
    size = numpy.prod(shape)

    # Size of array in bytes
    num_bytes = size * itemsize

    return num_bytes


# ==============
# get fill value
# ==============

def _get_fill_value(var_obj):
    """
    Finds missing value (or fill value) from wither of east of north velocity
    objects.
    """

    # Missing Value
    if hasattr(var_obj, '_FillValue') and \
            (not numpy.isnan(float(var_obj._FillValue))):
        fill_value = numpy.fabs(float(var_obj._FillValue))

    elif hasattr(var_obj, 'missing_value') and \
            (not numpy.isnan(float(var_obj.missing_value))):
        fill_value = numpy.fabs(float(var_obj.missing_value))

    elif hasattr(var_obj, 'fill_value') and \
            (not numpy.isnan(float(var_obj.fill_value))):
        fill_value = numpy.fabs(float(var_obj.fill_value))

    else:
        fill_value = 999.0

    return fill_value


# =================
# Make Array masked
# =================

def _make_array_masked(array, fill_value):
    """
    Often the array is not masked, but has nan or inf values. This function
    creates a masked array and mask nan and inf.

    Input:
        - array: is a 2D numpy array.
    Output:
        - array: is a 2D numpy.ma array.

    Note: array should be numpy object not netCDF object. So if you have a
          netCDF object, pass its numpy array with array[:] into this function.
    """

    if (not hasattr(array, 'mask')) or (numpy.isscalar(array.mask)):

        # Mask based on fill value
        mask = (array >= fill_value - 1.0)

        # Mask based on nan values
        mask_nan = numpy.isnan(array)
        if mask_nan.any():
            mask = numpy.logical_or(mask, mask_nan)

        # Mask based on inf values
        mask_inf = numpy.isinf(array)
        if mask_inf.any():
            mask = numpy.logical_or(mask, mask_inf)

        array = numpy.ma.masked_array(array, mask=mask)

    return array


# ===========
# Find Stride
# ===========

def find_stride(size):
    """
    Chooses a stride to avoid downloading a large dataset
    """

    stride = int(numpy.floor(size / 400.0)) + 1
    return stride


# =================
# Get Velocity Info
# =================

def _get_velocity_info(
        east_velocity_obj,
        north_velocity_obj,
        east_velocity_name,
        north_velocity_name,
        east_velocity_standard_name,
        north_velocity_standard_name,
        terminate):
    """
    Get dictionary of velocities.
    """

    # Fill value
    east_fill_value = _get_fill_value(east_velocity_obj)
    north_fill_value = _get_fill_value(north_velocity_obj)
    fill_value = numpy.min([east_fill_value, north_fill_value])

    # Get the number of indices to be selected for finding min and max.
    num_times = east_velocity_obj.shape[0]

    # Get the size of one of the velocity arrays
    num_bytes = _get_array_memory_size(east_velocity_obj, terminate)
    num_Mbytes = num_bytes / (1024**2)

    # Number of time instances to sample from velocity data
    if num_Mbytes >= 10.0:
        # If the array is larger than 10 MB, sample only one time of array
        num_time_indices = 1
    elif num_Mbytes >= 1.0:
        num_time_indices = 2
    else:
        num_time_indices = 5

    # Cap the number of time samples by the number of times
    if num_time_indices > num_times:
        num_time_indices = num_times

    # The selection of random time indices to be used for finding min and max
    if num_times > 1:
        numpy.random.seed(0)
        times_indices = numpy.random.randint(0, num_times - 1,
                                             num_time_indices)
    elif num_times == 1:
        times_indices = [0]
    else:
        _terminate_with_error('Velocity array time dimension has zero size.',
                              terminate)

    # Min/Max velocities for each time frame
    east_velocities_mean = numpy.zeros(len(times_indices), dtype=float)
    east_velocities_std = numpy.zeros(len(times_indices), dtype=float)
    north_velocities_mean = numpy.zeros(len(times_indices), dtype=float)
    north_velocities_std = numpy.zeros(len(times_indices), dtype=float)

    # Size of array
    if east_velocity_obj.ndim == 3:
        size_x = east_velocity_obj.shape[2]
        size_y = east_velocity_obj.shape[1]
    elif east_velocity_obj.ndim == 4:
        size_x = east_velocity_obj.shape[3]
        size_y = east_velocity_obj.shape[2]
    else:
        _terminate_with_error('Velocity ndim should be three or four.',
                              terminate)

    # Strides along x and y
    stride_x = find_stride(size_x)
    stride_y = find_stride(size_y)

    # Find Min and Max of each time frame
    for k in range(len(times_indices)):

        time_index = times_indices[k]

        with numpy.errstate(invalid='ignore'):

            # Find vel dimension is (time, lat, lon) or (time, depth, lat, lon)
            if east_velocity_obj.ndim == 3:

                # Velocity dimension is (time, lat, lon)
                east_velocity = east_velocity_obj[
                    time_index, ::stride_x, ::stride_y]
                north_velocity = north_velocity_obj[
                    time_index, ::stride_x, ::stride_y]

            elif east_velocity_obj.ndim == 4:

                # Velocity dimension is (time, depth, lat, lon)
                depth_index = 0
                east_velocity = east_velocity_obj[
                    time_index, depth_index, ::stride_x, ::stride_y]
                north_velocity = north_velocity_obj[
                    time_index, depth_index, ::stride_x, ::stride_y]

            else:
                _terminate_with_error('Velocity ndim should be three or four.',
                                      terminate)

            # Some dataset do not come with mask. Add mask here.
            east_velocity = _make_array_masked(east_velocity, fill_value)
            north_velocity = _make_array_masked(north_velocity, fill_value)

            # Get mean and std of velocities
            east_velocities_mean[k] = numpy.nanmean(east_velocity)
            east_velocities_std[k] = numpy.nanstd(east_velocity)
            north_velocities_mean[k] = numpy.nanmean(north_velocity)
            north_velocities_std[k] = numpy.nanstd(north_velocity)

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


# ========================
# Cornish-Fisher Expansion
# ========================

def _cornish_fisher_expansion(z, k3, k4, k5):
    """
    Compute the Cornish-Fisher expansion. This is used to approximate the
    quantile when there are higher order moments. The quantile of the variable
    is:

        y + std * w,

    where w is the Cornish-Fisher expansion. When only mean mu and the
    standard deviation exists, then, the above is simplified to:

        w = mu + std * z_score.

    When higher-order moments, like skewness and excess kurtosis also exists,
    then, this function computes w accordingly.

    Parameters
    ----------

    z : float
        The Z-score, obtained from the inverse cumulation distribution function
        for a given p-value. For instance:

        * If p=0.01 (99% confidence), z score is 2.56
        * If p=0.05 (95% confidence), z score is 1.96
        * If p=0.01 (90% confidence), z score is 1.65

    k3 : float
        Third cumulant. For a normalized random variable with zero mean and
        unit standard deviation, this is equal to the skewness.

    k4 : float
        Fourth cumulant. For a normalized random variable with zero mean and
        unit standard deviation, this is equal to the excess-kurtosis.

    k5 : float
        Fifth cumulant. For a normalized random variable with zero mean and
        unit standard deviation, this is equal to mu5 - 10*mu3*mu2 where
        mu are the central moments.

    Reference
    ---------

    * https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion
    * https://encyclopediaofmath.org/wiki/Cornish-Fisher_expansion
    * https://www.value-at-risk.net/the-cornish-fisher-expansion/
    """

    # Polynomials
    h1 = (z**2-1)/6.0
    h2 = (z**3-3*z)/24.0
    h3 = -(2*z**3 - 5*z)/36.0
    h4 = (z**4 - 6*z**2 + 3)/120.0
    h5 = -(z**4 - 5*z**2 + 2)/24.0
    h6 = (12*z**4 - 53*z**2 + 17)/324.0

    # CF expansion
    w = z + k3*h1 + k4*h2 + (k3**2)*h3 + k5*h4 + (k3*k4)*h5 + (k3**3)*h6

    return w


# =================
# Get Variable Info
# =================

def _get_variable_info(
        var_obj,
        var_name,
        terminate):
    """
    Get dictionary of a given variable.
    """

    # Fill value
    fill_value = _get_fill_value(var_obj)

    # Get the number of indices to be selected for finding min and max.
    num_times = var_obj.shape[0]

    # Get the size of one of the velocity arrays
    num_bytes = _get_array_memory_size(var_obj, terminate)
    num_Mbytes = num_bytes / (1024**2)

    # Number of time instances to sample from velocity data
    if num_Mbytes >= 10.0:
        # If the array is larger than 10 MB, sample only one time of array
        num_time_indices = 1
    elif num_Mbytes >= 1.0:
        num_time_indices = 2
    else:
        num_time_indices = 5

    # Cap the number of time samples by the number of times
    if num_time_indices > num_times:
        num_time_indices = num_times

    # The selection of random time indices to be used for finding min and max
    if num_times > 1:
        times_indices = [0, -1]  # Reading the first and last time indices
    elif num_times == 1:
        times_indices = [0]
    else:
        _terminate_with_error('Variable array time dimension has zero size.',
                              terminate)

    # Min/Max velocities for each time frame
    w_min = numpy.zeros(len(times_indices), dtype=float)
    w_max = numpy.zeros(len(times_indices), dtype=float)

    # Size of array
    if var_obj.ndim == 3:
        size_x = var_obj.shape[2]
        size_y = var_obj.shape[1]
    elif var_obj.ndim == 4:
        size_x = var_obj.shape[3]
        size_y = var_obj.shape[2]
    else:
        _terminate_with_error('Variable ndim should be three or four.',
                              terminate)

    # Strides along x and y
    stride_x = find_stride(size_x)
    stride_y = find_stride(size_y)

    # Find Min and Max of each time frame
    for k in range(len(times_indices)):

        time_index = times_indices[k]

        with numpy.errstate(invalid='ignore'):

            # Find vel dimension is (time, lat, lon) or (time, depth, lat, lon)
            if var_obj.ndim == 3:

                # Variable dimension is (time, lat, lon)
                var = var_obj[time_index, ::stride_x, ::stride_y]

            elif var_obj.ndim == 4:

                # Variable dimension is (time, depth, lat, lon)
                depth_index = 0
                var = var_obj[time_index, depth_index, ::stride_x, ::stride_y]

            else:
                _terminate_with_error('Array ndim should be three or four.',
                                      terminate)

            # Some dataset do not come with mask. Add mask here.
            var = _make_array_masked(var, fill_value)

            # Get moments
            mean = numpy.nanmean(var)
            std = numpy.nanstd(var)

            if std < 1e-6:
                # Example of such data is forward FTLE at the first time step
                # where all FTLE values are zero, or backward FTLE at the
                # last time step.
                w_min[k] = numpy.nan
                w_max[k] = numpy.nan

            else:
                # Normalize so that the variable is in N(0, 1)
                var_normalized = (var.ravel() - mean) / std

                # Normalized central moments
                mu2 = 1.0
                mu3 = numpy.nanmean(var_normalized**3)
                mu4 = numpy.nanmean(var_normalized**4)
                mu5 = numpy.nanmean(var_normalized**5)

                # Cumulants
                k3 = mu3
                k4 = mu4 - 3.0
                k5 = mu5 - 10.0*mu3*mu2

                # Compute Cornish-Fisher coefficient
                z_score = 2.56  # corresponding to 99% confidence
                w_min[k] = _cornish_fisher_expansion(-z_score, k3, k4, k5)
                w_max[k] = _cornish_fisher_expansion(z_score, k3, k4, k5)

    # Choose mean of the bounds
    w_min = numpy.nanmean(w_min)
    w_max = numpy.nanmean(w_max)

    # For FTLE, skew the min and max value for plotting purposes
    if var_name == "ftle":
        scale_min = 1.1
        scale_max = 3.1
        w_min *= scale_min
        w_max *= scale_max

    # Quantiles
    var_min = mean + w_min * std
    var_max = mean + w_max * std

    # Create a Variable Info Dict
    var_info_dict = {
        "VarName": var_name,
        "MinValue": str(var_min),
        "MaxValue": str(var_max),
    }

    return var_info_dict


# ====
# scan
# ====

def scan(
        input,
        scan_velocity=False,
        scan_other_var='',
        terminate=False,
        verbose=False):
    """
    Reads a netcdf file and returns data info.

    Parameters
    ----------

    input : str
        The input netcdf file URL (if remote file) or file name (if local
        data). The URL should either have no extension, if it does have an
        extension, the file extension should be either of ``.nc``, ``.nc4``,
        ``.ncd``, ``.nc.gz``, ``.ncml``, or ``.ncml.gz``.

    scan_velocity : bool, default=False
        Scans the velocity arrays in the file. This is useful to find the min
        and max range of the velocity data to adjust the color bar for
        plotting.

    scan_other_var : str, default=''
        Scans other arrays in the file. This is useful to find the min and max
        range of the other arrays data to adjust the color bar for plotting.
        This argument should be comma-separated name of variables.

    terminate : bool, default=False
        If `True`, the program exists with code 1. This is useful when this
        package is executed on a server to pass exit signals to a Node
        application. On the downside, this option causes an interactive python
        environment to both terminate the script and the python environment
        itself. To avoid this, set this option to `False`. In this case, upon
        an error, the ``ValueError`` is raised, which cases the script to
        terminate, however, an interactive python environment will not be
        exited.

    verbose : bool, default=False
        Prints the output dictionary. Note when ``scan`` is used in the
        command-line, the result is always verbose, whereas if used in the
        python environment, the verbosity is determined by this argument.

    Returns
    -------

    info : dict
        A dictionary containing information about the netcdf file dataset.

    See Also
    --------

    restoreio.restore

    Notes
    -----

    * If the ``scan_velocity`` option (or ``-V`` in command line) is used to
      scan min and max of velocities, we do not find the min and max of
      velocity for all time frames. This is because if the `nc` file is large,
      it takes a long time. Also we do not load the whole velocities like
      ``U[:]`` or ``V[:]`` because it the data is large, the netCDF4 package
      raises an error.

    Examples
    --------

    This code shows acquiring information for an HF radar dataset:

    .. code-block:: python
        :emphasize-lines: 6, 7

        >>> from restoreio import scan
        >>> input = 'https://transport.me.berkeley.edu/thredds/dodsC/' + \\
        ...         'root/WHOI-HFR/WHOI_HFR_2014_original.nc'

        >>> # Run script
        >>> info = scan(input, scan_velocity=True, terminate=False,
        ...             verbose=True)

    The above info variable is a Python dictionary. You can print it using
    ``print(info)``, but this may not produce a readable output. To achieve
    better readability, you can convert the dictionary into a JSON object and
    then print it with proper indentation. Here's how:

    .. code-block:: python

        >>> import json
        >>> json_obj = json.dumps(info, indent=4)
        >>> print(json_obj)
        {
            "Scan": {
                "ScanStatus": true,
                "Message": ""
            },
            "TimeInfo": {
                "InitialTime": {
                    "Year": "2014",
                    "Month": "07",
                    "Day": "01",
                    "Hour": "00",
                    "Minute": "00",
                    "Second": "00",
                    "Microsecond": "000000"
                },
                "FinalTime": {
                    "Year": "2014",
                    "Month": "09",
                    "Day": "30",
                    "Hour": "23",
                    "Minute": "29",
                    "Second": "59",
                    "Microsecond": "000000"
                },
                "TimeDuration": {
                    "Day": "91",
                    "Hour": "23",
                    "Minute": "29",
                    "Second": "59"
                },
                "TimeDurationInSeconds": "7946999.0",
                "DatetimeSize": "4416"
            },
            "SpaceInfo": {
                "DataResolution": {
                    "LongitudeResolution": "39",
                    "LatitudeResolution": "36"
                },
                "DataBounds": {
                    "MinLatitude": "41.08644",
                    "MidLatitude": "41.212500000000006",
                    "MaxLatitude": "41.33856",
                    "MinLongitude": "-70.797912",
                    "MidLongitude": "-70.616667",
                    "MaxLongitude": "-70.435422"
                },
                "DataRange": {
                    "LongitudeRange": "30355.79907013849",
                    "LatitudeRange": "28065.8700187999",
                    "ViewRange": "42498.11869819389",
                    "PitchAngle": "45.06303"
                },
                "CameraBounds": {
                    "MinLatitude": "41.036086627216",
                    "MaxLatitude": "41.388913372784",
                    "MinLongitude": "-70.87033700055551",
                    "MaxLongitude": "-70.3629969994445"
                }
            },
            "VelocityInfo": {
                "EastVelocityName": "east_vel",
                "NorthVelocityName": "north_vel",
                "EastVelocityStandardName": "eastward_wind",
                "NorthVelocityStandardName": "northward_wind",
                "VelocityStandardName": "wind",
                "MinEastVelocity": "-0.48760945773342956",
                "MaxEastVelocity": "0.3507359965788057",
                "MinNorthVelocity": "-0.36285106142279266",
                "MaxNorthVelocity": "0.312877327708193",
                "TypicalVelocitySpeed": "0.6121967517719955"
            }
        }
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

    # Open file
    agg = _load_dataset(input, terminate)

    # Load variables
    datetime_obj, longitude_obj, latitude_obj = \
        _load_time_and_space_variables(agg, terminate)

    # Get Time Info
    time_info_dict = _get_time_info(datetime_obj, terminate)
    dataset_info_dict['TimeInfo'] = time_info_dict

    # Get Space Info
    space_info_dict = _get_space_info(longitude_obj, latitude_obj)
    dataset_info_dict['SpaceInfo'] = space_info_dict

    # Velocities
    if scan_velocity is True:

        # Get velocity objects
        east_velocity_obj, north_velocity_obj, east_velocity_name, \
            north_velocity_name, east_velocity_standard_name, \
            north_velocity_standard_name = _load_velocity_variables(agg)

        # Read velocity data and find info
        velocity_info_dict = _get_velocity_info(
            east_velocity_obj, north_velocity_obj, east_velocity_name,
            north_velocity_name, east_velocity_standard_name,
            north_velocity_standard_name, terminate)

        # Store in dictionary
        dataset_info_dict['VelocityInfo'] = velocity_info_dict

    if scan_other_var != "":

        # Extract name of variables from comma-separated string
        var_names_list = scan_other_var.replace(' ', '').split(',')

        for var_name in var_names_list:

            # Get variable object
            var_obj = _load_variable(agg, var_name)

            # Get variable info
            var_info_dict = _get_variable_info(var_obj, var_name, terminate)

            # Store to dictionary
            dataset_info_dict[var_name] = var_info_dict

    agg.close()

    if verbose:
        dataset_info_json = json.dumps(dataset_info_dict, indent=4)
        print(dataset_info_json)
        sys.stdout.flush()

    return dataset_info_dict


# ====
# Main
# ====

def main():
    """
    Main function to be called when this script is called as an executable.
    """

    # Ignoring some warnings
    warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # Parse arguments
    arguments = _parse_arguments()

    # When this script is used as an entry-point, always make it verbose
    arguments['verbose'] = True

    # Main function
    scan(**arguments)


# ===========
# Script Main
# ===========

if __name__ == "__main__":
    main()
