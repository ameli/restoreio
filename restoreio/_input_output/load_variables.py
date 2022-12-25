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

__all__ = ['load_variables']


# ===============
# Search Variable
# ===============

def search_variable(agg, names_list, standard_namesList):
    """
    This function searches for a list of names and standard names to match a
    variable.

    Note: All strings are compared with their lowercase form.
    """

    variable_found = False

    # Search among standard names list
    for standard_name in standard_namesList:
        for key in agg.variables.keys():
            Variable = agg.variables[key]
            if hasattr(Variable, 'standard_name'):
                standard_nameInAgg = Variable.standard_name
                if standard_name.lower() == standard_nameInAgg.lower():
                    obj = agg.variables[key]
                    variable_found = True
                    break
        if variable_found is True:
            break

    # Search among names list
    if variable_found is False:
        for name in names_list:
            for key in agg.variables.keys():
                if name.lower() == key.lower():
                    obj = agg.variables[key]
                    variable_found = True
                    break
            if variable_found is True:
                break

    # Lat check to see if the variable is found
    if variable_found is False:
        obj = None

    return obj


# ==============
# Load Variables
# ==============

def load_variables(agg):
    """
    Finds the following variables from the aggregation object agg.

    - Time
    - Longitude
    - Latitude
    - Eastward velocity U
    - Northward velocity V
    """

    # Time
    time_names_list = ['time', 'datetime', 't']
    time_standard_names_list = ['time']
    datetime_obj = search_variable(agg, time_names_list,
                                   time_standard_names_list)
    if datetime_obj is None:
        raise RuntimeError('Time object can not be found in netCDF file.')

    # Longitude
    lon_names_list = ['longitude', 'lon', 'long']
    lon_standard_names_list = ['longitude']
    lon_obj = search_variable(agg, lon_names_list, lon_standard_names_list)
    if lon_obj is None:
        raise RuntimeError('Longitude object can not be found in netCDF file.')

    # Latitude
    lat_names_list = ['latitude', 'lat']
    lat_standard_names_list = ['latitude']
    lat_obj = search_variable(agg, lat_names_list, lat_standard_names_list)
    if lat_obj is None:
        raise RuntimeError('Latitude object can not be found in netCDF file.')

    # East Velocity
    east_vel_names_list = ['east_vel', 'eastward_vel', 'u', 'east_velocity',
                           'eastward_velocity']
    east_vel_standard_names_list = ['surface_eastward_sea_water_velocity',
                                    'eastward_sea_water_velocity']
    east_vel_obj = search_variable(agg, east_vel_names_list,
                                   east_vel_standard_names_list)
    if east_vel_obj is None:
        raise RuntimeError('EastVelocity object can not be found in ' +
                           'netCDF file.')

    # North Velocity
    north_vel_names_list = ['north_vel', 'northward_vel', 'v',
                            'north_velocity', 'northward_velocity']
    north_vel_standard_names_list = ['surface_northward_sea_water_velocity',
                                     'northward_sea_water_velocity']
    north_vel_obj = search_variable(agg, north_vel_names_list,
                                    north_vel_standard_names_list)
    if north_vel_obj is None:
        raise RuntimeError('NorthVelocity object can not be found in ' +
                           'netCDF file.')

    # East Velocity Error
    east_vel_error_names_list = ['east_err', 'dopx']
    east_vel_error_standard_names_list = []
    east_vel_error_obj = search_variable(agg, east_vel_error_names_list,
                                         east_vel_error_standard_names_list)

    # North Velocity Error
    north_vel_error_names_list = ['north_err', 'dopy']
    north_vel_error_standard_names_list = []
    north_vel_error_obj = search_variable(agg, north_vel_error_names_list,
                                          north_vel_error_standard_names_list)

    return datetime_obj, lon_obj, lat_obj, east_vel_obj, north_vel_obj, \
        east_vel_error_obj, north_vel_error_obj
