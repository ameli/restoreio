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

__all__ = ['write_output_file']


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
        datetime_info,
        longitude,
        latitude,
        mask_info,
        u_all_times_inpainted,
        v_all_times_inpainted,
        u_all_times_inpainted_error,
        v_all_times_inpainted_error,
        fill_value,
        output_filename,
        verbose=True):
    """
    Writes the inpainted array to an output netcdf file.
    """

    if verbose:
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

    # Datetime
    output_datetime = output_file.createVariable(
            'time', numpy.dtype('float64').char, ('time', ))
    output_datetime[:] = datetime_info['array']
    output_datetime.units = datetime_info['unit']
    output_datetime.calendar = datetime_info['calendar']
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

    if verbose:
        print("Wrote to: %s." % output_filename)
        print("Message: Writing to NetCDF file ... Done.")
        sys.stdout.flush()
