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
from .license import License                                       # noqa: E402

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
        output_filename,
        datetime_info,
        longitude,
        latitude,
        mask_info,
        fill_value,
        u_all_times_inpainted,
        v_all_times_inpainted,
        u_all_times_inpainted_error,
        v_all_times_inpainted_error,
        u_all_ensembles_inpainted=None,
        v_all_ensembles_inpainted=None,
        write_samples=False,
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
    if write_samples:
        num_samples = u_all_ensembles_inpainted.shape[0]
        output_file.createDimension('sample', num_samples)

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

    # Mask
    mask = output_file.createVariable(
            'mask', numpy.dtype('float64').char, ('time', 'lat', 'lon', ),
            fill_value=fill_value, zlib=True)
    mask[:] = mask_info
    mask.coordinates = 'longitude latitude datetime'
    mask.missing_value = fill_value
    mask.coordsys = "geographic"
    mask.comment = \
        "Segmentation of the spatial domain to (1) land, (2) domain " + \
        "with known velocity, (3) domain with missing velocity, and (4) " + \
        "ocean. This variable contains integer values of -1, 0, 1, and 2, " + \
        "representing the following domains: \n " + \
        "-1: Indicates points that are identified to be on land. These " + \
        "locations are masked. \n " + \
        " 0: Indicates points that are identified to be in ocean with " + \
        "known velocity data in the input dataset. \n" + \
        " 1: Indicates points that are identified to be in ocean inside " + \
        "the data domain, but they have missing velocity data in the " + \
        "input file. These locations are reconstructed. \n" + \
        " 2: Indicates points that are identified to be in ocean but " + \
        "outside of the data domain. These locations are masked."

    # Velocity U
    output_u = output_file.createVariable(
            'east_vel', numpy.dtype('float64').char, ('time', 'lat', 'lon', ),
            fill_value=fill_value, zlib=True)
    output_u[:] = u_all_times_inpainted
    output_u.units = 'm s-1'
    output_u.standard_name = 'surface_eastward_sea_water_velocity'
    output_u.positive = 'toward east'
    output_u.coordinates = 'longitude latitude datetime'
    output_u.missing_value = fill_value
    output_u.coordsys = "geographic"
    output_u.comment = \
        "Reconstructed east component of velocity. In areas where the " + \
        "'mask' variable holds values of -1 or 2, the velocity variable " + \
        "is masked. In regions where the 'mask' variable is 0, the " + \
        "velocity variable in the output file maintains the same values " + \
        "as the input file. In regions where the 'mask' variable equals " + \
        "1, the velocity variable undergoes reconstruction."

    # Velocity V
    output_v = output_file.createVariable(
            'north_vel', numpy.dtype('float64').char, ('time', 'lat', 'lon', ),
            fill_value=fill_value, zlib=True)
    output_v[:] = v_all_times_inpainted
    output_v.units = 'm s-1'
    output_v.standard_name = 'surface_northward_sea_water_velocity'
    output_v.positive = 'toward north'
    output_v.coordinates = 'longitude latitude datetime'
    output_v.missing_value = fill_value
    output_v.coordsys = "geographic"
    output_v.comment = \
        "Reconstructed north component of velocity. In areas where the " + \
        "'mask' variable holds values of -1 or 2, the velocity variable " + \
        "is masked. In regions where the 'mask' variable is 0, the " + \
        "velocity variable in the output file maintains the same values " + \
        "as the input file. In regions where the 'mask' variable equals " + \
        "1, the velocity variable undergoes reconstruction."

    # Velocity U Error
    if u_all_times_inpainted_error is not None:
        output_u_error = output_file.createVariable(
                'east_err', numpy.dtype('float64').char,
                ('time', 'lat', 'lon', ), fill_value=fill_value, zlib=True)
        output_u_error[:] = u_all_times_inpainted_error
        output_u_error.units = 'm s-1'
        output_u_error.positive = 'toward east'
        output_u_error.coordinates = 'longitude latitude datetime'
        output_u_error.missing_value = fill_value
        output_u_error.coordsys = "geographic"
        output_u_error.comment = \
            "East component of velocity error. In areas where the 'mask' " + \
            "variable holds values of -1 or 2, the velocity error " + \
            "variable is masked. In regions where the 'mask' variable is " + \
            "0, the velocity error variable is obtained from the " + \
            "velocity error or GDOP variable from the input file. In " + \
            "regions where the 'mask' variable equals 1, the velocity " + \
            "error variable is obtained from the standard deviation of " + \
            "the velocity ensemble where the missing domain of each " + \
            "ensemble is reconstructed."

    # Velocity V Error
    if v_all_times_inpainted_error is not None:
        output_v_error = output_file.createVariable(
                'north_err', numpy.dtype('float64').char,
                ('time', 'lat', 'lon', ), fill_value=fill_value, zlib=True)
        output_v_error[:] = v_all_times_inpainted_error
        output_v_error.units = 'm s-1'
        output_v_error.positive = 'toward north'
        output_v_error.coordinates = 'longitude latitude datetime'
        output_v_error.missing_value = fill_value
        output_v_error.coordsys = "geographic"
        output_v_error.comment = \
            "North component of velocity error. In areas where the 'mask' " + \
            "variable holds values of -1 or 2, the velocity error " + \
            "variable is masked. In regions where the 'mask' variable is " + \
            "0, the velocity error variable is obtained from the " + \
            "velocity error or GDOP variable from the input file. In " + \
            "regions where the 'mask' variable equals 1, the velocity " + \
            "error variable is obtained from the standard deviation of " + \
            "the velocity ensemble where the missing domain of each " + \
            "ensemble is reconstructed."

    # Velocity U Ensemble
    if (write_samples is True) and (u_all_ensembles_inpainted is not None):
        output_u_ens = output_file.createVariable(
                'east_vel_ensembles', numpy.dtype('float64').char,
                ('sample', 'lat', 'lon', ), fill_value=fill_value, zlib=True)
        output_u_ens[:] = u_all_ensembles_inpainted
        output_u_ens.units = 'm s-1'
        output_u_ens.positive = 'toward east'
        output_u_ens.coordinates = 'longitude latitude ensemble'
        output_u_ens.missing_value = fill_value
        output_u_ens.coordsys = "geographic"
        output_u_ens.comment = \
            "Ensemble of the east component of velocity. The first " + \
            "ensemble is identical to the 'east_vel' variable. The rest " + \
            "of the ensemble are randomly generated correlated " + \
            "perturbations around the east velocity variable. The mean of " + \
            "the ensemble is equivalent to the 'east_vel' variable. The " + \
            "standard deviation of the ensemble is equal to the " + \
            "'east_err' variable."

    # Velocity V Ensemble
    if (write_samples is True) and (v_all_ensembles_inpainted is not None):
        output_v_ens = output_file.createVariable(
                'north_vel_ensembles', numpy.dtype('float64').char,
                ('sample', 'lat', 'lon', ), fill_value=fill_value, zlib=True)
        output_v_ens[:] = v_all_ensembles_inpainted
        output_v_ens.units = 'm s-1'
        output_v_ens.positive = 'toward east'
        output_v_ens.coordinates = 'longitude latitude ensemble'
        output_v_ens.missing_value = fill_value
        output_v_ens.coordsys = "geographic"
        output_v_ens.comment = \
            "Ensemble of the north component of velocity. The first " + \
            "ensemble is identical to the 'north_vel' variable. The rest " + \
            "of the ensemble are randomly generated correlated " + \
            "perturbations around the north velocity variable. The mean " + \
            "of the ensemble is equivalent to the 'north_vel' variable. " + \
            "The standard deviation of the ensemble is equal to the " + \
            "'north_err' variable."

    # -----------------
    # Global Attributes
    # -----------------

    # Author
    output_file.creator_name = License.author_name
    output_file.creator_email = License.author_email
    output_file.creation_date = time.strftime("%x %X %Z")
    output_file.institution = License.author_institution
    output_file.title = License.output_file_title
    output_file.history = License.output_file_history
    output_file.project = License.output_file_project
    output_file.acknowledgement = License.output_file_acknowledgement
    output_file.license = License.output_file_license

    # Data specific
    output_file.Conventions = 'CF-1.10'
    output_file.standard_name_vocabulary = 'CF Standard Name Table v30'
    output_file.COORD_SYSTEM = 'GEOGRAPHIC'
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
