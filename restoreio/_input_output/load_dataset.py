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

import sys
import netCDF4
import pyncml
import os.path

try:
    # Python 3
    from urllib.parse import urlparse
except ImportError:
    # python 2
    from urlparse import urlparse

__all__ = ['load_dataset']


# ==================
# Load Local Dataset
# ==================

def load_local_dataset(filename):
    """
    Opens either ncml or nc file and returns the aggregation file object.
    """

    print("Message: Loading data ... ")
    sys.stdout.flush()

    # Check file extension
    file_extension = os.path.splitext(filename)[1]
    if file_extension == '.ncml':

        # Change directory
        data_dir = os.path.dirname(filename)
        current_dir = os.getcwd()
        os.chdir(data_dir)

        # NCML
        try:
            ncml_string = open(filename, 'r').read()
            ncml_string = ncml_string.encode('ascii')
            ncml = pyncml.etree.fromstring(ncml_string)
            nc = pyncml.scan(ncml=ncml)

            # Get nc files list
            files_list = [f.path for f in nc.members]
            os.chdir(current_dir)

            # Aggregate
            agg = netCDF4.MFDataset(files_list, aggdim='t')

        except BaseException as error:
            print('ERROR: Can not read local ncml file: ' + filename)
            raise error

        return agg

    elif file_extension == '.nc':

        try:
            nc = netCDF4.Dataset(filename)
        except BaseException as error:
            print('ERROR: Can not read local netcdf file: ' + filename)
            raise error

        return nc

    else:
        raise ValueError("File should be either *.ncml or *.nc.")


# ===================
# Load Remote Dataset
# ===================

def load_remote_dataset(url):
    """
    url can be point to a *.nc or *.ncml file.
    """

    try:
        nc = netCDF4.Dataset(url)
    except BaseException as error:
        print('ERROR: Can not read remote file: ' + url)
        raise error

    return nc


# ============
# Load Dataset
# ============

def load_dataset(input_filename):
    """
    Dispatches the execution to either of the following two functions:

    1. load_local_dataset: For files where the input_filename is a path on the
       local machine.
    2. load_remote_dataset: For files remotely where input_filename is a url.
    """

    # Check if the input_filename has a "host" name
    if bool(urlparse(input_filename).netloc):
        # input_filename is a url
        return load_remote_dataset(input_filename)
    else:
        # input_filename is a path
        return load_local_dataset(input_filename)
