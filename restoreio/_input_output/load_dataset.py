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
from .._server_utils import terminate_with_error

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

def load_local_dataset(filename, verbose=False):
    """
    Opens either ncml or nc file and returns the aggregation file object.
    """

    if verbose:
        print("Message: Loading local data files ... ")
    sys.stdout.flush()

    # Check file extension
    file_extension = os.path.splitext(filename)[1]
    if file_extension in ['.ncml', '.ncml.gz']:

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

    elif file_extension in ['.nc', '.nc4', '.ncd', '.nc.gz']:

        try:
            nc = netCDF4.Dataset(filename)
        except BaseException as error:
            print('ERROR: Can not read local netcdf file: ' + filename)
            raise error

        return nc

    else:
        terminate_with_error(
            "File format %s is not recognized." % file_extension)


# ===================
# Load Remote Dataset
# ===================

def load_remote_dataset(url, verbose=False):
    """
    url can be point to a *.nc or *.ncml file.
    """

    if verbose:
        print("Message: Connecting to remote data server ... ")
    sys.stdout.flush()

    # Check URL is opendap
    if (url.startswith('http://') is False) and \
       (url.startswith('https://') is False):
        terminate_with_error(
            'Input data URL does not seem to be a URL. A URL should start ' +
            'with "http://" or "https://".')

    elif ("/thredds/dodsC/" not in url) and ("opendap" not in url):
        terminate_with_error(
            'Input data URL is not an OpenDap URL or is not hosted on a ' +
            'THREDDs server. Check if your data URL contains ' +
            '"/thredds/dodsC/" or "/opendap/".')

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
            terminate_with_error(
                'The input data URL is not an netcdf file. The URL should ' +
                'end with ".nc", ".nc4", ".ncd", ".nc.gz", ".ncml", ' +
                '".ncml.gz", or without file extension.')

    try:

        # nc = open_url(url)
        nc = netCDF4.Dataset(url)

    except OSError:
        raise RuntimeError('Unable to read %s.' % url)

    return nc


# ============
# Load Dataset
# ============

def load_dataset(input_filename, verbose=False):
    """
    Dispatches the execution to either of the following two functions:

    1. load_local_dataset: For files where the input_filename is a path on the
       local machine.
    2. load_remote_dataset: For files remotely where input_filename is a url.
    """

    if input_filename == '':
        terminate_with_error(
            'Input data is empty. You should provide a local filename or an ' +
            'OpenDap URL or remote data.')

    # Check if the input_filename has a "host" name
    if bool(urlparse(input_filename).netloc):
        # input_filename is a url
        return load_remote_dataset(input_filename, verbose=verbose)
    else:
        # input_filename is a path
        return load_local_dataset(input_filename, verbose=verbose)
