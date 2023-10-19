#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

from restoreio import restore
import os
import glob


# ===========
# remove file
# ===========

def remove_file(filename):
    """
    Remove file.
    """

    # Get a list of all files matching wildcard
    files_list = glob.glob(filename)

    # Iterate over files
    for file in files_list:
        try:
            os.remove(file)
        except BaseException as error:
            print('An exception occurred: {}'.format(error))
            print("Error while removing file : ", file)


# ============
# test restore
# ============

def test_restore():
    """
    Test for `restore` function.
    """

    # Data
    input = 'Monterey_Small_2km_Hourly_2017_01.nc'
    output = 'output_local_data.nc'
    min_lon = float('nan')
    max_lon = float('nan')
    min_lat = float('nan')
    max_lat = float('nan')
    time = '2017-01-25T03:00:00'

    # Absolute path
    dir = os.path.dirname(os.path.realpath(__file__))
    input = os.path.join(dir, input)
    output = os.path.join(dir, output)
    plot = True

    # Check input exists
    if not os.path.exists(input):
        raise RuntimeError('File: %s does not exists.' % input)

    # Restore main file
    restore(input, min_file_index='', max_file_index='', output=output,
            min_lon=min_lon, max_lon=max_lon, min_lat=min_lat, max_lat=max_lat,
            time=time, sweep=False, detect_land=True, fill_coast=False,
            convex_hull=False, alpha=20, refine_grid=1,
            uncertainty_quant=False, plot=plot, verbose=True, terminate=False)

    # Uncertainty quantification
    restore(input, min_file_index='', max_file_index='', output=output,
            min_lon=min_lon, max_lon=max_lon, min_lat=min_lat, max_lat=max_lat,
            time=time, sweep=False, detect_land=True, fill_coast=False,
            convex_hull=False, alpha=20, refine_grid=1, uncertainty_quant=True,
            num_samples=200, ratio_num_modes=1, kernel_width=5,
            scale_error=0.08, write_samples=True, plot=plot, verbose=True,
            terminate=False)

    # Remove outputs
    remove_file('*.svg')
    remove_file('*.pdf')
    remove_file(output)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    test_restore()
