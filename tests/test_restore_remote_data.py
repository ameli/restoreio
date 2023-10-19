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

    # The Monterey bay data is commented since often their server is down and
    # causes this test script to halt.
    # Monterey Bay data
    # input = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/' + \
    #         'hourly/' + \
    #         'RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd'
    # min_lon = -122.344
    # max_lon = -121.781
    # min_lat = 36.507
    # max_lat = 36.9925
    # time = '2017-01-25T03:00:00'

    # Martha's Vineyard
    input = 'https://transport.me.berkeley.edu/thredds/dodsC/root/' + \
            'WHOI-HFR/WHOI_HFR_2014_original.nc'
    min_lon = -70.7
    max_lon = -70.5
    min_lat = 41.2
    max_lat = 41.3
    min_time = '2014-07-01T20:00:00'
    max_time = '2014-07-03T20:00:00'

    # Output
    output = 'output_remote_data.nc'
    plot = False

    # Absolute path
    dir = os.path.dirname(os.path.realpath(__file__))
    output = os.path.join(dir, output)

    # Restore main file
    restore(input, min_file_index='', max_file_index='', output=output,
            min_lon=min_lon, max_lon=max_lon, min_lat=min_lat, max_lat=max_lat,
            min_time=min_time, max_time=max_time, sweep=False,
            detect_land=True, fill_coast=False, convex_hull=False, alpha=20,
            refine_grid=1, uncertainty_quant=False, plot=plot, verbose=True)

    # These lines are commented since WHOI-HFR data don't have error variables.
    # Uncertainty quantification
    # restore(input, min_file_index='', max_file_index='', output=output,
    #         min_lon=min_lon, max_lon=max_lon, min_lat=min_lat,
    #         max_lat=max_lat, time=time, sweep=False, detect_land=True,
    #         fill_coast=False, convex_hull=False, alpha=20, refine_grid=1,
    #         uncertainty_quant=True, num_samples=200, ratio_num_modes=1,
    #         kernel_width=5, scale_error=0.08, write_samples=True,
    #         plot=plot, verbose=True)

    # Remove outputs
    remove_file('*.svg')
    remove_file('*.pdf')
    remove_file(output)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    test_restore()
