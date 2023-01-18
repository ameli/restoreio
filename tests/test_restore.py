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

    input = 'Monterey_Small_2km_Hourly_2017_01.nc'
    output = 'output.nc'
    timeframe = 117

    # Absolute path
    dir = os.path.dirname(os.path.realpath(__file__))
    input = os.path.join(dir, input)
    output = os.path.join(dir, output)

    # Check input exists
    if not os.path.exists(input):
        raise RuntimeError('File: %s does not exists.' % input)

    # Restore main file
    restore(input, min_file_index='', max_file_index='', output=output,
            sweep=False, detect_land=2, fill_coast=False, convex_hull=False,
            alpha=20, refine_grid=1, timeframe=timeframe,
            uncertainty_quant=False, num_ensembles=200, num_modes=None,
            kernel_window=5, scale_error=0.08, plot=True)

    # Uncertainty quantification
    restore(input, min_file_index='', max_file_index='', output=output,
            sweep=False, detect_land=2, fill_coast=False, convex_hull=False,
            alpha=20, refine_grid=1, timeframe=timeframe,
            uncertainty_quant=True, num_ensembles=200, num_modes=None,
            kernel_window=5, scale_error=0.08, plot=True)

    # Remove outputs
    remove_file('*.svg')
    remove_file('*.pdf')
    remove_file(output)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    test_restore()
