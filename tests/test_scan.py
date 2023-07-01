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

from restoreio import scan


# =========
# test scan
# =========

def test_scan():
    """
    Test for `scan` function.
    """

    # Martha's Vineyard
    input = 'https://transport.me.berkeley.edu/thredds/dodsC/root/' + \
            'WHOI-HFR/WHOI_HFR_2014_original.nc'

    # Run script
    scan(input, scan_velocity=True, terminate=False, verbose=True)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    test_scan()
