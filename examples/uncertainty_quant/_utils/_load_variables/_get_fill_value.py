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

__all__ = ['get_fill_value']


# ==============
# get fill value
# ==============

def get_fill_value(east_vel_obj, north_vel_obj):
    """
    Finds missing value (or fill value) from wither of east of north velocity
    objects.
    """

    # Missing Value
    if hasattr(east_vel_obj, '_FillValue') and \
            (not numpy.isnan(float(east_vel_obj._FillValue))):
        fill_value = numpy.fabs(float(east_vel_obj._FillValue))

    elif hasattr(north_vel_obj, '_FillValue') and \
            (not numpy.isnan(float(north_vel_obj._FillValue))):
        fill_value = numpy.fabs(float(north_vel_obj._FillValue))

    elif hasattr(east_vel_obj, 'missing_value') and \
            (not numpy.isnan(float(east_vel_obj.missing_value))):
        fill_value = numpy.fabs(float(east_vel_obj.missing_value))

    elif hasattr(north_vel_obj, 'missing_value') and \
            (not numpy.isnan(float(north_vel_obj.missing_value))):
        fill_value = numpy.fabs(float(north_vel_obj.missing_value))

    elif hasattr(east_vel_obj, 'fill_value') and \
            (not numpy.isnan(float(east_vel_obj.fill_value))):
        fill_value = numpy.fabs(float(east_vel_obj.fill_value))

    elif hasattr(north_vel_obj, 'fill_value') and \
            (not numpy.isnan(float(north_vel_obj.fill_value))):
        fill_value = numpy.fabs(float(north_vel_obj.fill_value))

    else:
        fill_value = 999.0

    return fill_value
