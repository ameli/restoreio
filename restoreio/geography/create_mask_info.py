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
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap, maskoceans
import multiprocessing
from functools import partial
import sys

# Convex Hull
from scipy.spatial import ConvexHull
from matplotlib import path

# Alpha shape
import shapely.geometry
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay

__all__ = ['create_mask_info']


# ================
# Create Mask Info
# ================

def create_mask_info( \
            U_OneTime, \
            LandIndices, \
            MissingIndicesInOceanInsideHull, \
            MissingIndicesInOceanOutsideHull, \
            ValidIndices):
    """
    Create a masked array.

    0:  Valid Indices
    1:  MissingIndicesInOceanInsideHull
    2:  MissingIndicesInOceanOutsideHull
    -1: LandIndices
    """

    # zero for all valid indices
    MaskInfo = numpy.zeros(U_OneTime.shape, dtype=int)

    # Missing indices in ocean inside hull
    for i in range(MissingIndicesInOceanInsideHull.shape[0]):
        MaskInfo[MissingIndicesInOceanInsideHull[i, 0], MissingIndicesInOceanInsideHull[i, 1]] = 1

    # Missing indices in ocean outside hull
    for i in range(MissingIndicesInOceanOutsideHull.shape[0]):
        MaskInfo[MissingIndicesInOceanOutsideHull[i, 0], MissingIndicesInOceanOutsideHull[i, 1]] = 2

    # Land indices
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):
            MaskInfo[LandIndices[i, 0], LandIndices[i, 1]] = -1

    return MaskInfo
