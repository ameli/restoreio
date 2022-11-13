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

__all__ = ['detect_land_ocean']

# =================
# Detect Land Ocean
# =================

def detect_land_ocean(lon, lat, method)

        if method == 0:
            # Returns nan for Land indices, and returns all available indices for ocean.
            land_indices, ocean_indices = detect_land_ocean(Longitude, Latitude, arguments['ExcludeLandFromOcean'])
        elif method == 1:
            # Separate land and ocean. Most accurate, very slow for points on land.
            land_indices, ocean_indices = Geography.FindLandAndocean_indices1(Longitude, Latitude)
        elif method == 2:
            # Separate land and ocean. Least accurate, very fast
            land_indices, ocean_indices = Geography.FindLandAndocean_indices2(Longitude, Latitude)
        elif method == 3:
            # Currently Not working well.
            land_indices, ocean_indices = Geography.FindLandAndocean_indices3(Longitude, Latitude)  # Not working (land are not detected)
        else:
            raise RuntimeError("ExcludeLandFromOcean option is invalid.")


# ============================
# Do not Detect Land And Ocean
# ============================

def do_not_detect_land_and_ocean(Longitude, Latitude):
    """
    This function is as oppose to "FindLandAndOceanIndices". If the user choose not to detect any land, we treat the entire 
    domain as it is in ocean. So in this function we return a LandIdices as nan, and Ocean Indices as all available indices
    in the grid.
    """

    # Do not detect any land.
    LandIndices = numpy.nan

    # We treat all the domain as it is the ocean
    OceanIndicesList = []

    for LatitudeIndex in range(Latitude.size):
        for LongitudeIndex in range(Longitude.size):
            Tuple = (LatitudeIndex, LongitudeIndex)
            OceanIndicesList.append(Tuple)

    # Convert form list to array 
    OceanIndices = numpy.array(OceanIndicesList, dtype=int)

    return LandIndices, OceanIndices

# =============================
# Detect Land Ocean in Parallel
# =============================

def detect_land_ocean_in_parallel(map, Longitude, Latitude, PointId):
    """
    This function is used in the parallel section of "FindLandAndOceanIndices1". This function is passed
    to pool.imap_unoderd as a partial function. The parallel 'for' loop section iterates over the forth 
    argument 'PointIds'.
    """

    LandIndicesListInProcess = []
    OceanIndicesListInProcess = []

    # for PointId in PointIds:

    # Convert PointId to index
    LongitudeIndex = PointId % Longitude.size
    LatitudeIndex = int(PointId / Longitude.size)

    # Determine where the point is located at
    x, y = map(Longitude[LongitudeIndex], Latitude[LatitudeIndex])
    Tuple = (LatitudeIndex, LongitudeIndex) # order should be lat, lon to be consistent with data array
    if map.is_land(x, y):
        LandIndicesListInProcess.append(Tuple)
    else:
        OceanIndicesListInProcess.append(Tuple)

    return LandIndicesListInProcess, OceanIndicesListInProcess

# ===================
# Detect Land Ocean 1
# ===================

def detect_land_ocean_1(Longitude, Latitude):
    """
    Method:
    This function uses basemap.is_land(). It is very accurate, but for points inside land it is very slow.
    So if the grid has many points inside land it takes several minutes to finish.

    Description:
    Creates two arrays of sizes Px2 and Qx2 where each are a list of indices(i, j) of longitudes and latitudes.
    The first array are indices of points on land and the second is the indices of points on ocean. Combination 
    of the two list creates the ALL points on the grid, irregardless of wether they are missing points or valid points.

    Inputs:
    1. Longitude: 1xN array
    2. Latitude 1xM array
    
    Outputs:
    1. LandIndices: Px2 array of (i, j) indices of points on land
    2. OceanIndices: Qx2 array of (i, j) indices of points on ocean

    In above: P + Q = N * M.

    The land polygins are based on: "GSHHG - A Global Self-consistent, Hierarchical, High-resolution Geography Database"
    The data of coastlines are available at: https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html

    IMPORTANT NOTE:
    Order of LandIndices array in each tuple is (Latitude, Longitude), not (Longitude, Latitude).
    This is in order to be consistent with the Data array (velocities).
    Indeed, this is how the data should be stored and also viewed geophysically.
    """

    print("Message: Detecting land area ... ")
    sys.stdout.flush()

    # Define area to create basemap with. Offset it neded to include boundary points in to basemap.is_land()
    LongitudeOffset = 0.05 * numpy.abs(Longitude[-1] - Longitude[0])
    LatitudeOffset = 0.05 * numpy.abs(Latitude[-1] - Latitude[0])

    MinLongitude = numpy.min(Longitude) - LongitudeOffset
    MidLongitude = numpy.mean(Longitude)
    MaxLongitude = numpy.max(Longitude) + LongitudeOffset
    MinLatitude = numpy.min(Latitude) - LatitudeOffset
    MidLatitude = numpy.mean(Latitude)
    MaxLatitude = numpy.max(Latitude) + LatitudeOffset

    # Create basemap
    map = Basemap( \
            projection='aeqd', \
            llcrnrlon=MinLongitude, \
            llcrnrlat=MinLatitude, \
            urcrnrlon=MaxLongitude, \
            urcrnrlat=MaxLatitude, \
            area_thresh = 0.001, \
            lon_0 = MidLongitude, \
            lat_0 = MidLatitude, \
            resolution='f')

    print("Message: Locate grid points inside/outside land ...")
    sys.stdout.flush()

    # Multiprocessing
    NumProcessors = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=NumProcessors)

    # Iterable list
    # PointIds = numpy.arange(Latitude.size * Longitude.size).tolist()
    NumPointIds = Latitude.size * Longitude.size
    PointIds = range(NumPointIds)

    # Determine chunk size
    ChunkSize = int(len(PointIds) / NumProcessors)
    Ratio = 4.0
    ChunkSize = int(ChunkSize / Ratio)
    if ChunkSize > 40:
        ChunkSize = 40
    elif ChunkSize < 1:
        ChunkSize = 1

    # Partial function
    FindLandAndOceanIndicesInParallel_PartialFunct = partial( \
            FindLandAndOceanIndicesInParallel, \
            map, \
            Longitude, \
            Latitude)

    # List of output Ids
    LandIndicesList = []
    OceanIndicesList = []
 
    # Parallel section
    for LandIndicesListInProcess, OceanIndicesListInProcess in pool.imap_unordered(FindLandAndOceanIndicesInParallel_PartialFunct, PointIds, chunksize=ChunkSize):
        LandIndicesList.extend(LandIndicesListInProcess)
        OceanIndicesList.extend(OceanIndicesListInProcess)

    # Convert list to numpy array
    LandIndices = numpy.array(LandIndicesList, dtype=int)
    OceanIndices = numpy.array(OceanIndicesList, dtype=int)

    print("Message: Detecting land area ... Done.")
    sys.stdout.flush()

    return LandIndices, OceanIndices

# ===================
# Detect Land ocean 2
# ===================

def detect_land_ocean_2(Longitude, Latitude):
    """
    Method:
    This method uses maskoceans(). It is very fast but has less resolution than basemap.is_land().
    For higher resolution uses "FindLandAndOceanIndices()" function in this file.
    """

    print("Message: Detecting land area ... ")
    sys.stdout.flush()

    # Create a fake array, we will mask it later on ocean areas.
    Array = numpy.ma.zeros((Latitude.size, Longitude.size))

    # Mesh of latitudes and logitudes
    LongitudeGrid, LatitudeGrid = numpy.meshgrid(Longitude, Latitude)

    # Mask ocean on the array
    Array_MaskedOcean = maskoceans(LongitudeGrid, LatitudeGrid, Array, resolution='f', grid=1.25)

    # List of output Ids
    LandIndicesList = []
    OceanIndicesList = []

    for LatitudeIndex in range(Latitude.size):
        for LongitudeIndex in range(Longitude.size):
            Tuple = (LatitudeIndex, LongitudeIndex)
            if Array_MaskedOcean.mask[LatitudeIndex, LongitudeIndex] == True:
                # Point is masked, it means it is in the ocean
                OceanIndicesList.append(Tuple)
            else:
                # Point is not masked, it means it is on land
                LandIndicesList.append(Tuple)

    # Convert list to numpy array
    LandIndices = numpy.array(LandIndicesList, dtype=int)
    OceanIndices = numpy.array(OceanIndicesList, dtype=int)

    print("Message: Detecting land area ... Done.")
    sys.stdout.flush()

    return LandIndices, OceanIndices

# ===================
# Detect Land Ocean 3
# ===================

def detect_land_ocean_3(Longitude, Latitude):
    """
    Method:
    This function uses polygon.contain_point.
    This is similar to FindLandAndOceanIndices but currently it is inaccurate. This is not detecting any land indices since the
    polygons are not closed.
    """

    print("Message: Detecting land area ... ")
    sys.stdout.flush()

    # Define area to create basemap with. Offset it neded to include boundary points in to basemap.is_land()
    LongitudeOffset = 0.05 * numpy.abs(Longitude[-1] - Longitude[0])
    LatitudeOffset = 0.05 * numpy.abs(Latitude[-1] - Latitude[0])

    MinLongitude = numpy.min(Longitude) - LongitudeOffset
    MidLongitude = numpy.mean(Longitude)
    MaxLongitude = numpy.max(Longitude) + LongitudeOffset
    MinLatitude = numpy.min(Latitude) - LatitudeOffset
    MidLatitude = numpy.mean(Latitude)
    MaxLatitude = numpy.max(Latitude) + LatitudeOffset

    # Create basemap
    map = Basemap( \
            projection='aeqd', \
            llcrnrlon=MinLongitude, \
            llcrnrlat=MinLatitude, \
            urcrnrlon=MaxLongitude, \
            urcrnrlat=MaxLatitude, \
            area_thresh = 0.001, \
            lon_0 = MidLongitude, \
            lat_0 = MidLatitude, \
            resolution='f')

    print("Message: Locate grid points inside/outside land ...")
    sys.stdout.flush()

    # List of output Ids
    LandIndicesList = []
    OceanIndicesList = []

    PointsInLandStatusArray = numpy.zeros((Latitude.size, Longitude.size), dtype=bool)

    Polygons = [Path(p.boundary) for p in map.landpolygons]

    for Polygon in Polygons:

        for LatitudeIndex in range(Latitude.size):
            for LongitudeIndex in range(Longitude.size):
                x, y = map(Latitude[LatitudeIndex], Longitude[LongitudeIndex])
                Location = numpy.array([x, y])
                PointsInLandStatusArray[LatitudeIndex, LongitudeIndex] += Polygon.contains_point(Location)

    # Retrieve array to list of indices
    for LatitudeIndex in range(Latitude.size):
        for LongitudeIndex in range(Longitude.size):
            Tuple = (LatitudeIndex, LongitudeIndex)
            PointIsInLand = PointsInLandStatusArray[LatitudeIndex, LongitudeIndex]

            if PointIsInLand == True:
                LandIndicesList.append(Tuple)
            else:
                OceanIndicesList.append(Tuple)

    # Convert list to numpy array
    LandIndices = numpy.array(LandIndicesList, dtype=int)
    OceanIndices = numpy.array(OceanIndicesList, dtype=int)

    print(LandIndices)
    # print(OceanIndices)

    print("Message: Detecting land area ... Done.")
    sys.stdout.flush()

    return LandIndices, OceanIndices
