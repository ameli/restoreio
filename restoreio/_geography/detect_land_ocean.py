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
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap, maskoceans
import multiprocessing
from functools import partial
import sys

__all__ = ['detect_land_ocean']


# =================
# Detect Land Ocean
# =================

def detect_land_ocean(lon, lat, method):
    """
    Separates land and ocean indices.
    """

    if method == 0:
        # Sets nan for land indices, all other indices are ocean.
        land_indices, ocean_indices = do_not_detect_land_ocean(lon, lat)
    elif method == 1:
        # Separate land and ocean. Most accurate, very slow for land.
        land_indices, ocean_indices = detect_land_ocean_1(lon, lat)
    elif method == 2:
        # Separate land and ocean. Least accurate, very fast
        land_indices, ocean_indices = detect_land_ocean_2(lon, lat)
    elif method == 3:
        # Currently not working, do not use (land are not detected)
        land_indices, ocean_indices = detect_land_ocean_3(lon, lat)
    else:
        raise RuntimeError("ExcludeLandFromOcean option is invalid.")

    return land_indices, ocean_indices


# ========================
# Do not Detect Land Ocean
# ========================

def do_not_detect_land_ocean(lon, lat):
    """
    This function is as oppose to "detect_land_ocean_1,2,3". If user chooses
    to not to detect any land, we treat the entire domain as it is in ocean. So
    in this function we return a land_indices as nan, and Ocean Indices as all
    available indices in the grid.
    """

    # Do not detect any land.
    land_indices = numpy.nan

    # We treat all the domain as it is the ocean
    ocean_indices_list = []

    for lat_index in range(lat.size):
        for lon_index in range(lon.size):
            tuple = (lat_index, lon_index)
            ocean_indices_list.append(tuple)

    # Convert form list to array
    ocean_indices = numpy.array(ocean_indices_list, dtype=int)

    return land_indices, ocean_indices


# ============================
# Detect Land Ocean 1 Parallel
# ============================

def detect_land_ocean_1_parallel(map, lon, lat, point_id):
    """
    This function is used in the parallel section of "detect_land_ocean_1".
    This function is passed to pool.imap_unoderd as a partial function. The
    parallel 'for' loop section iterates over the forth argument 'point_ids'.
    """

    land_indices_list_in_process = []
    ocean_indices_list_in_process = []

    # Convert point_id to index
    lon_index = point_id % lon.size
    lat_index = int(point_id / lon.size)

    # Determine where the point is located at
    x, y = map(lon[lon_index], lat[lat_index])

    # In tuple, order should be lat, lon to be consistent with data array
    tuple = (lat_index, lon_index)

    if map.is_land(x, y):
        land_indices_list_in_process.append(tuple)
    else:
        ocean_indices_list_in_process.append(tuple)

    return land_indices_list_in_process, ocean_indices_list_in_process


# ===================
# Detect Land Ocean 1
# ===================

def detect_land_ocean_1(lon, lat):
    """
    Method:
    This function uses basemap.is_land(). It is very accurate, but for points
    inside land it is very slow. So if the grid has many points inside land it
    takes several minutes to finish.

    Description:
    Creates two arrays of sizes Px2 and Qx2 where each are a list of indices
    (i, j) of longitudes and latitudes. The first array are indices of points
    on land and the second is the indices of points on ocean. Combination of
    the two list creates the ALL points on the grid, irregardless of whether
    they are missing points or valid points.

    Inputs:
    1. lon: 1xN array
    2. lat 1xM array

    Outputs:
    1. land_indices: Px2 array of (i, j) indices of points on land
    2. ocean_indices: Qx2 array of (i, j) indices of points on ocean

    In above: P + Q = N * M.

    The land polygins are based on: "GSHHG - A Global Self-consistent,
    Hierarchical, High-resolution Geography Database" The data of coastlines
    are available at: https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html

    IMPORTANT NOTE:
    Order of land_indices array in each tuple is (lat, lon), not (lon, lat).
    This is in order to be consistent with the Data array (velocities).
    Indeed, this is how the data should be stored and also viewed
    geophysically.
    """

    print("Message: Detecting land area ... ")
    sys.stdout.flush()

    # Define area to create basemap with. An offset is needed to include
    # boundary points into basemap.is_land()
    lon_offset = 0.05 * numpy.abs(lon[-1] - lon[0])
    lat_offset = 0.05 * numpy.abs(lat[-1] - lat[0])

    min_lon = numpy.min(lon) - lon_offset
    mid_lon = numpy.mean(lon)
    max_lon = numpy.max(lon) + lon_offset
    min_lat = numpy.min(lat) - lat_offset
    mid_lat = numpy.mean(lat)
    max_lat = numpy.max(lat) + lat_offset

    # Create basemap
    map = Basemap(
            projection='aeqd', llcrnrlon=min_lon, llcrnrlat=min_lat,
            urcrnrlon=max_lon, urcrnrlat=max_lat, area_thresh=0.001,
            lon_0=mid_lon, lat_0=mid_lat, resolution='f')

    print("Message: Locate grid points inside/outside land ...")
    sys.stdout.flush()

    # Multiprocessing
    num_processors = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processors)

    # Iterable list
    # point_ids = numpy.arange(lat.size * lon.size).tolist()
    num_point_ids = lat.size * lon.size
    point_ids = range(num_point_ids)

    # Determine chunk size
    chunk_size = int(len(point_ids) / num_processors)
    ratio = 4.0
    chunk_size = int(chunk_size / ratio)
    if chunk_size > 40:
        chunk_size = 40
    elif chunk_size < 1:
        chunk_size = 1

    # Partial function
    detect_land_ocean_1_parallel_partial_func = partial(
            detect_land_ocean_1_parallel, map, lon, lat)

    # List of output Ids
    land_indices_list = []
    ocean_indices_list = []

    # Parallel section
    for land_indices_list_in_process, ocean_indices_list_in_process in \
            pool.imap_unordered(detect_land_ocean_1_parallel_partial_func,
                                point_ids, chunksize=chunk_size):

        land_indices_list.extend(land_indices_list_in_process)
        ocean_indices_list.extend(ocean_indices_list_in_process)

    # Convert list to numpy array
    land_indices = numpy.array(land_indices_list, dtype=int)
    ocean_indices = numpy.array(ocean_indices_list, dtype=int)

    print("Message: Detecting land area ... Done.")
    sys.stdout.flush()

    return land_indices, ocean_indices


# ===================
# Detect Land ocean 2
# ===================

def detect_land_ocean_2(lon, lat):
    """
    Method:
    This method uses maskoceans(). It is very fast but has less resolution than
    basemap.is_land(). For higher resolution uses "FindLandAndocean_indices()"
    function in this file.
    """

    print("Message: Detecting land area ... ")
    sys.stdout.flush()

    # Create a fake array, we will mask it later on ocean areas.
    array = numpy.ma.zeros((lat.size, lon.size))

    # Mesh of latitudes and longitudes
    lon_grid, lat_grid = numpy.meshgrid(lon, lat)

    # Mask ocean on the array
    array_masked_ocean = maskoceans(lon_grid, lat_grid, array, resolution='f',
                                    grid=1.25)

    # List of output Ids
    land_indices_list = []
    ocean_indices_list = []

    for lat_index in range(lat.size):
        for lon_index in range(lon.size):
            tuple = (lat_index, lon_index)
            if bool(array_masked_ocean.mask[lat_index, lon_index]) is True:
                # Point is masked, it means it is in the ocean
                ocean_indices_list.append(tuple)
            else:
                # Point is not masked, it means it is on land
                land_indices_list.append(tuple)

    # Convert list to numpy array
    land_indices = numpy.array(land_indices_list, dtype=int)
    ocean_indices = numpy.array(ocean_indices_list, dtype=int)

    print("Message: Detecting land area ... Done.")
    sys.stdout.flush()

    return land_indices, ocean_indices


# ===================
# Detect Land Ocean 3
# ===================

def detect_land_ocean_3(lon, lat):
    """
    Method:
    This function uses polygon.contain_point.
    This is similar to FindLandAndocean_indices but currently it is inaccurate.
    This is not detecting any land indices since the polygons are not closed.
    """

    print("Message: Detecting land area ... ")
    sys.stdout.flush()

    # Define area to create basemap with. An offset it needed to include
    # boundary points in to basemap.is_land()
    lon_offset = 0.05 * numpy.abs(lon[-1] - lon[0])
    lat_offset = 0.05 * numpy.abs(lat[-1] - lat[0])

    min_lon = numpy.min(lon) - lon_offset
    mid_lon = numpy.mean(lon)
    max_lon = numpy.max(lon) + lon_offset
    min_lat = numpy.min(lat) - lat_offset
    mid_lat = numpy.mean(lat)
    max_lat = numpy.max(lat) + lat_offset

    # Create basemap
    map = Basemap(
            projection='aeqd', llcrnrlon=min_lon, llcrnrlat=min_lat,
            urcrnrlon=max_lon, urcrnrlat=max_lat, area_thresh=0.001,
            lon_0=mid_lon, lat_0=mid_lat, resolution='f')

    print("Message: Locate grid points inside/outside land ...")
    sys.stdout.flush()

    # List of output Ids
    land_indices_list = []
    ocean_indices_list = []

    land_points_status_array = numpy.zeros((lat.size, lon.size), dtype=bool)

    polygons = [Path(p.boundary) for p in map.landpolygons]

    for polygon in polygons:

        for lat_index in range(lat.size):
            for lon_index in range(lon.size):
                x, y = map(lat[lat_index], lon[lon_index])
                location = numpy.array([x, y])
                land_points_status_array[lat_index, lon_index] += \
                    polygon.contains_point(location)

    # Retrieve array to list of indices
    for lat_index in range(lat.size):
        for lon_index in range(lon.size):
            tuple = (lat_index, lon_index)
            points_is_land = land_points_status_array[lat_index, lon_index]

            if points_is_land is True:
                land_indices_list.append(tuple)
            else:
                ocean_indices_list.append(tuple)

    # Convert list to numpy array
    land_indices = numpy.array(land_indices_list, dtype=int)
    ocean_indices = numpy.array(ocean_indices_list, dtype=int)

    # print(land_indices)
    # print(ocean_indices)

    print("Message: Detecting land area ... Done.")
    sys.stdout.flush()

    return land_indices, ocean_indices
