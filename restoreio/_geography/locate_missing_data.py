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
import sys
from scipy.spatial import ConvexHull
from matplotlib import path
import shapely.geometry
from ._find_alpha_shapes import find_alpha_shapes

__all__ = ['locate_missing_data']


# =================
# Compute Max alpha
# =================

def _compute_max_alpha(lon, lat):
    """
    Computes the smallest possible alpha based on the smallest cell. The
    smallest cell (element) is a right angle formed between tree adjacent
    points on the grid with right edges delta_longitude and delta_latitude
    """

    diff_lon = numpy.diff(lon, 1)
    diff_lat = numpy.diff(lat, 1)

    min_delta_lon = numpy.min(diff_lon)
    min_delta_lat = numpy.min(diff_lat)

    min_circumcircle_radius = numpy.sqrt(
            min_delta_lon**2 + min_delta_lat**2) / 2.0

    max_alpha = 1.0 / min_circumcircle_radius

    return max_alpha


# ============================================================
# Find Status Of All Missing Points In Ocean With Concave Hull
# ============================================================

def _find_status_of_all_missing_points_in_ocean_with_concave_hull(
        hull_body_points_coord,
        all_missing_points_in_ocean_coord,
        alpha,
        all_missing_indices_in_ocean,
        lon,
        lat):
    """
    All points in ocean can be separated into valid points and missing points.
    The two arguments fo this functions are valid points (plus maybe the land
    points) and also the missing points in ocean.

    Input:
        - hull_body_points_coord: Nx2 numpy array. This is the coordinates of
          points that we will draw a concave hull around it. This can be either
          just the valid_points_coord, or the combination of valid_points_coord
          and the land_points_coord.

        - all_missing_points_in_ocean_coord: Mx2 array. This is the coordinate
          of all missing points in ocean.

        - alpha: The circumcircle radius = 1 / alpha. The larger alpha means
          the alpha shape attaches to the points more, where as smaller alpha
          mean the alpha shape is more tends to be the convex hull.

    Output:
        - all_missing_points_in_ocean_status_in_hull: A boolean array of size
          Mx2 (the same size as all_missing_points_in_ocean_coord). If a point
          is inside the concave hull the element of this array is flagged as
          True. Points outside are flagged as False.

        - hull_points_coord_list: A list that each member of tha list are the
          coordinates of one pf the separate concave hulls. There might be many
          separate concave hulls. For each, the corresponding member of the
          list is a Qx2 numpy array. Q is the number of points on the exterior
          (or boundary) or the polygon and it varies for each of the polygons.
    """

    # Find the concave hull of points
    concave_hull_polygon = find_alpha_shapes(hull_body_points_coord, alpha)

    # detect the number of shapes
    concave_hull_polygons_list = []
    num_shapes = 0
    if type(concave_hull_polygon) is shapely.geometry.polygon.Polygon:

        # Only one shape
        num_shapes = 1
        concave_hull_polygons_list.append(concave_hull_polygon)

    elif type(concave_hull_polygon) is \
            shapely.geometry.multipolygon.MultiPolygon:

        # Multi shapes
        concave_hull_polygons_list = list(concave_hull_polygon.geoms)

    else:
        raise RuntimeError("Invalid polygon type: %s."
                           % type(concave_hull_polygon))

    # Number of polygon shapes
    num_shapes = len(concave_hull_polygons_list)

    # Allocate output
    num_all_missing_points_in_ocean = \
        all_missing_points_in_ocean_coord.shape[0]
    all_missing_points_in_ocean_status_in_hull = numpy.zeros(
            num_all_missing_points_in_ocean, dtype=bool)

    # Find the all_missing_points_in_ocean_status_in_hull
    for j in range(num_shapes):
        # Iterate over all False points
        for i in range(num_all_missing_points_in_ocean):
            # Only check those points that are not yet seen to be inside one of
            # the shape polygons
            if (bool(all_missing_points_in_ocean_status_in_hull[i]) is False):

                point_coord = all_missing_points_in_ocean_coord[i, :]
                point_index = all_missing_indices_in_ocean[i, :]

                # Get delta_lon (Note: lon is the second index of points)
                if point_index[1] == lon.size - 1:
                    delta_lon = numpy.abs(lon[-1] - lon[-2])
                elif point_index[1] < lon.size - 1:
                    delta_lon = numpy.abs(
                            lon[point_index[1]+1] - lon[point_index[1]])
                else:
                    raise RuntimeError("Wrong lon index: %d, lon size: %d"
                                       % (point_index[1], lon.size))

                # Get delta_lat (Note: lat is the first index of points)
                if point_index[0] == lat.size - 1:
                    delta_lat = numpy.abs(lat[-1] - lat[-2])
                elif point_index[0] < lat.size - 1:
                    delta_lat = numpy.abs(
                            lat[point_index[0]+1] - lat[point_index[0]])
                else:
                    raise RuntimeError("Wrong lat index: %d, lat size: %d"
                                       % (point_index[0], lat.size))

                # Ratio of the element size which we check the auxiliary points
                delta_ratio = 0.05

                # Try the point itself:
                geometry_point_obj = shapely.geometry.Point(
                    point_coord[0], point_coord[1])
                point_status_in_ocean_in_hull = \
                    concave_hull_polygons_list[j].contains(geometry_point_obj)
                if bool(point_status_in_ocean_in_hull) is True:
                    all_missing_points_in_ocean_status_in_hull[i] = True
                    continue

                # Try point above
                point_coord_above = numpy.copy(point_coord)
                point_coord_above[1] += delta_lat * delta_ratio
                geometry_point_obj = shapely.geometry.Point(
                    point_coord_above[0], point_coord_above[1])
                point_status_in_ocean_in_hull = \
                    concave_hull_polygons_list[j].contains(geometry_point_obj)
                if bool(point_status_in_ocean_in_hull) is True:
                    all_missing_points_in_ocean_status_in_hull[i] = True
                    continue

                # Try point below
                point_coord_below = numpy.copy(point_coord)
                point_coord_below[1] -= delta_lat * delta_ratio
                geometry_point_obj = shapely.geometry.Point(
                        point_coord_below[0], point_coord_below[1])
                point_status_in_ocean_in_hull = \
                    concave_hull_polygons_list[j].contains(geometry_point_obj)
                if bool(point_status_in_ocean_in_hull) is True:
                    all_missing_points_in_ocean_status_in_hull[i] = True
                    continue

                # Try point Left
                point_coord_left = numpy.copy(point_coord)
                point_coord_left[0] -= delta_lon * delta_ratio
                geometry_point_obj = shapely.geometry.Point(
                    point_coord_left[0], point_coord_left[1])
                point_status_in_ocean_in_hull = \
                    concave_hull_polygons_list[j].contains(geometry_point_obj)
                if bool(point_status_in_ocean_in_hull) is True:
                    all_missing_points_in_ocean_status_in_hull[i] = True
                    continue

                # Try point Right
                point_coord_right = numpy.copy(point_coord)
                point_coord_right[0] += delta_lon * delta_ratio
                geometry_point_obj = shapely.geometry.Point(
                    point_coord_right[0], point_coord_right[1])
                point_status_in_ocean_in_hull = \
                    concave_hull_polygons_list[j].contains(geometry_point_obj)
                if bool(point_status_in_ocean_in_hull) is True:
                    all_missing_points_in_ocean_status_in_hull[i] = True
                    continue

    # Find hull_points_coord_list
    hull_points_coord_list = [None] * num_shapes
    for i in range(num_shapes):
        one_hull_points_coord_XY = concave_hull_polygons_list[i].exterior.xy
        hull_points_coord_list[i] = numpy.array(one_hull_points_coord_XY).T

    return all_missing_points_in_ocean_status_in_hull, \
        hull_points_coord_list


# ===========================================================
# Find Status Of All Missing Points In Ocean With Convex Hull
# ===========================================================

def _find_status_of_all_missing_points_in_ocean_with_convex_hull(
        hull_body_points_coord,
        all_missing_points_in_ocean_coord,
        all_missing_indices_in_ocean,
        lon,
        lat):
    """
    Al points in ocean can be separated into valid points and missing points.
    The two arguments of this function are valid points and missing points.

    Input:
        - hull_body_points_coord: Nx2 numpy array. This is the coordinate of
          the points that we will draw a convex hull around it.
          This is usually the valid_points_coord.

        - all_missing_points_in_ocean_coord: Mx2 numpy array. This is the
          coordinates of all missing points in ocean.

    Output:
        - all_missing_points_in_ocean_status_in_hull: A boolean array of size
          1xM (the same size as all_missing_points_in_ocean_coord). If a point
          is inside the convex hull the element on this array is flagged as
          True. Points outside are flagged as False.

        - hull_points_coord_list: A list that has one member. The only member
          (hull_points_coord_list[0]) is a numpy array of size Qx2 where Q is
          the number of convex hull exterior (boundary) points. These are the
          coordinates of the polygon that wraps the hull.
    """

    # Find the convex hull around data
    hull_polygon = ConvexHull(hull_body_points_coord)
    hull_points_coord = hull_body_points_coord[hull_polygon.vertices, :]
    hull_points_coord_list = [hull_points_coord]

    # Create path from hull points
    hull_path = path.Path(hull_points_coord)

    num_all_missing_points = all_missing_points_in_ocean_coord.shape[0]
    all_missing_points_in_ocean_status_in_hull = numpy.zeros(
            num_all_missing_points, dtype=bool)
    delta_ratio = 0.05

    # Check if missing points are inside the hull (True:Inside, False:Outside)
    for i in range(num_all_missing_points):
        point_coord = all_missing_points_in_ocean_coord[i, :]
        point_index = all_missing_indices_in_ocean[i, :]

        # Get delta_lon (Note: lon is the second index of points)
        if point_index[1] == lon.size - 1:
            delta_lon = numpy.abs(lon[-1] - lon[-2])
        elif point_index[1] < lon.size - 1:
            delta_lon = numpy.abs(lon[point_index[1]+1] - lon[point_index[1]])
        else:
            raise RuntimeError("Wrong lon index: %d, lon size: %d"
                               % (point_index[1], lon.size))

        # Get delta_lat (Note: lat is the first index of points)
        if point_index[0] == lat.size - 1:
            delta_lat = numpy.abs(lat[-1] - lat[-2])
        elif point_index[0] < lat.size - 1:
            delta_lat = numpy.abs(lat[point_index[0]+1] - lat[point_index[0]])
        else:
            raise RuntimeError("Wrong lat index: %d, lat size: %d"
                               % (point_index[0], lat.size))

        # Try the point itself:
        if hull_path.contains_point(point_coord) is True:
            all_missing_points_in_ocean_status_in_hull[i] = True
            continue

        # Try point above
        point_coord_above = numpy.copy(point_coord)
        point_coord_above[1] += delta_lat * delta_ratio
        if hull_path.contains_point(point_coord_above) is True:
            all_missing_points_in_ocean_status_in_hull[i] = True
            continue

        # Try point below
        point_coord_below = numpy.copy(point_coord)
        point_coord_below[1] -= delta_lat * delta_ratio
        if hull_path.contains_point(point_coord_below) is True:
            all_missing_points_in_ocean_status_in_hull[i] = True
            continue

        # Try point Left
        point_coord_left = numpy.copy(point_coord)
        point_coord_left[0] -= delta_lon * delta_ratio
        if hull_path.contains_point(point_coord_left) is True:
            all_missing_points_in_ocean_status_in_hull[i] = True
            continue

        # Try point Right
        point_coord_right = numpy.copy(point_coord)
        point_coord_right[0] += delta_lon * delta_ratio
        if hull_path.contains_point(point_coord_right) is True:
            all_missing_points_in_ocean_status_in_hull[i] = True
            continue

    return all_missing_points_in_ocean_status_in_hull, hull_points_coord_list


# ============================================================
# Exclude Points In Land Lake From Points In Ocean Inside Hull
# ============================================================

def _exclude_points_in_land_lake_from_points_in_ocean_in_hull(
        lon, lat,
        land_points_coord,
        all_missing_points_in_ocean_status_in_hull,
        all_missing_indices_in_ocean,
        all_missing_points_in_ocean_coord):
    """
    This functions removes some rows from "missing_indices_in_ocean_in_hull"
    and adds them to "missing_indices_in_ocean_outside_hull". The reason is
    that some points might belong to a lake on the land, but when we include
    land to the hull points, all points including the lakes are also considered
    as missing points inside hull. But these points are not in ocean.

    To detect these points, we draw another alpha shape, ONLY around the land
    and check which of the points inside hull are surrounded by the land's
    alpha shape. If a point is found, we remove is from the points inside hull
    list and add them to the points outside the hull.
    """

    # Do nothing if there is no land in the area.
    if bool(numpy.any(numpy.isnan(land_points_coord))):
        return all_missing_points_in_ocean_status_in_hull

    # True means the points that are identified to be inside the hull. We get
    # their ID with respect to this array.
    ids_in_hull = numpy.where(all_missing_points_in_ocean_status_in_hull)[0]

    # Getting the Indices of missing points inside hull
    missing_indices_in_ocean_in_hull = \
        all_missing_indices_in_ocean[ids_in_hull, :]

    # Get the coordinate of missing points inside the hull
    missing_points_in_ocean_in_hull_coord = \
        all_missing_points_in_ocean_coord[ids_in_hull, :]

    # Use large alpha to create an alpha shape that closely follows land points
    alpha = 60
    max_alpha = _compute_max_alpha(lon, lat)
    if alpha > max_alpha:
        alpha = max_alpha * 0.9

    missing_points_in_ocean_in_hull_status_in_land, land_points_coordList = \
        _find_status_of_all_missing_points_in_ocean_with_concave_hull(
            land_points_coord, missing_points_in_ocean_in_hull_coord,
            alpha, missing_indices_in_ocean_in_hull, lon, lat)

    # Edit original array with newer status. True means points are inside land.
    for i in range(ids_in_hull.size):
        point_is_inside_land = \
                missing_points_in_ocean_in_hull_status_in_land[i]

        # If point IS in land, edit original array and make it outside of hull
        if point_is_inside_land is True:
            point_id = ids_in_hull[i]
            all_missing_points_in_ocean_status_in_hull[point_id] = False

    return all_missing_points_in_ocean_status_in_hull


# ===================
# Locate Missing data
# ===================

def locate_missing_data(
        lon,
        lat,
        land_indices,
        data,
        include_land_for_hull,
        convex_hull,
        alpha):
    """
    All points in grid are divided into these groups:

    1- Land points: (Nx2 array)
       This is not defined by this function. Rather it is computer earlier
       before calling this function. The rest of the points are the points in
       the ocean.

    2- Valid points: (Mx2 array)
       The valid points are merelly defined on the valid points of the DATA,
       and they do not include the land area. All valid points are in the
       ocean.

       Note: The forth argument "data" can be any of North velocities and East
       velocities. This is used to find masked points.

    3- Convex Hull: (Kx2 array of x-y coordinates where K is the number of hull
       polygon simplex) around valid points we draw a convex hull. All missing
       points inside the convex hull will be used to be inpainted.

    4- Points in ocean inside convex hull:
       There are in ocean (off land) and inside the convex hull. These points
       have missing data. These points will be inpainted.

    5- Points in ocean outside convex hull:
       There are in ocean (off land) and outside the convex hull. There points
       have missing data. There points will not be inpainred.

    Boolean settings:

    - include_land_for_hull: If set to True, the valid points AND the land
      points are combined to be used for finding the convex/concave hull. This
      is used when we want to fill the gap between the data in ocean and the
      coast. So that the output data is filled upto the coast. If aet to False,
      only the valid points are used to draw a hull around them.

    - convex_hull: If set to True, the hull is the convex hull around
      targeted points. If set to False, the hull is a concave hull with an
      alpha shape.

    This function finds the followings:

    Array sizes:

    Input:
        - lon:           1xL1       float
        - lon:           1xL2       float
        - land_indices:         Nx2        int
        - data:                L1xL2      float, masked
        - include_land_for_hull   Scalar     boolean

    Output:
        -all_missing_indices_in_ocean:         Hx2     int
             Indices include points indies and outside hull, Here H = H1 + H2
        -missing_indices_in_ocean_in_hull   H1x2    int
             A part of all_missing_indices_in_ocean only inside hull
        -missing_indices_in_ocean_outside_hull  H2x2    int
             A part of all_missing_indices_in_ocean only outside hull
        -valid_indices                      Qx2     int
             Indices of points in ocean that are not missing. It does not
             include land points.
        -hull_points_coord_list         List    numpy.array
             Each element is numpy.array of size Kx2 (x, y) point coordinates
             of K points on the vertices of hull polygon

    In above:
        - Total grid points:                    L1 x L2 = N + H + Q
        - Land points:                          N
        - Ocean points:                         H + Q
        - Valid ocean points:                   Q
        - Missing ocean points:                 H = H1 + H2
        - Missing ocean points inside hull:     H1
        - Missing ocean points outside hull:    H2
    """

    # Missing points flag array
    if hasattr(data, 'mask'):
        missing_points_bool_array = numpy.copy(data.mask)
    else:
        # Some dataset does not declare missing points with mask, rather they
        # use nan.
        missing_points_bool_array = numpy.isnan(data)

    # Get indices of valid data points. Valid points do not include land points
    valid_indices_I, valid_indices_J = \
        numpy.where(numpy.logical_not(missing_points_bool_array))
    valid_indices = numpy.vstack((valid_indices_I, valid_indices_J)).T

    # Flag land points to not to be missing points
    if bool(numpy.any(numpy.isnan(land_indices))) is False:
        for i in range(land_indices.shape[0]):
            missing_points_bool_array[
                    land_indices[i, 0], land_indices[i, 1]] = False

    # All missing indices in ocean
    # NOTE: First index I are lats not lons. Second index J are lons not lats
    all_missing_indices_in_ocean_I, all_missing_indices_in_ocean_J = \
        numpy.where(missing_points_bool_array)
    all_missing_indices_in_ocean = numpy.vstack(
            (all_missing_indices_in_ocean_I, all_missing_indices_in_ocean_J)).T

    # Mesh of longitudes and latitudes
    lon_grid, lat_grid = numpy.meshgrid(lon, lat)

    # lon and lat of points where data are valid
    valid_lon = lon_grid[valid_indices_I, valid_indices_J]
    valid_lat = lat_grid[valid_indices_I, valid_indices_J]

    # lon and latitude of missing point in the ocean
    all_missing_lon_in_ocean = lon_grid[
            all_missing_indices_in_ocean[:, 0],
            all_missing_indices_in_ocean[:, 1]]
    all_missing_lat_in_ocean = lat_grid[
            all_missing_indices_in_ocean[:, 0],
            all_missing_indices_in_ocean[:, 1]]

    # Land latitudes and longitudes
    if bool(numpy.any(numpy.isnan(land_indices))) is False:
        land_lon = lon_grid[land_indices[:, 0], land_indices[:, 1]]
        land_lat = lat_grid[land_indices[:, 0], land_indices[:, 1]]
    else:
        land_lon = numpy.nan
        land_lat = numpy.nan

    # Points coordinates for valid points, missing points in ocean, and land
    valid_points_coord = numpy.c_[valid_lon, valid_lat]
    all_missing_points_in_ocean_coord = numpy.c_[
            all_missing_lon_in_ocean, all_missing_lat_in_ocean]

    if bool(numpy.any(numpy.isnan(land_indices))) is False:
        land_points_coord = numpy.c_[land_lon, land_lat]
    else:
        land_points_coord = numpy.nan

    # Determine which points should be used to determine the body of the hull
    # with. Use -l in arguments to include lands.
    if include_land_for_hull is True:
        # The Land points are also merged to valid points to find the hull
        hull_body_points_coord = numpy.vstack(
                (valid_points_coord, land_points_coord))
    else:
        # The land points are not included to the hull.
        hull_body_points_coord = valid_points_coord

    # Get the status of all missing points in ocean (In array, True means the
    # point is inside the concave/convex hull). Use -c in arguments for convex.
    if convex_hull is True:
        # Use Convex Hull
        all_missing_points_in_ocean_status_in_hull, hull_points_coord_list = \
                _find_status_of_all_missing_points_in_ocean_with_convex_hull(
                        hull_body_points_coord,
                        all_missing_points_in_ocean_coord,
                        all_missing_indices_in_ocean, lon, lat)
    else:
        # Use Concave Hull. alpha is determined by user with -a argument.
        max_alpha = _compute_max_alpha(lon, lat)
        if alpha > max_alpha:
            alpha = max_alpha * 0.9
            print("Message: alpha is changed to: %f" % alpha)
            sys.stdout.flush()

        all_missing_points_in_ocean_status_in_hull, hull_points_coord_list = \
            _find_status_of_all_missing_points_in_ocean_with_concave_hull(
                    hull_body_points_coord,
                    all_missing_points_in_ocean_coord, alpha,
                    all_missing_indices_in_ocean, lon, lat)

    # Edit "MissingPointsInOceanStatusInsideHull" to exclude points in lake
    # inside land.
    all_missing_points_in_ocean_status_in_hull = \
        _exclude_points_in_land_lake_from_points_in_ocean_in_hull(
                lon, lat,
                land_points_coord,
                all_missing_points_in_ocean_status_in_hull,
                all_missing_indices_in_ocean,
                all_missing_points_in_ocean_coord)

    # Missing Points Indices inside hull
    missing_indices_in_ocean_in_hull = \
        all_missing_indices_in_ocean[
            all_missing_points_in_ocean_status_in_hull, :]

    # Missing points indices outside hull
    missing_indices_in_ocean_outside_hull = \
        all_missing_indices_in_ocean[numpy.logical_not(
            all_missing_points_in_ocean_status_in_hull), :]

    return all_missing_indices_in_ocean, missing_indices_in_ocean_in_hull, \
        missing_indices_in_ocean_outside_hull, valid_indices, \
        hull_points_coord_list
