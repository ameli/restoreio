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

# ==================================
# Do not Find Land And Ocean Indices
# ==================================

def DoNotFindLandAndOceanIndices(Longitude, Latitude):
    """
    This function is as oppose to "FindLandAndOceanIndices". If the user choose not to detect any land, we treat the entire 
    domain as it is in ocean. So in this function we return a LandIdices as nan, and Ocean Indices as all available indices
    in the grid.
    """

    # Do not detect any land.
    LandIndices = numpy.nan

    # We terat all the domain as it is the ocean
    OceanIndicesList = []

    for LatitudeIndex in range(Latitude.size):
        for LongitudeIndex in range(Longitude.size):
            Tuple = (LatitudeIndex, LongitudeIndex)
            OceanIndicesList.append(Tuple)

    # Convert form list to array 
    OceanIndices = numpy.array(OceanIndicesList, dtype=int)

    return LandIndices, OceanIndices

# =======================================
# Find Land And Ocean Indices In Parallel
# =======================================

def FindLandAndOceanIndicesInParallel(map, Longitude, Latitude, PointId):
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

# ============================
# Find Land And Ocean Indices1
# ============================

def FindLandAndOceanIndices1(Longitude, Latitude):
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

# =============================
# Find Land And Ocean Indices 2
# =============================

def FindLandAndOceanIndices2(Longitude, Latitude):
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

# =============================
# Find Land And Ocean Indices 3
# =============================

def FindLandAndOceanIndices3(Longitude, Latitude):
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

# =================
# Find Alpha Shapes
# =================

def FindAlphaShapes(PointsCoordinates, Alpha):
    """
    Finds the alpha shape polygons.

    Inputs:
        -PointsCoordinates: An array of size Nx2 where each row is (x, y) coordinate of one point.
                            These points are the input data points which we wantto fraw an alpha shape around them.
        - Alpha:            A real number. 1/Alpha is the circle radius for alpha shapes.

    Outputs:
        - AlphaShapePolygon: there are two cases:
            1. If it finds one shape, it returns shapely.geometry.polygon.Polygon object
            2. If it finds more than one shape, it returns shapely.geometry.multipolygon.MultiPolygon object, which is a list.
               Each element in the list is shapely.geometry.polygon.Polygon object.
    """

    # ----------------
    # Find Convex Hull
    # ----------------

    def FindConvexHull(PointsCoordinates):
        """
        Input:
            - PointsCoordinates: A numpy array Nx2 of point coordinates

        Output:
            - A Shapely geometry Polygon of convex hull.
        """

        # Shaped Points List
        ShapedPointsList = []
        for i in range(NumPoints):
            Tuple = (PointsCoordinates[i, 0], PointsCoordinates[i, 1])
            PointDictionary = {"type":"Point", "coordinates":Tuple}
            ShapedPointsList.append(shapely.geometry.shape(PointDictionary))

        # Points Collection
        PointsCollection = shapely.geometry.MultiPoint(ShapedPointsList)

        # get the convex hull of the points collection
        return PointsCollection.convex_hull

    # --------
    # Add Edge
    # --------

    def AddEdge(Edges, EdgesPointsCoordinates, PointsCoordinates, PointIndexI, PointIndexJ):
        """
        This added a line between PointIndexI and PointIndexJ if it is not in the list already.

        Inputs and Outputs:
            - Edge: Set of N tuples of like (PointIndexI, PointIndexJ)
            - EdgesPointsCoordinates: List N elements where each element is a 2x2 numpy array of type numpy.array([[PointI_X, PointI_Y], [PointJ_X, PointJ_Y]])

        Inputs:
            - PointsCoordinates: Numpy array of size Nx2
            - PointIndexI: An index for I-th point
            - PointIndexJ: An index for J-th point
        """

        if ((PointIndexI, PointIndexJ) in Edges) or ((PointIndexJ, PointIndexI) in Edges):
            # This line is already an edge.
            return

        # Add (I, J) tuple to edges
        Edges.add((PointIndexI, PointIndexJ))

        # Append the coordinates of the two points that was added as an edge
        EdgesPointsCoordinates.append(PointsCoordinates[[PointIndexI, PointIndexJ], :])

    # -------------------
    # Compute Edge Length
    # -------------------

    def ComputeEdgeLength(Point1Coordinates, Point2Coordinates):
        """
        Inputs:
            - Point1Coordinates: 1x2 numpy array
            - Point2Coordinates: 1x2 numpy array

        Output:
            - Distance between two points
        """
        return numpy.sqrt(sum((Point2Coordinates-Point1Coordinates)**2))

    # ------------------------------------------
    # Compute Radius Of CircumCirlce Of Triangle
    # ------------------------------------------

    def ComputeRadiusOfCircumCircleOfTriangle(TriangleVerticesCoordinates):
        """
        Input:
            - TriangleVerticesCoordinates: 3x2 numpy array.

        Output:
            - Radius of the cirumcircle that embdeds the triangle.
        """
        Length1 = ComputeEdgeLength(TriangleVerticesCoordinates[0, :], TriangleVerticesCoordinates[1, :])
        Length2 = ComputeEdgeLength(TriangleVerticesCoordinates[1, :], TriangleVerticesCoordinates[2, :])
        Length3 = ComputeEdgeLength(TriangleVerticesCoordinates[2, :], TriangleVerticesCoordinates[0, :])

        # Semiperimeter of triangle
        # Semiperimeter = (Length1 + Length2 + Length3) / 2.0

        # Area of triangle (Heron formula)
        # Area = numpy.sqrt(Semiperimeter * (Semiperimeter - Length1) * (Semiperimeter - Length2) * (Semiperimeter - Length3))

        # Put lengths in an array in assending order
        Lengths = numpy.array([Length1, Length2, Length3])
        Lengths.sort()             # descending order
        Lengths = Lengths[::-1]    # ascending order

        # Area of triangle (Heron's stablized formula)
        S = (Lengths[2] + (Lengths[1] + Lengths[0])) * \
            (Lengths[0] - (Lengths[2] - Lengths[1])) * \
            (Lengths[0] + (Lengths[2] - Lengths[1])) * \
            (Lengths[2] + (Lengths[1] - Lengths[0]))

        if (S < 0.0) and (S > -1e-8):
            Area = 0.0;
        else:
            Area = 0.25 * numpy.sqrt(S)

        # Cimcumcircle radius
        if Area < 1e-14:
            # Lengths[0] is a very small number. We assume (Lengths[1] - Lengths[2]) = 0
            CircumCircleRadius = (Lengths[1] * Lengths[2]) / (Lengths[1] + Lengths[2])
        else:
            # Use normal formula
            CircumCircleRadius = (Lengths[0] * Lengths[1] * Lengths[2]) / (4.0 * Area)

        return CircumCircleRadius

    # ------------------------------------

    NumPoints = PointsCoordinates.shape[0]
    if NumPoints < 4:
        # Can not find concave hull with 3 points. Return the convex hull which is is triangle.
        return FindConvexHull(PointsCoordinates)
    
    # Delaunay Triangulations
    # Triangulations = Delaunay(PointsCoordinates)
    Triangulations = Delaunay(numpy.asarray(PointsCoordinates))  # 2021/05/20. I changed this line to avoid error: "qhull ValueError: Input points cannot be a masked array"
    
    # Initialize set of edges and list of edge points coordinates
    Edges = set()
    EdgePointsCoordinates = []

    # Loop over triangles
    for TriangleVerticesIndices in Triangulations.vertices:

        # Get coordinates of vertices
        TriangleVerticesCoordinates = PointsCoordinates[TriangleVerticesIndices, :]

        # Get circumcircle radius of the triangle
        CircumcircleRadius = ComputeRadiusOfCircumCircleOfTriangle(TriangleVerticesCoordinates)

        # Add edges that have smaller radius than Max Radius
        MaxRadius = 1.0 / Alpha
        if CircumcircleRadius < MaxRadius:
            # Add all three edges of triangle. Here the outputs are "Edges" and "EdgePointsCoordinates".
            # The variable "Edges" is only used to find wether a pair of two points are previously added to the list of
            # polygons or not. The actual output that we will use later is "EdgePointsCoordinates".
            AddEdge(Edges, EdgePointsCoordinates, PointsCoordinates, TriangleVerticesIndices[0], TriangleVerticesIndices[1])
            AddEdge(Edges, EdgePointsCoordinates, PointsCoordinates, TriangleVerticesIndices[1], TriangleVerticesIndices[2])
            AddEdge(Edges, EdgePointsCoordinates, PointsCoordinates, TriangleVerticesIndices[2], TriangleVerticesIndices[0])

    # Using "EdgePointsCoordinates" to find their cascade union polygon object
    EdgeString = shapely.geometry.MultiLineString(EdgePointsCoordinates)
    Triangles = list(polygonize(EdgeString))
    AlphaShapePolygon = cascaded_union(Triangles)

    return AlphaShapePolygon

# ===================
# Locate Missing Data
# ===================

def LocateMissingData( \
        Longitude, \
        Latitude, \
        LandIndices, \
        Data, \
        IncludeLandForHull, \
        UseConvexHull, \
        Alpha):
    """
    All points in grid are divided into these groups:

    1- Land points: (Nx2 array)
        This is not defined by this function. Rather it is computer earlier before calling this function.
        The rest of the points are the points in the ocean.

    2- Valid points: (Mx2 array)
        The valid points are merelly defined on the valid points of the DATA, and they do not include the land area.
        All valid points are in the ocean.

        Note: The forth argument "Data" can be any of North velocities and East velocities. This is used to find masked points.
    
    - Convex Hull: (Kx2 array of x-y coordinates where K is the number of hull polygon simplex)
        Aroud valid points we draw a convex hull. All missing points inside the convex hull will be used to be inpainted.

    - Points in ocean inside convex hull: 
        There are in ocean (off land) and inside the convex hull. These points have missing data. These points will be inpainted.

    - Points in ocean outside convex hull
        There are in ocean (off land) and outside the convex hull. There points have missing data. There points will not be inpainred.

    Boolean settings:

    - IncludeLandForHull: If set to True, the valid points AND the land points are combined to be used for finding the convex/concave hull. This is used when
        we want to fill the gap between the data in ocean and the coast. So that the output data is filled upto the coast. If aet to False, only the valid points 
        are used to draw a hull around them.

    - UseConvexHull: If set to True, the hull is the convex hull around targeted points. If set to False, the hull is a concave hull with an alpha shape.


    This function finds the followngs:

    Array sizes:

    Input:
        - Longitude:           1xL1       float
        - Longitude:           1xL2       float
        - LandIndices:         Nx2        int
        - Data:                L1xL2      float, masked
        - IncludeLandForHull   Scalar     boolean

    Output:
        -AllMissingIndicesInOcean:         Hx2     int           Indices include points indies and outside hull, Here H = H1 + H2
        -MissingIndicesInOceanInsideHull   H1x2    int           A part of AllMissingIndicesInOcean only inside hull
        -MissingIndicesInOceanOutsideHull  H2x2    int           A part of AllMissingIndicesInOcean only outside hull
        -ValidIndices                      Qx2     int           Indices of points in ocean that are not missing. It does not include land points.
        -HullPointsCoordinatesList         List    numpy.array   Each element is numpy.array of size Kx2 (x, y) point coordinates of K points on the vertices of hull polygon

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
    if hasattr(Data, 'mask'):
        MissingPointsBooleanArray = numpy.copy(Data.mask)
    else:
        # Some dataset does not declare missing points with mask, rather they use nan.
        MissingPointsBooleanArray = numpy.isnan(Data)
    
    # Get indices of valid data points. The valid points do not include land points
    ValidIndices_I, ValidIndices_J = numpy.where(MissingPointsBooleanArray == False)
    ValidIndices = numpy.vstack((ValidIndices_I, ValidIndices_J)).T

    # Flag land points to not to be missing points
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):
            MissingPointsBooleanArray[LandIndices[i, 0], LandIndices[i, 1]] = False

    # All missing indices in ocean
    # NOTE: First index I are Latitudes not Longitudes. Second index J are Longitudes not Latitudes
    AllMissingIndicesInOcean_I, AllMissingIndicesInOcean_J = numpy.where(MissingPointsBooleanArray == True)
    AllMissingIndicesInOcean = numpy.vstack((AllMissingIndicesInOcean_I, AllMissingIndicesInOcean_J)).T

    # Mesh of longitudes and latitudes
    LongitudesGrid, LatitudesGrid = numpy.meshgrid(Longitude, Latitude)

    # Longitude and Latitude of points where data are valid
    ValidLongitudes = LongitudesGrid[ValidIndices_I, ValidIndices_J]
    ValidLatitudes = LatitudesGrid[ValidIndices_I, ValidIndices_J]

    # Longitude and latitude of missing point in the ocean
    AllMissingLongitudesInOcean = LongitudesGrid[AllMissingIndicesInOcean[:, 0], AllMissingIndicesInOcean[:, 1]]
    AllMissingLatitudesInOcean = LatitudesGrid[AllMissingIndicesInOcean[:, 0], AllMissingIndicesInOcean[:, 1]]

    # Land latitudes and longitudes
    if numpy.any(numpy.isnan(LandIndices)) == False:
        LandLongitudes = LongitudesGrid[LandIndices[:, 0], LandIndices[:, 1]]
        LandLatitudes = LatitudesGrid[LandIndices[:, 0], LandIndices[:, 1]]
    else:
        LandLongitudes = numpy.nan
        LandLatitudes = numpy.nan

    # Points coordinates for valid points, missing points in ocean, and land
    ValidPointsCoordinates = numpy.c_[ValidLongitudes, ValidLatitudes]
    AllMissingPointsInOceanCoordinates = numpy.c_[AllMissingLongitudesInOcean, AllMissingLatitudesInOcean]

    if numpy.any(numpy.isnan(LandIndices)) == False:
        LandPointsCoordinates = numpy.c_[LandLongitudes, LandLatitudes]
    else:
        LandPointsCoordinates = numpy.nan

    # -----------------
    # Compute Max Alpha
    # -----------------

    def ComputeMaxAlpha():
        """
        Computes the smallest possible alpha based on the smallest cell. The smallest cell (element) is a right angle
        formed between tree adjacent points on the grid with right edges delta_longitude and delta_latitude
        """

        Diff_Longitude = numpy.diff(Longitude, 1)
        Diff_Latitude = numpy.diff(Latitude, 1)

        Min_Delta_Longitude = numpy.min(Diff_Longitude)
        Min_Delta_Latitude = numpy.min(Diff_Latitude)

        Min_CircumcircleRadius = numpy.sqrt(Min_Delta_Longitude**2 + Min_Delta_Latitude**2) / 2.0

        MaxAlpha = 1.0 / Min_CircumcircleRadius

        return MaxAlpha

    # ------------------------------------------------------------
    # Find Status Of All Missing Points In Ocean With Concave Hull
    # ------------------------------------------------------------

    def FindStatusOfAllMissingPointsInOceanWithConcaveHull( \
            HullBodyPointsCoordinates, \
            AllMissingPointsInOceanCoordinates, \
            Alpha, \
            AllMissingIndicesInOcean, \
            Longitude, \
            Latitude):
        """
        All points in ocean can be separated into valid points and missing points. The two arguements fo this functions are valid points (plus
        maybe the land points) and also the missing points in ocean.

        Input:
            - HullBodyPointsCoordinates: Nx2 numpy array. This is the coordinate sof points that we will draw a concave hull around it.
              This can be either just the ValidPointsCoordinates, or the combination of ValidPointsCoordinates and the LandPointsCoordinates.

            - AllMissingPointsInOceanCoordinates: Mx2 array. This is the coordinate of all missing points in ocean.

            -Alpha: The circumcircle radius = 1 / Alpha. The larger alpha means the alpha shape attaches to the points more, where as smaller
             alpha mean the alpha shape is more tends to be the convex hull.

        Output:
            - AllMissingPointsInOceanStatusInsideHull: A boolean array of size Mx2 (the same size as AllMissingPointsInOceanCoordinates)
            If a point is inside the concave hull the element of this array is flagged as True. Points outside are flagged as False.

            - HullPointsCoordinatesList: A list that each member of tha list are the coordinates of one pf the separate concave hulls. There might be
            many separate concave hulls. For each, the corresponding member of the list is a Qx2 numpy array. Q is the number of points on the exterior
            (or boundary) or the polygon and it varies for each of the polygons.
        """

        # Find the concave hull of points
        ConcaveHullPolygon = FindAlphaShapes(HullBodyPointsCoordinates, Alpha)

        # detect the number of shapes
        ConcaveHullPolygonsList = []
        NumberOfShapes = 0
        if type(ConcaveHullPolygon) is shapely.geometry.polygon.Polygon:

            # Only one shape
            NumberOfShapes = 1
            ConcaveHullPolygonsList.append(ConcaveHullPolygon)

        elif type(ConcaveHullPolygon) is shapely.geometry.multipolygon.MultiPolygon:

            # Multi shapes
            NumberOfShapes = len(ConcaveHullPolygon)
            for i in range(NumberOfShapes):
                ConcaveHullPolygonsList.append(ConcaveHullPolygon[i])

        else:
            raise RuntimeError("Invalid polygon type: %s."%type(ConcaveHullPolygon))

        # Allocate output
        NumberOfAllMissingPointsInOcean = AllMissingPointsInOceanCoordinates.shape[0]
        AllMissingPointsInOceanStatusInsideHull = numpy.zeros(NumberOfAllMissingPointsInOcean, dtype=bool)

        # Find the AllMissingPointsInOceanStatusInsideHull
        for j in range(NumberOfShapes):
            # Iterate over all False points
            for i in range(NumberOfAllMissingPointsInOcean):
                # Only check those points that are not yet seen to be inside one of the shape polygons
                if(AllMissingPointsInOceanStatusInsideHull[i] == False):

                    PointCoordinates = AllMissingPointsInOceanCoordinates[i, :]
                    PointIndex = AllMissingIndicesInOcean[i, :]

                    # Get Delta_Longitude (Note: Longitude is the second index of points)
                    if PointIndex[1] == Longitude.size - 1:
                        Delta_Longitude = numpy.abs(Longitude[-1] - Longitude[-2])
                    elif PointIndex[1] < Longitude.size - 1:
                        Delta_Longitude = numpy.abs(Longitude[PointIndex[1]+1] - Longitude[PointIndex[1]])
                    else:
                        raise RuntimeError("Wrong Longitude index: %d, Longitude size: %d"%(PointIndex[1], Longitude.size))

                    # Get Delta_Latitude (Note: Latitude is the first index of points)
                    if PointIndex[0] == Latitude.size - 1:
                        Delta_Latitude = numpy.abs(Latitude[-1] - Latitude[-2])
                    elif PointIndex[0] < Latitude.size - 1:
                        Delta_Latitude = numpy.abs(Latitude[PointIndex[0]+1] - Latitude[PointIndex[0]])
                    else:
                        raise RuntimeError("Wrong Latitude index: %d, Latitude size: %d"%(PointIndex[0], Latitude.size))

                    # Ratio of the element size that which we check the auxilliary points
                    Delta_Ratio = 0.05

                    # Try the point itself:
                    GeometryPointObject = shapely.geometry.Point(PointCoordinates[0], PointCoordinates[1])
                    PointStatusInOceanInsideHull = ConcaveHullPolygonsList[j].contains(GeometryPointObject)
                    if PointStatusInOceanInsideHull == True:
                        AllMissingPointsInOceanStatusInsideHull[i] = True
                        continue

                    # Try point above
                    PointCoordinatesAbove = numpy.copy(PointCoordinates)
                    PointCoordinatesAbove[1] += Delta_Latitude * Delta_Ratio
                    GeometryPointObject = shapely.geometry.Point(PointCoordinatesAbove[0], PointCoordinatesAbove[1])
                    PointStatusInOceanInsideHull = ConcaveHullPolygonsList[j].contains(GeometryPointObject)
                    if PointStatusInOceanInsideHull == True:
                        AllMissingPointsInOceanStatusInsideHull[i] = True
                        continue

                    # Try point below
                    PointCoordinatesBelow = numpy.copy(PointCoordinates)
                    PointCoordinatesBelow[1] -= Delta_Latitude * Delta_Ratio
                    GeometryPointObject = shapely.geometry.Point(PointCoordinatesBelow[0], PointCoordinatesBelow[1])
                    PointStatusInOceanInsideHull = ConcaveHullPolygonsList[j].contains(GeometryPointObject)
                    if PointStatusInOceanInsideHull == True:
                        AllMissingPointsInOceanStatusInsideHull[i] = True
                        continue

                    # Try point Left
                    PointCoordinatesLeft = numpy.copy(PointCoordinates)
                    PointCoordinatesLeft[0] -= Delta_Longitude * Delta_Ratio
                    GeometryPointObject = shapely.geometry.Point(PointCoordinatesLeft[0], PointCoordinatesLeft[1])
                    PointStatusInOceanInsideHull = ConcaveHullPolygonsList[j].contains(GeometryPointObject)
                    if PointStatusInOceanInsideHull == True:
                        AllMissingPointsInOceanStatusInsideHull[i] = True
                        continue

                    # Try point Right
                    PointCoordinatesRight = numpy.copy(PointCoordinates)
                    PointCoordinatesRight[0] += Delta_Longitude * Delta_Ratio
                    GeometryPointObject = shapely.geometry.Point(PointCoordinatesRight[0], PointCoordinatesRight[1])
                    PointStatusInOceanInsideHull = ConcaveHullPolygonsList[j].contains(GeometryPointObject)
                    if PointStatusInOceanInsideHull == True:
                        AllMissingPointsInOceanStatusInsideHull[i] = True
                        continue

        # Find HullPointsCoordinatesList
        HullPointsCoordinatesList = [None] * NumberOfShapes
        for i in range(NumberOfShapes):
            OneHullPointsCoordinates_XY = ConcaveHullPolygonsList[i].exterior.xy
            HullPointsCoordinatesList[i] = numpy.array(OneHullPointsCoordinates_XY).T

        return AllMissingPointsInOceanStatusInsideHull, HullPointsCoordinatesList

    # -----------------------------------------------------------
    # Find Status Of All Missing Points In Ocean With Convex Hull
    # -----------------------------------------------------------

    def FindStatusOfAllMissingPointsInOceanWithConvexHull( \
            HullBodyPointsCoordinates, \
            AllMissingPointsInOceanCoordinates, \
            AllMissingIndicesInOcean, \
            Longitude, \
            Latitude):
        """
        Al points in ocean can be separated into valid points and missing points. The two arguments of this function are valid
        points and missing points.

        Input:
            - HullBodyPointsCoordinates: Nx2 numpy array. This is the coordinate of the points that we will draw a convex hull around it.
              This is usually the ValidPointsCoordinates.

            - AllMissingPointsInOceanCoordinates: Mx2 numpy array. This is the coordinaes of all missing points in ocean.

        Output:
            - AllMissingPointsInOceanStatusInsideHull: A boolean array of size 1xM (the same size as AllMissingPointsInOceanCoordinates)
              If a point is inside the convex hull the element on this array is flagged as True. Points outside are flagged as False.

            - HullPointsCoordinatesList: A list that has one member. The only member (HullPointsCoordinatesList[0]) is a numpy array of size Qx2
            which Q is the number of convex hull exterior (boundary) points. These are the coordinates of the polygon that wraps the hull.
        """

        # Find the convex hull around data
        HullPolygon = ConvexHull(HullBodyPointsCoordinates)
        HullPointsCoordinates = HullBodyPointsCoordinates[HullPolygon.vertices, :]
        HullPointsCoordinatesList = [HullPointsCoordinates]

        # Create path from hull points
        HullPath = path.Path(HullPointsCoordinates)

        NumberOfAllMissingPoints = AllMissingPointsInOceanCoordinates.shape[0]
        AllMissingPointsInOceanStatusInsideHull = numpy.zeros(NumberOfAllMissingPoints, dtype=bool)
        Delta_Ratio = 0.05

        # Check wether missing points are inside the hull (True:Inside, False:Outside)
        for i in range(NumberOfAllMissingPoints):
            PointCoordinates = AllMissingPointsInOceanCoordinates[i, :]
            PointIndex = AllMissingIndicesInOcean[i, :]
            
            # Get Delta_Longitude (Note: Longitude is the second index of points)
            if PointIndex[1] == Longitude.size - 1:
                Delta_Longitude = numpy.abs(Longitude[-1] - Longitude[-2])
            elif PointIndex[1] < Longitude.size - 1:
                Delta_Longitude = numpy.abs(Longitude[PointIndex[1]+1] - Longitude[PointIndex[1]])
            else:
                raise RuntimeError("Wrong Longitude index: %d, Longitude size: %d"%(PointIndex[1], Longitude.size))

            # Get Delta_Latitude (Note: Latitude is the first index of points)
            if PointIndex[0] == Latitude.size - 1:
                Delta_Latitude = numpy.abs(Latitude[-1] - Latitude[-2])
            elif PointIndex[0] < Latitude.size - 1:
                Delta_Latitude = numpy.abs(Latitude[PointIndex[0]+1] - Latitude[PointIndex[0]])
            else:
                raise RuntimeError("Wrong Latitude index: %d, Latitude size: %d"%(PointIndex[0], Latitude.size))

            # Try the point itself:
            if HullPath.contains_point(PointCoordinates) == True:
                AllMissingPointsInOceanStatusInsideHull[i] = True
                continue

            # Try point above
            PointCoordinatesAbove = numpy.copy(PointCoordinates)
            PointCoordinatesAbove[1] += Delta_Latitude * Delta_Ratio
            if HullPath.contains_point(PointCoordinatesAbove) == True:
                AllMissingPointsInOceanStatusInsideHull[i] = True
                continue

            # Try point below
            PointCoordinatesBelow = numpy.copy(PointCoordinates)
            PointCoordinatesBelow[1] -= Delta_Latitude * Delta_Ratio
            if HullPath.contains_point(PointCoordinatesBelow) == True:
                AllMissingPointsInOceanStatusInsideHull[i] = True
                continue

            # Try point Left
            PointCoordinatesLeft = numpy.copy(PointCoordinates)
            PointCoordinatesLeft[0] -= Delta_Longitude * Delta_Ratio
            if HullPath.contains_point(PointCoordinatesLeft) == True:
                AllMissingPointsInOceanStatusInsideHull[i] = True
                continue

            # Try point Right
            PointCoordinatesRight = numpy.copy(PointCoordinates)
            PointCoordinatesRight[0] += Delta_Longitude * Delta_Ratio
            if HullPath.contains_point(PointCoordinatesRight) == True:
                AllMissingPointsInOceanStatusInsideHull[i] = True
                continue

        return AllMissingPointsInOceanStatusInsideHull, HullPointsCoordinatesList

    # ------------------------------------------------------------
    # Exclude Points In Land Lake From Points In Ocean Inside Hull
    # ------------------------------------------------------------

    def ExcludePointsInLandLakeFromPointsInOceanInsideHull( \
            LandPointsCoordinates, \
            AllMissingPointsInOceanStatusInsideHull, \
            AllMissingIndicesInOcean, \
            AllMissingPointsInOceanCoordinates):
        """
        This functions removes some rows from "MissingIndicesInOceanInsideHull" and adds them to "MissingIndicesInOceanOutsideHull".
        The reason is that some points might belong to a lake on the land, but when we include land to the hull points, all points
        including the lakes are also considered as missing points inside hull. But these points are not in ocean.

        To detect these points, we draw another alpha shape, ONLY around the land and check which of the points inside hull are
        sorounded by the land's alpha shape. If a point is found, we remove is from the points iniside hull list and add them to 
        the points outside the hull.
        """

        # Do nothing if there is no land in the area.
        if numpy.any(numpy.isnan(LandPointsCoordinates)) == True:
            return AllMissingPointsInOceanStatusInsideHull

        # True means the points that are identified to be inside the hull. We get their ID with respect to this array.
        IdsInHull = numpy.where(AllMissingPointsInOceanStatusInsideHull == True)[0]

        # Getting the Indices of missing points inside hull
        MissingIndicesInOceanInsideHull = AllMissingIndicesInOcean[IdsInHull, :]

        # Get the coordinate of missing points inside the hull
        MissingPointsInOceanInsideHullCoordinates = AllMissingPointsInOceanCoordinates[IdsInHull, :]

        # Use a large alpha to create an alpha shape that closely follows the land points
        Alpha = 60
        MaxAlpha = ComputeMaxAlpha()
        if Alpha > MaxAlpha:
            Alpha = MaxAlpha * 0.9

        MissingPointsInOceanInsideHullStatusInLand, LandPointsCoordinatesList = \
                FindStatusOfAllMissingPointsInOceanWithConcaveHull( \
                    LandPointsCoordinates, \
                    MissingPointsInOceanInsideHullCoordinates, \
                    Alpha, \
                    MissingIndicesInOceanInsideHull, \
                    Longitude, \
                    Latitude)

        # Edit original array with the newer status. True means the points are inside land.
        for i in range(IdsInHull.size):
            PointIsInsideLand = MissingPointsInOceanInsideHullStatusInLand[i]

            # If the point IS in land, edit original array and make it to be outside of hull
            if PointIsInsideLand == True:
                PointId = IdsInHull[i]
                AllMissingPointsInOceanStatusInsideHull[PointId] = False

        return AllMissingPointsInOceanStatusInsideHull

    # ------------------------------

    # Determine which points should be used to determine the body of the hull with. Use -l in arguments to include lands.
    if IncludeLandForHull == True:
        # The Land points are also merged to valid points to find the hull
        HullBodyPointsCoordinates = numpy.vstack((ValidPointsCoordinates, LandPointsCoordinates))
    else:
        # The land points are not included to the hull.
        HullBodyPointsCoordinates = ValidPointsCoordinates

    # Get the status of all missing points in ocean (In array, True means the point is inside the concave/convex hull). Use -c in arguments to use convex.
    if UseConvexHull == True:
        # Use Convex Hull
        AllMissingPointsInOceanStatusInsideHull, HullPointsCoordinatesList = \
                FindStatusOfAllMissingPointsInOceanWithConvexHull( 
                        HullBodyPointsCoordinates, \
                        AllMissingPointsInOceanCoordinates, \
                        AllMissingIndicesInOcean, \
                        Longitude, \
                        Latitude)
    else:
        # Use Concave Hull. Alpha is detemined by user with -a option in arguments.
        MaxAlpha = ComputeMaxAlpha()
        if Alpha > MaxAlpha:
            Alpha = MaxAlpha * 0.9
            print("Message: Alpha is changed to: %f"%Alpha)
            sys.stdout.flush()

        AllMissingPointsInOceanStatusInsideHull, HullPointsCoordinatesList = \
                FindStatusOfAllMissingPointsInOceanWithConcaveHull( \
                HullBodyPointsCoordinates, \
                AllMissingPointsInOceanCoordinates, \
                Alpha, \
                AllMissingIndicesInOcean, \
                Longitude, \
                Latitude)

    # Edit "MissingPointsInOceanStatusInsideHull" to exclude points in lake inside land.
    AllMissingPointsInOceanStatusInsideHull = ExcludePointsInLandLakeFromPointsInOceanInsideHull( \
            LandPointsCoordinates, \
            AllMissingPointsInOceanStatusInsideHull, \
            AllMissingIndicesInOcean, \
            AllMissingPointsInOceanCoordinates)

    # Missing Points Indices inside hull
    MissingIndicesInOceanInsideHull = AllMissingIndicesInOcean[AllMissingPointsInOceanStatusInsideHull, :]

    # Missing points indices outside hull
    MissingIndicesInOceanOutsideHull = AllMissingIndicesInOcean[numpy.logical_not(AllMissingPointsInOceanStatusInsideHull), :]

    return AllMissingIndicesInOcean, MissingIndicesInOceanInsideHull, MissingIndicesInOceanOutsideHull, ValidIndices, HullPointsCoordinatesList

# ======================
# Create Mask Info Array
# ======================

def CreateMaskInfo( \
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
