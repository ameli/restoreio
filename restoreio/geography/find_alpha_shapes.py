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

__all__ = ['find_alpha_shapes']


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
