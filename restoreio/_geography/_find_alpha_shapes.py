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
import shapely.geometry
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay

__all__ = ['find_alpha_shapes']


# ================
# Find Convex Hull
# ================

def find_convex_hull(
        points_coord,
        num_points):
    """
    Input:
        - points_coord: A numpy array Nx2 of point coordinates

    Output:
        - A Shapely geometry Polygon of convex hull.
    """

    # Shaped Points List
    shaped_points_list = []
    for i in range(num_points):
        tuple = (points_coord[i, 0], points_coord[i, 1])
        point_dict = {
                "type": "Point",
                "coordinates": tuple
        }
        shaped_points_list.append(shapely.geometry.shape(point_dict))

    # Points Collection
    points_collection = shapely.geometry.MultiPoint(shaped_points_list)

    # get the convex hull of the points collection
    return points_collection.convex_hull


# ========
# Add Edge
# ========

def add_edge(
        edges,
        edges_points_coord,
        points_coord,
        point_index_I,
        point_index_J):
    """
    This added a line between point_index_I and point_index_J if it is not in
    the list already.

    Inputs and Outputs:
        - Edge: Set of N tuples of like (point_index_I, point_index_J)
        - edges_points_coord: List N elements where each element is a 2x2 numpy
          array of type
          numpy.array([[PointI_X, PointI_Y], [PointJ_X, PointJ_Y]])

    Inputs:
        - points_coord: Numpy array of size Nx2
        - point_index_I: An index for I-th point
        - point_index_J: An index for J-th point
    """

    if ((point_index_I, point_index_J) in edges) or \
            ((point_index_J, point_index_I) in edges):
        # This line is already an edge.
        return

    # Add (I, J) tuple to edges
    edges.add((point_index_I, point_index_J))

    # Append the coordinates of the two points that was added as an edge
    edges_points_coord.append(points_coord[[point_index_I, point_index_J], :])


# ===================
# Compute Edge Length
# ===================

def compute_edge_length(
        point_1_coord,
        point_2_coord):
    """
    Inputs:
        - point_1_coord: 1x2 numpy array
        - point_2_coord: 1x2 numpy array

    Output:
        - Distance between two points
    """
    return numpy.sqrt(sum((point_2_coord-point_1_coord)**2))


# ==========================================
# Compute Radius Of CircumCirlce Of Triangle
# ==========================================

def compute_triangle_circumcicle(triangle_vertices_coord):
    """
    Input:
        - triangle_vertices_coord: 3x2 numpy array.

    Output:
        - Radius of the cirumcircle that embeds the triangle.
    """
    length_1 = compute_edge_length(triangle_vertices_coord[0, :],
                                   triangle_vertices_coord[1, :])
    length_2 = compute_edge_length(triangle_vertices_coord[1, :],
                                   triangle_vertices_coord[2, :])
    length_3 = compute_edge_length(triangle_vertices_coord[2, :],
                                   triangle_vertices_coord[0, :])

    # Semiperimeter of triangle
    # Semiperimeter = (length_1 + length_2 + length_3) / 2.0

    # area of triangle (Heron formula)
    # area = numpy.sqrt(Semiperimeter * (Semiperimeter - length_1) * \
    #         (Semiperimeter - length_2) * (Semiperimeter - length_3))

    # Put lengths in an array in ascending order
    lengths = numpy.array([length_1, length_2, length_3])
    lengths.sort()             # descending order
    lengths = lengths[::-1]    # ascending order

    # area of triangle (Heron's stabilized formula)
    S = (lengths[2] + (lengths[1] + lengths[0])) * \
        (lengths[0] - (lengths[2] - lengths[1])) * \
        (lengths[0] + (lengths[2] - lengths[1])) * \
        (lengths[2] + (lengths[1] - lengths[0]))

    if (S < 0.0) and (S > -1e-8):
        area = 0.0
    else:
        area = 0.25 * numpy.sqrt(S)

    # Cimcumcircle radius
    if area < 1e-14:
        # lengths[0] is a very small. Assume (lengths[1] - lengths[2]) = 0
        circumcircle_radius = (lengths[1] * lengths[2]) / \
                (lengths[1] + lengths[2])
    else:
        # Use normal formula
        circumcircle_radius = (lengths[0] * lengths[1] * lengths[2]) / \
                (4.0 * area)

    return circumcircle_radius


# =================
# Find alpha Shapes
# =================

def find_alpha_shapes(points_coord, alpha):
    """
    Finds the alpha shape polygons.

    Inputs:
        -points_coord: An array of size Nx2 where each row is (x, y) coordinate
                       of one point. These points are the input data points
                       which we wantto fraw an alpha shape around them.
        - alpha:       A real number. 1/alpha is the circle radius for alpha
                       shapes.

    Outputs:
        - alpha_shape_polygon: there are two cases:
            1. If it finds one shape, it returns
               shapely.geometry.polygon.Polygon object
            2. If it finds more than one shape, it returns
               shapely.geometry.multipolygon.MultiPolygon object, which is a
               list. Each element in the list is
               shapely.geometry.polygon.Polygon object.
    """

    num_points = points_coord.shape[0]

    if num_points < 4:
        # Can not find concave hull with 3 points. Return the convex hull which
        # is is triangle.
        return find_convex_hull(points_coord, num_points)

    # Delaunay triangulations
    # triangulations = Delaunay(points_coord)
    # 2021/05/20. I changed this line to avoid error: "qhull ValueError: Input
    # points cannot be a masked array"
    triangulations = Delaunay(numpy.asarray(points_coord))

    # Initialize set of edges and list of edge points coordinates
    edges = set()
    edge_points_coord = []

    # Loop over triangles
    for triangle_vertices_indices in triangulations.vertices:

        # Get coordinates of vertices
        triangle_vertices_coord = points_coord[triangle_vertices_indices, :]

        # Get circumcircle radius of the triangle
        circumcircle_radius = compute_triangle_circumcicle(
                triangle_vertices_coord)

        # Add edges that have smaller radius than Max Radius
        max_radius = 1.0 / alpha
        if circumcircle_radius < max_radius:
            # Add all three edges of triangle. Here the outputs are "edges" and
            # "edge_points_coord". The variable "edges" is only used to find
            # whether a pair of two points are previously added to the list of
            # polygons or not. The actual output that we will use later is
            # "edge_points_coord".
            add_edge(edges, edge_points_coord, points_coord,
                     triangle_vertices_indices[0],
                     triangle_vertices_indices[1])
            add_edge(edges, edge_points_coord, points_coord,
                     triangle_vertices_indices[1],
                     triangle_vertices_indices[2])
            add_edge(edges, edge_points_coord, points_coord,
                     triangle_vertices_indices[2],
                     triangle_vertices_indices[0])

    # Using "edge_points_coord" to find their cascade union polygon object
    edge_string = shapely.geometry.MultiLineString(edge_points_coord)
    triangles = list(polygonize(edge_string))
    alpha_shape_polygon = cascaded_union(triangles)

    return alpha_shape_polygon
