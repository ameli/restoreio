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

# 2021/05/20. I added this line fix the error: KeyError: 'PROJ_LIB'
import os
PROJ_LIB = '/opt/miniconda3/share/proj'
if not os.path.isdir(PROJ_LIB):
    raise FileNotFoundError('The directory %s does not exists.' % PROJ_LIB)
os.environ['PROJ_LIB'] = PROJ_LIB

import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib 
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Change font family
plt.rc('font', family='serif')

# ===============================
# Plot Color And Grayscale Images
# ===============================

def PlotColorAndGrayscaleImages( \
        GrayscaleImage_U, \
        GrayScaleImage_V, \
        ColorImage):
    """
    Plots the followings:
        - Grayscale image for east velocity U
        - Grayscale image for north velocity V
        - Color image with depths U, V and zero.
    """

    fig = plt.subplots(figsize=(19, 6))

    # Grayscale U
    plt.subplot(131)
    plt.imshow(GrayScaleImage_U, cmap='gray', interpolation='nearest', origin='lower')
    plt.title("8-bit grayscale image of East velocity")

    # Grayscale V
    plt.subplot(132)
    plt.imshow(GrayScaleImage_V, cmap='gray', interpolation='nearest', origin='lower')
    plt.title("8-bit grayscale image of north velocity")

    # Color image
    plt.subplot(133)
    plt.imshow(ColorImage, interpolation='nearest', origin='lower')
    plt.title('8-bit 3-channel color image of velocity vector')

    plt.savefig('ColorImage.png', transparent=True)
    print("ColorImage.png saved to disk.")
    plt.show()

# ============
# Plot Results
# ============

def PlotResults( \
        Longitude, \
        Latitude, \
        U_Original, \
        V_Original, \
        U_Inpainted, \
        V_Inpainted, \
        AllMissingIndices, \
        MissingIndicesInsideHull, \
        MissingIndicesOutsideHull, \
        ValidIndices, \
        LandIndices, \
        HullPointsCoordinatesList):
    """
    This function is called from the main() function, but is commented. To plot, uncomment this function in main(). You may
    disable iteration through all TimeIndex, and only plot for one TimeIndex inside the main().

    Note: Inside this function, there is a nested function "DrawMap()", which calls Basemap. If the  attirubte "resolution" of the 
    basemap is set to 'f' (meaning full resolution), it takes alot time to generate the image. For faster rendering, set
    the resolution to 'i'.
    
    It plots 3 figures:

    Figure 1: 
        Axes[0]: Plot of all valid points and all issing points.
        Axes[1]: Plot of all valid points, missing points inside the convex hull, and missing points outside the convex hull.

    Figure 2:
        Axes[0]: Plot of original east velocity.
        Axes[1]: Plot of restored east velocity.

    Figure 3:
        Axes[0]: Plot of original north velocity.
        Axes[1]: Plot of restored north velocity.
    """
 
    # Mesh grid
    LongitudesGrid, LatitudesGrid = numpy.meshgrid(Longitude, Latitude)

    # All Missing points coordinates
    AllMissingLongitudes = LongitudesGrid[AllMissingIndices[:, 0], AllMissingIndices[:, 1]]
    AllMissingLatitudes = LatitudesGrid[AllMissingIndices[:, 0], AllMissingIndices[:, 1]]
    AllMissingPointsCoordinates = numpy.vstack((AllMissingLongitudes, AllMissingLatitudes)).T

    # Missing points coordinates inside hull
    MissingLongitudesInsideHull = LongitudesGrid[MissingIndicesInsideHull[:, 0], MissingIndicesInsideHull[:, 1]]
    MissingLatitudesInsideHull = LatitudesGrid[MissingIndicesInsideHull[:, 0], MissingIndicesInsideHull[:, 1]]
    MissingPointsCoordinatesInsideHull = numpy.vstack((MissingLongitudesInsideHull, MissingLatitudesInsideHull)).T

    # Missing points coordinates outside hull
    MissingLongitudesOutsideHull = LongitudesGrid[MissingIndicesOutsideHull[:, 0], MissingIndicesOutsideHull[:, 1]]
    MissingLatitudesOutsideHull = LatitudesGrid[MissingIndicesOutsideHull[:, 0], MissingIndicesOutsideHull[:, 1]]
    MissingPointsCoordinatesOutsideHull = numpy.vstack((MissingLongitudesOutsideHull, MissingLatitudesOutsideHull)).T

    # Valid points coordinates
    ValidLongitudes = LongitudesGrid[ValidIndices[:, 0], ValidIndices[:, 1]]
    ValidLatitudes = LatitudesGrid[ValidIndices[:, 0], ValidIndices[:, 1]]
    ValidPointsCoordinates = numpy.c_[ValidLongitudes, ValidLatitudes]

    # Land Point Coordinates
    if numpy.any(numpy.isnan(LandIndices)) == False:
        LandLongitudes = LongitudesGrid[LandIndices[:, 0], LandIndices[:, 1]]
        LandLatitudes = LatitudesGrid[LandIndices[:, 0], LandIndices[:, 1]]
        LandPointCoordinates = numpy.c_[LandLongitudes, LandLatitudes]
    else:
        LandPointCoordinates = numpy.nan

    # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
    Percent = 0.05   # For Monterey Dtaaset
    # Percent = 0.1     # For Martha Dataet
    LongitudeOffset = Percent * numpy.abs(Longitude[-1] - Longitude[0])
    LatitudeOffset = Percent * numpy.abs(Latitude[-1] - Latitude[0])

    MinLongitude = numpy.min(Longitude)
    MinLongitudeWithOffset = MinLongitude - LongitudeOffset
    MidLongitude = numpy.mean(Longitude)
    MaxLongitude = numpy.max(Longitude)
    MaxLongitudeWithOffset = MaxLongitude + LongitudeOffset
    MinLatitude = numpy.min(Latitude)
    MinLatitudeWithOffset = MinLatitude - LatitudeOffset
    MidLatitude = numpy.mean(Latitude)
    MaxLatitude = numpy.max(Latitude)
    MaxLatitudeWithOffset = MaxLatitude + LatitudeOffset

    # --------
    # Draw map
    # --------

    def DrawMap(axis):

        # Basemap (set resolution to 'i' for faster rasterization and 'f' for full resolution but very slow.)
        map = Basemap( \
                ax = axis, \
                projection = 'aeqd', \
                llcrnrlon=MinLongitudeWithOffset, \
                llcrnrlat=MinLatitudeWithOffset, \
                urcrnrlon=MaxLongitudeWithOffset, \
                urcrnrlat=MaxLatitudeWithOffset, \
                area_thresh = 0.1, \
                lon_0 = MidLongitude, \
                lat_0 = MidLatitude, \
                resolution='i')

        # Map features
        map.drawcoastlines()
        # map.drawstates()
        # map.drawcountries()
        # map.drawcounties()
        map.drawlsmask(land_color='Linen', ocean_color='#C7DCEF', lakes=True, zorder=-2)
        # map.fillcontinents(color='red', lake_color='white', zorder=0)
        map.fillcontinents(color='moccasin', zorder=-1)

        # map.bluemarble()
        # map.shadedrelief()
        # map.etopo()

        # Latitude and Longitude lines
        LongitudeLines = numpy.linspace(numpy.min(Longitude), numpy.max(Longitude), 2)
        LatitudeLines = numpy.linspace(numpy.min(Latitude), numpy.max(Latitude), 2)
        map.drawparallels(LatitudeLines, labels=[1, 0, 0, 0], fontsize=10)
        map.drawmeridians(LongitudeLines, labels=[0, 0, 0, 1], fontsize=10)

        return map

    # ---------------------
    # Fig 1: Missing points
    # ---------------------

    def Plot1():
        """
        Missing points
        """

        fig_1, axes_1 = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        axes_1[0].set_aspect('equal')
        axes_1[1].set_aspect('equal')
        map_11 = DrawMap(axes_1[0])
        map_12 = DrawMap(axes_1[1])

        # Draw Mapscale
        # Index = int(Latitude.size / 4)
        # x0, y0 = map_11(Longitude[0], Latitude[0])
        # x1, y1 = map_11(Longitude[Index], Latitude[0])
        # Distance = (x1 - x0) / 1000 # Length of scale in Km
        Distance = 40 # For Monterey Dataset
        # Distance = 5 # For Martha Dataset
        map_11.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')
        map_12.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')

        # Get Map coordinates (Valid points, missing points inside and outside hull, land points)
        ValidPointsCoordinates_X, ValidPointsCoordinates_Y = map_11(ValidPointsCoordinates[:, 0], ValidPointsCoordinates[:, 1])
        AllMissingPointsCoordinates_X, AllMissingPointsCoordinates_Y = map_11(AllMissingPointsCoordinates[:, 0], AllMissingPointsCoordinates[:, 1])
        MissingPointsCoordinatesInsideHull_X, MissingPointsCoordinatesInsideHull_Y = map_11(MissingPointsCoordinatesInsideHull[:, 0], MissingPointsCoordinatesInsideHull[:, 1])
        MissingPointsCoordinatesOutsideHull_X, MissingPointsCoordinatesOutsideHull_Y = map_11(MissingPointsCoordinatesOutsideHull[:, 0], MissingPointsCoordinatesOutsideHull[:, 1])
        if numpy.any(numpy.isnan(LandPointCoordinates)) == False:
            LandPointsCoordinates_X, LandPointsCoordinates_Y = map_11(LandPointCoordinates[:, 0], LandPointCoordinates[:, 1])

        # Plot All missing points
        MarkerSize = 4
        map_11.plot(ValidPointsCoordinates_X, ValidPointsCoordinates_Y, 'o', markerfacecolor='lightgreen', markeredgecolor='lightgreen', markersize=MarkerSize)
        # map_11.plot(ValidPointsCoordinates_X, ValidPointsCoordinates_Y, 'o', markerfacecolor='lightgreen', markersize=MarkerSize)
        map_11.plot(AllMissingPointsCoordinates_X, AllMissingPointsCoordinates_Y, 'o', markerfacecolor='red', markeredgecolor='red', markersize=MarkerSize)
        # map_11.plot(AllMissingPointsCoordinates_X, AllMissingPointsCoordinates_Y, 'o', markerfacecolor='red', markersize=MarkerSize)
        if numpy.any(numpy.isnan(LandPointCoordinates)) == False:
            map_11.plot(LandPointsCoordinates_X, LandPointsCoordinates_Y, 'o', markerfacecolor='red', markeredgecolor='red', markersize=MarkerSize)
            # map_11.plot(LandPointsCoordinates_X, LandPointsCoordinates_Y, 'o', markerfacecolor='red', markersize=MarkerSize)
        plt.suptitle('All unavailbale points')

        # Plot all hulls boundary polygons
        NumberOfHullPolygons = len(HullPointsCoordinatesList)
        HullPolygons = [None] * NumberOfHullPolygons
        for i in range(NumberOfHullPolygons):
            HullPoints_X, HullPoints_Y = map_11(HullPointsCoordinatesList[i][:, 0], HullPointsCoordinatesList[i][:, 1])
            HullPoints_XY = numpy.vstack((HullPoints_X, HullPoints_Y)).T.tolist()
            # HullPolygons[i] = Polygon(HullPoints_XY, facecolor='lightgoldenrodyellow', alpha=0.6, closed=True, linewidth=1)
            HullPolygons[i] = Polygon(HullPoints_XY, facecolor='honeydew', edgecolor='none', alpha=0.6, closed=True, linewidth=2)
            plt.gca().add_patch(HullPolygons[i])

        # Plot Hull and missing inside/outside the hull, and land points
        map_12.plot(ValidPointsCoordinates_X, ValidPointsCoordinates_Y, 'o', markerfacecolor='lightgreen', markeredgecolor='lightgreen', markersize=MarkerSize)
        # map_12.plot(ValidPointsCoordinates_X, ValidPointsCoordinates_Y, 'o', markerfacecolor='lightgreen', markersize=MarkerSize)
        map_12.plot(MissingPointsCoordinatesInsideHull_X, MissingPointsCoordinatesInsideHull_Y, 'o', markerfacecolor='red', markeredgecolor='red', markersize=MarkerSize)
        # map_12.plot(MissingPointsCoordinatesInsideHull_X, MissingPointsCoordinatesInsideHull_Y, 'o', markerfacecolor='red', markersize=MarkerSize)
        map_12.plot(MissingPointsCoordinatesOutsideHull_X, MissingPointsCoordinatesOutsideHull_Y, 'o', markerfacecolor='royalblue', markeredgecolor='royalblue', markersize=MarkerSize)
        # map_12.plot(MissingPointsCoordinatesOutsideHull_X, MissingPointsCoordinatesOutsideHull_Y, 'o', markerfacecolor='royalblue', markersize=MarkerSize)
        if numpy.any(numpy.isnan(LandPointCoordinates)) == False:
            map_12.plot(LandPointsCoordinates_X, LandPointsCoordinates_Y, 'o', markerfacecolor='sandybrown', markeredgecolor='sandybrown', markersize=MarkerSize)
            # map_12.plot(LandPointsCoordinates_X, LandPointsCoordinates_Y, 'o', markerfacecolor='sandybrown', markersize=MarkerSize)
        plt.suptitle('Missing data inside convex hull')

        # another hull polyon on the top layer of all plots without facecolor.
        HullPolygons2 = [None] * NumberOfHullPolygons
        for i in range(NumberOfHullPolygons):
            HullPoints_X, HullPoints_Y = map_11(HullPointsCoordinatesList[i][:, 0], HullPointsCoordinatesList[i][:, 1])
            HullPoints_XY = numpy.vstack((HullPoints_X, HullPoints_Y)).T.tolist()
            HullPolygons2[i] = Polygon(HullPoints_XY, facecolor='none', edgecolor='black', alpha=0.6, closed=True, linewidth=2)
            plt.gca().add_patch(HullPolygons2[i])

    Plot1()

    # -----------------
    # Fig 2: Velocities
    # -----------------

    def Plot2():
        """
        U and V velocities
        """

        fig_2, axes_2 = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        axes_2[0, 0].set_rasterization_zorder(0)
        axes_2[0, 1].set_rasterization_zorder(0)
        axes_2[1, 0].set_rasterization_zorder(0)
        axes_2[1, 1].set_rasterization_zorder(0)
        map_2_11 = DrawMap(axes_2[0, 0])
        map_2_12 = DrawMap(axes_2[0, 1])
        map_2_21 = DrawMap(axes_2[1, 0])
        map_2_22 = DrawMap(axes_2[1, 1])

        # Draw Mapscale
        # Index = int(Latitude.size / 4)
        # x0, y0 = map_2_11(Longitude[0], Latitude[0])
        # x1, y1 = map_2_11(Longitude[Index], Latitude[0])
        # Distance = (x1 - x0) / 1000 # Length of scale in Km
        # Distance = 40 # For Monterey Dataset
        Distance = 5 # For Martha Dataset
        map_2_11.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')
        map_2_12.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')
        map_2_21.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')
        map_2_22.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')

        ContourLevels = 300

        LongitudesGridOnMap, LatitudesGridOnMap= map_2_11(LongitudesGrid, LatitudesGrid)
        UO = Contour_U_Original = map_2_11.contourf(LongitudesGridOnMap, LatitudesGridOnMap, U_Original, ContourLevels, corner_mask=False, cmap=cm.jet, zorder=-1, rasterized=True)
        UI = Contour_U_Inpainted = map_2_12.contourf(LongitudesGridOnMap, LatitudesGridOnMap, U_Inpainted, ContourLevels, corner_mask=False, cmap=cm.jet, zorder=-1, rasterized=True)
        VO = Contour_V_Original = map_2_21.contourf(LongitudesGridOnMap, LatitudesGridOnMap, V_Original, ContourLevels, corner_mask=False, cmap=cm.jet, zorder=-1, rasterized=True)
        VI = Contour_V_Inpainted = map_2_22.contourf(LongitudesGridOnMap, LatitudesGridOnMap, V_Inpainted, ContourLevels, corner_mask=False, cmap=cm.jet, zorder=-1)

        # Colorbars
        divider_00 = make_axes_locatable(axes_2[0, 0])
        cax_00 = divider_00.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(UO, cax=cax_00)

        divider_01 = make_axes_locatable(axes_2[0, 1])
        cax_01 = divider_01.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(UI, cax=cax_01)

        divider_10 = make_axes_locatable(axes_2[1, 0])
        cax_10 = divider_10.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(VO, cax=cax_10)

        divider_11 = make_axes_locatable(axes_2[1, 1])
        cax_11 = divider_11.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(VI, cax=cax_11)

        axes_2[0, 0].set_title('Original East velocity')
        axes_2[0, 1].set_title('Restored East velocity')
        axes_2[1, 0].set_title('Original North velocity')
        axes_2[1, 1].set_title('Restored North velocity')

    Plot2()

    # -----------------
    # Fig 3: Streamplot
    # -----------------

    def Plot3():
        """
        Streamplots
        """

        # plot streamlines
        fig_3, axes_3 = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        axes_3[0, 0].set_rasterization_zorder(0)
        axes_3[0, 1].set_rasterization_zorder(0)
        axes_3[1, 0].set_rasterization_zorder(0)
        axes_3[1, 1].set_rasterization_zorder(0)
        map_3_11 = DrawMap(axes_3[0, 0])
        map_3_12 = DrawMap(axes_3[0, 1])

        # -----------------------
        # Draw Map For StreamPlot
        # -----------------------

        def DrawMapForStreamPlot(axis):
            """
            This map does not plot coasts and ocean. This is for only plotting streamlines on a white
            background so that we can scale it and suporimpose it on the previous plots in inkscape.
            """

            MinLongitude = numpy.min(Longitude)
            MidLongitude = numpy.mean(Longitude)
            MaxLongitude = numpy.max(Longitude)
            MinLatitude = numpy.min(Latitude)
            MidLatitude = numpy.mean(Latitude)
            MaxLatitude = numpy.max(Latitude)
        
            # Since we do not plot coasts, we use 'i' option for lowest resolution.
            map = Basemap( \
                    ax = axis, \
                    projection = 'aeqd', \
                    llcrnrlon=MinLongitude, \
                    llcrnrlat=MinLatitude, \
                    urcrnrlon=MaxLongitude, \
                    urcrnrlat=MaxLatitude, \
                    area_thresh = 0.1, \
                    lon_0 = MidLongitude, \
                    lat_0 = MidLatitude, \
                    resolution='i')

            return map

        # -------------------

        map_3_21 = DrawMapForStreamPlot(axes_3[1, 0])
        map_3_22 = DrawMapForStreamPlot(axes_3[1, 1])

        # For streamplot, we should use the projected lat and lon in the native coordinate of projection of the map
        ProjectedLongitudesGridOnMap, ProjectedLatitudesGridOnMap = map_3_21.makegrid(U_Original.shape[1], U_Original.shape[0], returnxy=True)[2:4]

        # These are needed for Martha's dataset, but not needed for MontereyBay
        # ProjectedLongitudesGridOnMap = ProjectedLongitudesGridOnMap[::-1, :]
        # ProjectedLatitudesGridOnMap = ProjectedLatitudesGridOnMap[::-1, :]

        VelocityMagnitude_Original = numpy.ma.sqrt(U_Original**2 + V_Original**2)
        VelocityMagnitude_Inpainted = numpy.sqrt(U_Inpainted**2 + V_Inpainted**2)

        LineWidth_Original = 3 * VelocityMagnitude_Original / VelocityMagnitude_Original.max()
        LineWidth_Inpainted = 3 * VelocityMagnitude_Inpainted / VelocityMagnitude_Inpainted.max()

        MinValue_Original = numpy.min(VelocityMagnitude_Original)
        MaxValue_Original = numpy.max(VelocityMagnitude_Original)
        MinValue_Inpainted = numpy.min(VelocityMagnitude_Inpainted)
        MaxValue_Inpainted = numpy.max(VelocityMagnitude_Inpainted)

        # MinValue_Original -= (MaxValue_Original - MinValue_Original) * 0.2
        # MinValue_Inpainted -= (MaxValue_Inpainted - MinValue_Inpainted) * 0.2

        Norm_Original = matplotlib.colors.Normalize(vmin=MinValue_Original, vmax=MaxValue_Original)
        Norm_Inpainted = matplotlib.colors.Normalize(vmin=MinValue_Inpainted, vmax=MaxValue_Inpainted)

        StreamPlot_Original = map_3_21.streamplot(ProjectedLongitudesGridOnMap, ProjectedLatitudesGridOnMap, U_Original, V_Original, color=VelocityMagnitude_Original, density=5, \
                linewidth=LineWidth_Original, cmap=plt.cm.ocean_r, norm=Norm_Original, zorder=-1)
        StreamPlot_Inpainted = map_3_22.streamplot(ProjectedLongitudesGridOnMap, ProjectedLatitudesGridOnMap, U_Inpainted, V_Inpainted, color=VelocityMagnitude_Inpainted, density=5, \
                linewidth=LineWidth_Inpainted, cmap=plt.cm.ocean_r, norm=Norm_Inpainted, zorder=-1)

        # Create axes for colorbar that is the same size as the plot axes
        divider_10 = make_axes_locatable(axes_3[1, 0])
        cax_10 = divider_10.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(StreamPlot_Original.lines, cax=cax_10)

        divider_11 = make_axes_locatable(axes_3[1, 1])
        cax_11 = divider_11.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(StreamPlot_Inpainted.lines, cax=cax_11)

        axes_3[0, 0].set_title('Original velocity streamlines')
        axes_3[0, 1].set_title('Restored velocity streamlines')

        # Draw Mapscale
        # Index = int(Latitude.size / 4)
        # x0, y0 = map_3_11(Longitude[0], Latitude[0])
        # x1, y1 = map_3_11(Longitude[Index], Latitude[0])
        # Distance = (x1 - x0) / 1000 # Length of scale in Km
        Distance = 40 # For Monterey Dataset
        # Distance = 5 # For Martha Dataset
        map_3_11.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')
        map_3_12.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')
        map_3_21.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')
        map_3_22.drawmapscale(MidLongitude, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')

    Plot3()

    # -------------
    # Fig 4: Quiver
    # -------------

    def Plot4():
        """
        Quivers
        """

        # Plot quiver of u and v
        fig_4, axes_4 = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        map_41 = DrawMap(axes_4[0])
        map_42 = DrawMap(axes_4[1])
        map_41.quiver(LongitudesGridOnMap, LatitudesGridOnMap, U_Original, V_Original, VelocityMagnitude_Original, scale=1000, scale_units='inches')
        map_42.quiver(LongitudesGridOnMap, LatitudesGridOnMap, U_Inpainted, V_Inpainted, VelocityMagnitude_Inpainted, scale=1000, scale_units='inches')
        axes_4[0].set_title('Original velocity vector field')
        axes_4[1].set_title('Restored velocity vector field')

    # Plot4()

    plt.show()
