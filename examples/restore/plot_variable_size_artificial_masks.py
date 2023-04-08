#! /usr/bin/env python

"""
Example:
./plot_variable_size_artificial_masks.py -i http://transport.me.berkeley.edu/thredds/dodsC/root/WHOI-HFR/WHOI_HFR_2014_original.nc -o ~/Desktop/output.nc -L 0 -d 5 -s -c -t 46
"""

# =======
# Imports
# =======

import numpy
import sys
import matplotlib.pyplot as plt
import random

# Modules
import InputOutput
import Plots
import DigitalImage
import Geography

# =================
# Create Extra Mask
# =================

def CreateExtraMask():
    """
    Inserts mask with a list of indices.
    """

    # Small set
    OffsetLatitudeIndex = 11
    OffsetLongitudeIndex = 15
    ExtraMaskIndicesList = [ \
            (1,0),(2,0),(3,0), \
            (0,1),(1,1),(2,1),(3,1),(4,1), \
            (0,2),(1,2),(2,2),(3,2),(4,2),(5,2), \
            (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),(6,3), \
            (1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4), \
            (3,5),(4,5),(5,5),(6,5),(7,5),(8,5), \
            (1,6),(2,6),(3,6),(4,6),(5,6),(6,6),(7,6), \
            (0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7), \
            (0,8),(1,8),(2,8),(3,8),(4,8),(5,8), \
            (0,9),(1,9),(2,9),(3,9),(4,9), \
            (1,10),(2,10),(3,10), \
            ]
    # ExtraMaskIndicesList = [ \
    #         (1,0),(2,0),(3,0), \
    #         (0,1),(1,1),(2,1),(3,1),(4,1), \
    #         (0,2),(1,2),(2,2),(3,2),(4,2),(5,2), \
    #         (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),(6,3), \
    #         (1,4),(2,4),(3,4),(5,4),(6,4),(7,4), \
    #         (3,5),(4,5),(7,5),(8,5), \
    #         (1,6),(2,6),(3,6),(5,6),(6,6),(7,6), \
    #         (0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7), \
    #         (0,8),(1,8),(2,8),(3,8),(4,8),(5,8), \
    #         (0,9),(1,9),(2,9),(3,9),(4,9), \
    #         (1,10),(2,10),(3,10), \
    #         ]

    # Larger set
    OffsetLatitudeIndex = 10
    OffsetLongitudeIndex = 14
    # ExtraMaskIndicesList = [ \
    #         (1,0),(2,0),(3,0),(4,0), \
    #         (0,1),(1,1),(2,1),(3,1),(4,1),(5,1), \
    #         (0,2),(1,2),(2,2),(3,2),(4,2),(5,2),(6,2), \
    #         (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),(7,3), \
    #         (1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4), \
    #         (2,5),(3,5),(4,5),(5,5),(6,5),(7,5),(8,5),(9,5), \
    #         (3,6),(4,6),(5,6),(6,6),(7,6),(8,6),(9,6),(10,6), \
    #         (2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(8,7),(9,7), \
    #         (1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(8,8), \
    #         (0,9),(1,9),(2,9),(3,9),(4,9),(5,9),(6,9),(7,9), \
    #         (0,10),(1,10),(2,10),(3,10),(4,10),(5,10),(6,10), \
    #         (0,11),(1,11),(2,11),(3,11),(4,11),(5,11), \
    #         (1,12),(2,12),(3,12),(4,12), \
    #         ]

    # ExtraMaskIndicesList = [ \
    #         (1,0),(2,0),(3,0),(4,0), \
    #         (0,1),(1,1),(2,1),(3,1),(4,1),(5,1), \
    #         (0,2),(1,2),(2,2),(3,2),(4,2),(5,2),(6,2), \
    #         (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),(7,3), \
    #         (1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4), \
    #         (2,5),(3,5),(4,5),(5,5),(6,5),(7,5),(8,5),(9,5), \
    #         (3,6),(4,6),(5,6),(6,6),(7,6),(8,6),(9,6),(10,6), \
    #         (2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(8,7),(9,7), \
    #         (1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(8,8), \
    #         (0,9),(1,9),(2,9),(3,9),(4,9),(5,9),(6,9),(7,9), \
    #         (0,10),(1,10),(2,10),(3,10),(4,10),(5,10),(6,10), \
    #         (0,11),(1,11),(2,11),(3,11),(4,11),(5,11), \
    #         (1,12),(2,12),(3,12),(4,12), \
    #         ]

    ExtraMaskIndices = numpy.array(ExtraMaskIndicesList)
    ExtraMaskIndices[:,0] += OffsetLatitudeIndex
    ExtraMaskIndices[:,1] += OffsetLongitudeIndex

    return ExtraMaskIndices

# ================================
# Create Random Extra Mask Indices
# ================================

def CreateRandomExtraMaskIndices( \
        ExtraMaskBoxWidth, \
        ExtraMaskBoxHeight, \
        ExtraMaskOffsetLatitudeIndex, \
        ExtraMaskOffsetLongitudeIndex, \
        NumExtraMaskPoints):
    """
    This funcrtion creates a random selection of N = "NumExtraMaskPoints" points inside a box of size Width and Height
    with the given longitude and latitude offset. The return value is a Nx2 array of N random indices.
    """

    # -----------------------
    # Point Id to Point Index
    # -----------------------

    def PointIdToPointIndex(PointId,ExtraMaskBoxWidth):
        """
        Converts a goven point Id to the latitude and longitude indices with respect to the original grid.
        The output is ordered as [LatitudeIndex,LongitudeIndex] tuple.
        """
        PointLatitudeIndex = int(PointId/ExtraMaskBoxWidth)
        PointLongitudeIndex = PointId % ExtraMaskBoxWidth
        Tuple = (PointLatitudeIndex,PointLongitudeIndex)
        return Tuple

    # -----------------------

    # Get a random sample of point ids in the box
    NumPointsInBox = ExtraMaskBoxWidth * ExtraMaskBoxHeight
    RandomIntegers = random.sample(range(0,NumPointsInBox),NumExtraMaskPoints)

    # Convert the points ids to point index
    ExtraMaskIndices = numpy.zeros((NumExtraMaskPoints,2),dtype=int)
    for i in range(NumExtraMaskPoints):
        Tuple = PointIdToPointIndex(RandomIntegers[i],ExtraMaskBoxWidth)
        ExtraMaskIndices[i,0] = Tuple[0] + ExtraMaskOffsetLatitudeIndex
        ExtraMaskIndices[i,1] = Tuple[1] + ExtraMaskOffsetLongitudeIndex

    return ExtraMaskIndices

# ====================================
# Restore Missing Points Inside Domain
# ====================================

def RestoreMissingPointsInsideDomain( \
        ExtraMaskIndices, \
        LandIndices, \
        U_Original, \
        V_Original, \
        U_InpaintedAllMissingPoints, \
        V_InpaintedAllMissingPoints):
    """
    This function takes the inpainted image, and retains only the inpainted points that are inside the convex hull.

    The function "InpaintAllMissingPoints" inpaints all points including inside and outside the convex hull. However
    this function discards the missing points that are outside the convex hull.

    Masked points:
        -Points on land
        -All missing points in ocean outside hull

    Numeric points:
        -Valid points from original dataset (This does not include lan points)
        -Missing points in ocean inside hull that are inpainted.
    """

    U_Inpainted_Masked = numpy.ma.copy(U_Original)
    V_Inpainted_Masked = numpy.ma.copy(V_Original)

    # Restore U
    for i in range(ExtraMaskIndices.shape[0]):
        U_Inpainted_Masked[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]] = U_InpaintedAllMissingPoints[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]]

    # Restore V
    for i in range(ExtraMaskIndices.shape[0]):
        V_Inpainted_Masked[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]] = V_InpaintedAllMissingPoints[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]]

    return U_Inpainted_Masked,V_Inpainted_Masked

# ==============================
# Restore Time Frame Per Process
# ==============================

def RestoreTimeFramePerProcess( \
        Longitude, \
        Latitude, \
        LandIndices, \
        U_AllTimes, \
        V_AllTimes, \
        Diffusivity, \
        SweepAllDirections, \
        PlotTimeFrame, \
        IncludeLandForHull, \
        UseConvexHull, \
        Alpha, \
        NumExtraMaskPoints, \
        TimeIndex):
    """
    Do all calculations for one time frame. This function is called from multiprocessing object. Each time frame is
    duspatched to a processor.
    """

    # Get one time frame of U and V velocities.
    U_Original = U_AllTimes[TimeIndex,:]
    V_Original = V_AllTimes[TimeIndex,:]

    # -----------------
    # Make Array Masked
    # -----------------

    def MakeArrayMasked(Array):

        if (not hasattr(Array,'mask')) or (Array.mask.size == 1):
            if numpy.isnan(Array).any() or numpy.isinf(Array).any():
                # This array is not masked. Make a mask based no nan and inf
                Mask_nan = numpy.isnan(Array)
                Mask_inf = numpy.isinf(Array)
                Mask = numpy.logical_or(Mask_nan,Mask_inf)
                Array = numpy.ma.masked_array(Array,mask=Mask)
        else:
            # This array is masked. But check if any non-masked value is nan or inf
            for i in range(Array.shape[0]):
                for j in range(Array.shape[1]):
                    if Array.mask[i,j] == False:
                        if numpy.isnan(Array[i,j]) or numpy.isinf(Array[i,j]):
                            Array.mask[i,j] = True

        return Array

    # --------------------------

    # Make sure arrays are masked arrays
    U_Original = MakeArrayMasked(U_Original)
    V_Original = MakeArrayMasked(V_Original)

    # Alter the original arrays by adding extra mask
    U_Original_Altered = numpy.ma.copy(U_Original)
    V_Original_Altered = numpy.ma.copy(V_Original)

    # Add extra mask indices
    # ExtraMaskIndices = CreateExtraMask()

    # Create artificial mask points. The followings are the geometry of the box where missing points are introduced.
    ExtraMaskBoxWidth = 10
    ExtraMaskBoxHeight = 10
    ExtraMaskOffsetLatitudeIndex = 10
    ExtraMaskOffsetLongitudeIndex = 14

    # Create indices of random mask points
    ExtraMaskIndices = CreateRandomExtraMaskIndices( \
            ExtraMaskBoxWidth, \
            ExtraMaskBoxHeight, \
            ExtraMaskOffsetLatitudeIndex, \
            ExtraMaskOffsetLongitudeIndex, \
            NumExtraMaskPoints)

    # Mask the array with artifial extra masked points
    for i in range(ExtraMaskIndices.shape[0]):
        U_Original_Altered[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]] = numpy.ma.masked
        V_Original_Altered[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]] = numpy.ma.masked

    # Find indices of valid points, missing points inside and outside the domain
    AllMissingIndicesInOcean,MissingIndicesInOceanInsideHull,MissingIndicesInOceanOutsideHull,ValidIndices,HullPointsCoordinatesList = \
            Geography.LocateMissingData( \
            Longitude, \
            Latitude, \
            LandIndices, \
            U_Original_Altered, \
            IncludeLandForHull, \
            UseConvexHull, \
            Alpha)

    # Set data on land to be zero (Note: This should be done after finding the convex hull)
    if hasattr(U_Original_Altered,'mask'):
        U_Original_Altered.unshare_mask()

    if hasattr(V_Original_Altered,'mask'):
        V_Original_Altered.unshare_mask()

    if not numpy.isnan(LandIndices):
        if LandIndices.shape[0] > 0:
            for LandId in range(LandIndices.shape[0]):
                U_Original_Altered[LandIndices[LandId,0],LandIndices[LandId,1]] = 0.0
                V_Original_Altered[LandIndices[LandId,0],LandIndices[LandId,1]] = 0.0

    # Inpaint all missing points including inside and outside the domain
    U_InpaintedAllMissingPoints,V_InpaintedAllMissingPoints = DigitalImage.InpaintAllMissingPoints( \
            AllMissingIndicesInOcean, \
            LandIndices, \
            ValidIndices, \
            U_Original_Altered, \
            V_Original_Altered, \
            Diffusivity, \
            SweepAllDirections)

    # Use the inpainted point of missing points ONLY inside the domain to restore the data
    U_Inpainted_Masked,V_Inpainted_Masked = RestoreMissingPointsInsideDomain( \
            ExtraMaskIndices, \
            LandIndices, \
            U_Original, \
            V_Original, \
            U_InpaintedAllMissingPoints, \
            V_InpaintedAllMissingPoints)

    # Plot the grid and inpainted results
    PlotResults = False
    if PlotResults == True:
        print("Plotting timeframe: %d ..."%PlotTimeFrame)

        Plots.PlotResults( \
                Longitude, \
                Latitude, \
                U_Original, \
                V_Original, \
                U_Inpainted_Masked, \
                V_Inpainted_Masked, \
                AllMissingIndicesInOcean, \
                MissingIndicesInOceanInsideHull, \
                MissingIndicesInOceanOutsideHull, \
                ValidIndices, \
                LandIndices, \
                HullPointsCoordinatesList)

        return

    return TimeIndex,U_Original,V_Original,U_Inpainted_Masked,V_Inpainted_Masked,ExtraMaskIndices

# ============================================
# Compute Relative Error Of Extra Mask Indices
# ============================================

def ComputeRelativeErrorOfExtraMaskIndices( \
        U_Original, \
        V_Original, \
        U_Inpainted_Masked, \
        V_Inpainted_Masked, \
        ExtraMaskIndices):
    """
    Computes error for extra mask indices between original and inpainted arrays.
    """

    NumberOfExtraMaskedPoints = ExtraMaskIndices.shape[0]
    RelativeErrors_Magnitude = numpy.zeros(NumberOfExtraMaskedPoints,dtype=float)
    RelativeErrors_Direction = numpy.zeros(NumberOfExtraMaskedPoints,dtype=float)

    # ---------------------------
    # Get Magnitude And Direction
    # ---------------------------

    def GetMagnitudeAndDirection(u,v):
        r = numpy.sqrt(u**2 + v**2)
        theta = numpy.arctan2(u,v)

        # return u,v
        return r,theta

    # ---------------------------

    for i in range(NumberOfExtraMaskedPoints):

        # Original
        Magnitude_Original,Direction_Original = GetMagnitudeAndDirection( \
                U_Original[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]], \
                V_Original[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]])

        # Inpainted
        Magnitude_Inpainted,Direction_Inpainted = GetMagnitudeAndDirection( \
                U_Inpainted_Masked[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]], \
                V_Inpainted_Masked[ExtraMaskIndices[i,0],ExtraMaskIndices[i,1]])

        # Error on magnitude
        RelativeErrors_Magnitude[i] = numpy.abs((Magnitude_Original - Magnitude_Inpainted) / (Magnitude_Original + 1e-15))
        RelativeErrors_Direction[i] = numpy.abs((Direction_Original - Direction_Inpainted) / (Direction_Original + 1e-15))

    MeanRelativeError_Magnitude = numpy.mean(RelativeErrors_Magnitude)
    MeanRelativeError_Direction = numpy.mean(RelativeErrors_Direction)

    return MeanRelativeError_Magnitude,MeanRelativeError_Direction 

# ===========
# Plot Errors
# ===========

def PlotErrors( \
        Statistical_AVG_RelativeErrors_Magnitude, \
        Statistical_STD_RelativeErrors_Magnitude, \
        Statistical_AVG_RelativeErrors_Direction, \
        Statistical_STD_RelativeErrors_Direction):

    fig,ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    NumTests = Statistical_AVG_RelativeErrors_Magnitude.size
    Range = numpy.arange(1,1+NumTests)

    ax1.plot(Range,Statistical_AVG_RelativeErrors_Magnitude,color='darkred',label='Velocity magnitude mean relative error')
    ax1.fill_between(Range, \
            Statistical_AVG_RelativeErrors_Magnitude - Statistical_STD_RelativeErrors_Magnitude, \
            Statistical_AVG_RelativeErrors_Magnitude + Statistical_STD_RelativeErrors_Magnitude, \
            facecolor='tomato',alpha=0.4,edgecolor='tomato')

    ax1.plot(Range,Statistical_AVG_RelativeErrors_Direction,color='green',label='Velocity direction mean relative error')
    ax1.fill_between(Range, \
            Statistical_AVG_RelativeErrors_Direction - Statistical_STD_RelativeErrors_Direction, \
            Statistical_AVG_RelativeErrors_Direction + Statistical_STD_RelativeErrors_Direction, \
            facecolor='greenyellow',alpha=0.4,edgecolor='greenyellow')

    plt.xlabel('Number of artificial scattered missing points')
    plt.ylabel('Mean relative error')
    plt.xlim(Range[0],Range[-1])
    plt.ylim(ymin=0.0)
    plt.legend(loc='upper left')

    plt.show()

# ====
# Main
# ====

def main(argv):
    """
    These parameters should be set for the opencv.inpaint method:

    Diffusivity: 
        (Default = 20) The difusiob coefficient

    SweepAllDirections:
        (Default to = True) If set to True, the inpaint is performed 4 times on the flipped left/right and up/down of the image.

    Notes on parallelization:
        - We have used multiprocessing.Pool.imap_unordered. Other options are apply, apply_async,map,imap, etc.
        - The imap_unordered can only accept functions with one argument, where the argument is the iterator of the parallelization.
        - In order to pass a multi-argument function, we have used functool.partial.
        - The imap_unordered distributes all tasks to processes by a chunk_size. Meaning that each process is assigned a chunk size
          number of iterators of tasks to do, before loads the next chunk size. By default the chunk size is 1. This causes many 
          function calls and slows down the paralleization. By setting the chunk_size=100, each process is assigned 100 iteration,
          with only 1 function call. So if we have 4 processors, each one perform 100 tasks. After each process is done with a 100
          task, it loads another 100 task from the pool of tasks in an unordered manner. The "map" in imap_unorderdd ensures that
          all processes are assigned a task without having an idle process.
    """

    # Parse arguments
    Arguments = InputOutput.ParseArguments(argv)
    InputFilename = Arguments['FullPathInputFilename']
    OutputFilename = Arguments['FullPathOutputFilename']
    Diffusivity = Arguments['Diffusivity']
    SweepAllDirections = Arguments['SweepAllDirections']
    PlotTimeFrame = Arguments['TimeFrame']
    IncludeLandForHull = Arguments['IncludeLandForHull']
    UseConvexHull = Arguments['UseConvexHull']
    Alpha = Arguments['Alpha']

    # Open file
    agg = InputOutput.LoadDataset(InputFilename)

    # Load variables
    DatetimeObject,LongitudeObject,LatitudeObject,EastVelocityObject,NorthVelocityObject,EastVelocityErrorObject,NorthVelocityErrorObject = InputOutput.LoadVariables(agg)

    # Get arrays
    Datetime = DatetimeObject[:]
    Longitude = LongitudeObject[:]
    Latitude = LatitudeObject[:]
    U_AllTimes = EastVelocityObject[:]
    V_AllTimes = NorthVelocityObject[:]

    # Determine the land
    LandIndices,OceanIndices = Geography.DoNotFindLandAndOceanIndices(Longitude,Latitude)

    ExtraMaskBoxWidth = 10
    ExtraMaskBoxHeigth = 10
    NumMaskPoints = ExtraMaskBoxWidth * ExtraMaskBoxHeigth - 1
    Statistical_AVG_RelativeErrors_Magnitude = numpy.zeros(NumMaskPoints,dtype=float)
    Statistical_STD_RelativeErrors_Magnitude = numpy.zeros(NumMaskPoints,dtype=float)
    Statistical_AVG_RelativeErrors_Direction = numpy.zeros(NumMaskPoints,dtype=float)
    Statistical_STD_RelativeErrors_Direction = numpy.zeros(NumMaskPoints,dtype=float)

    for i in range(NumMaskPoints):

        # Use more test experiements when number of maks points are less, and less experiemtn when the mask box is filled more of mask points.
        # This is to make sure that the STD is evenly comparable with different number of mask points
        NumExperiments = NumMaskPoints - int(0.9*i)

        NumExtraMaskPoints = i+1

        MeanRelativeErrors_Experiment_Magnitude = numpy.zeros(NumExperiments,dtype=float)
        MeanRelativeErrors_Experiment_Direction = numpy.zeros(NumExperiments,dtype=float)

        for j in range(NumExperiments):

            print("NumExtraMaskPoints: %d, Test: %d"%(NumExtraMaskPoints,j+1))

            TimeIndex,U_Original,V_Original,U_Inpainted_Masked,V_Inpainted_Masked,ExtraMaskIndices = RestoreTimeFramePerProcess( \
                    Longitude, \
                    Latitude, \
                    LandIndices, \
                    U_AllTimes, \
                    V_AllTimes, \
                    Diffusivity, \
                    SweepAllDirections, \
                    PlotTimeFrame, \
                    IncludeLandForHull, \
                    UseConvexHull, \
                    Alpha, \
                    NumExtraMaskPoints, \
                    PlotTimeFrame)

            MeanRelativeError_Magnitude,MeanRelativeError_Direction = ComputeRelativeErrorOfExtraMaskIndices( \
                    U_Original, \
                    V_Original, \
                    U_Inpainted_Masked, \
                    V_Inpainted_Masked, \
                    ExtraMaskIndices)

            # Store result of each experiment
            MeanRelativeErrors_Experiment_Magnitude[j] = MeanRelativeError_Magnitude
            MeanRelativeErrors_Experiment_Direction[j] = MeanRelativeError_Direction

        print(" ")

        # Get average and std of experiments
        Statistical_AVG_RelativeError_Magnitude = numpy.mean(MeanRelativeErrors_Experiment_Magnitude)
        Statistical_STD_RelativeError_Magnitude = numpy.std(MeanRelativeErrors_Experiment_Magnitude)
        Statistical_AVG_RelativeError_Direction = numpy.mean(MeanRelativeErrors_Experiment_Direction)
        Statistical_STD_RelativeError_Direction = numpy.std(MeanRelativeErrors_Experiment_Direction)

        # Get average and std of each experiemnt
        Statistical_AVG_RelativeErrors_Magnitude[i] = Statistical_AVG_RelativeError_Magnitude
        Statistical_STD_RelativeErrors_Magnitude[i] = Statistical_STD_RelativeError_Magnitude
        Statistical_AVG_RelativeErrors_Direction[i] = Statistical_AVG_RelativeError_Direction
        Statistical_STD_RelativeErrors_Direction[i] = Statistical_STD_RelativeError_Direction

    # Plot
    PlotErrors( \
            Statistical_AVG_RelativeErrors_Magnitude, \
            Statistical_STD_RelativeErrors_Magnitude, \
            Statistical_AVG_RelativeErrors_Direction, \
            Statistical_STD_RelativeErrors_Direction)

    agg.close()

# ===========
# System Main
# ===========

if __name__ == "__main__":

    # Main function
    main(sys.argv)
