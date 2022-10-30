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

import multiprocessing
from functools import partial
import time
import warnings

# Modules
import InputOutput
import Plots
import DigitalImage
import Geography
import UncertaintyQuantification
import FileUtilities

# ==========================
# Refine Grid By Adding Mask
# ==========================

def RefineGridByAddingMask( \
        RefinementLevel, \
        Data_Longitude, \
        Data_Latitude, \
        Data_U_AllTimes, \
        Data_V_AllTimes):
    """
    Increases the size of grid by the factor of RefinementLevel.
    The extra points on the grid will be numpy.ma.mask.

    Note that this does NOT refine the data. Rather, this just increases the size of grid.
    That is, between each two points we introduce a few grid points and we mask them.
    By masking these new points we will tend to restore them later.
    """

    # No refinement for level 1
    if RefinementLevel == 1:
        return Data_Longitude, Data_Latitude, Data_U_AllTimes, Data_V_AllTimes

    # Longitude
    Longitude = numpy.zeros(RefinementLevel*(Data_Longitude.size-1)+1, dtype=float)
    for i in range(Data_Longitude.size):

        # Data points
        Longitude[RefinementLevel*i] = Data_Longitude[i]

        # Fill in extra points
        if i < Data_Longitude.size - 1:
            for j in range(1, RefinementLevel):
                Weight = float(j)/float(RefinementLevel)
                Longitude[RefinementLevel*i+j] = ((1.0-Weight) * Data_Longitude[i]) + (Weight * Data_Longitude[i+1])

    # Latitude
    Latitude = numpy.zeros(RefinementLevel*(Data_Latitude.size-1)+1, dtype=float)
    for i in range(Data_Latitude.size):

        # Data points
        Latitude[RefinementLevel*i] = Data_Latitude[i]

        # Fill in extra points
        if i < Data_Latitude.size - 1:
            for j in range(1, RefinementLevel):
                Weight = float(j)/float(RefinementLevel)
                Latitude[RefinementLevel*i+j] = ((1.0-Weight) * Data_Latitude[i]) + (Weight * Data_Latitude[i+1])

    # East Velocity
    U_AllTimes = numpy.ma.masked_all( \
            (Data_U_AllTimes.shape[0], \
            RefinementLevel*(Data_U_AllTimes.shape[1]-1)+1, \
            RefinementLevel*(Data_U_AllTimes.shape[2]-1)+1), \
            dtype=numpy.float64)

    U_AllTimes[:, ::RefinementLevel, ::RefinementLevel] = Data_U_AllTimes[:, :, :]

    # North Velocity
    V_AllTimes = numpy.ma.masked_all( \
            (Data_V_AllTimes.shape[0], \
            RefinementLevel*(Data_V_AllTimes.shape[1]-1)+1, \
            RefinementLevel*(Data_V_AllTimes.shape[2]-1)+1), \
            dtype=numpy.float64)

    V_AllTimes[:, ::RefinementLevel, ::RefinementLevel] = Data_V_AllTimes[:, :, :]
 
    return Longitude, Latitude, U_AllTimes, V_AllTimes 

# ============================
# Refine Grid By Interpolation
# ============================

def RefineGridByInterpolation( \
        RefinementLevel, \
        Data_Longitude, \
        Data_Latitude, \
        Data_U_AllTimes, \
        Data_V_AllTimes):
    """
    Refines grid by means of interpolation. 
    Note that this actuallty interpolates the data which is in contrast to the previous function: "RefineGridByAddingMask"
    """

    # TODO
    print('hi')

# =================
# Make Array Masked
# =================

def MakeArrayMasked(Array):
    """
    Often the Array is not masked, but has nan or inf values. 
    This function creates a masked array and mask nan and inf.

    Input:
        - Array: is a 2D numpy array.
    Output:
        - Array: is a 2D numpy.ma array.

    Note: Array should be numpy obkect not netCDF object. So if you have a netCDF
          object, pass its numpy array with Array[:] into this function.
    """

    if (not hasattr(Array, 'mask')) or (Array.mask.size == 1):
        if numpy.isnan(Array).any() or numpy.isinf(Array).any():
            # This array is not masked. Make a mask based no nan and inf
            Mask_nan = numpy.isnan(Array)
            Mask_inf = numpy.isinf(Array)
            Mask = numpy.logical_or(Mask_nan, Mask_inf)
            Array = numpy.ma.masked_array(Array, mask=Mask)
    else:
        # This array is masked. But check if any non-masked value is nan or inf
        for i in range(Array.shape[0]):
            for j in range(Array.shape[1]):
                if Array.mask[i, j] == False:
                    if numpy.isnan(Array[i, j]) or numpy.isinf(Array[i, j]):
                        Array.mask[i, j] = True

    return Array

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
        Plot, \
        IncludeLandForHull, \
        UseConvexHull, \
        Alpha, \
        TimeIndex):
    """
    Do all calculations for one time frame. This function is called from multiprocessing object. Each time frame is
    dispatched to a processor.
    """

    # Get one time frame of U and V velocities.
    U_Original = U_AllTimes[TimeIndex, :]
    V_Original = V_AllTimes[TimeIndex, :]

    # Make sure arrays are masked arrays
    U_Original = MakeArrayMasked(U_Original)
    V_Original = MakeArrayMasked(V_Original)

    # Find indices of valid points, missing points inside and outside the domain
    # Note: In the following line, all indices outputs are Nx2, where the first column are latitude indices (not longitude)
    # and the second column indics are longitude indices (not latitude)
    AllMissingIndicesInOcean, MissingIndicesInOceanInsideHull, MissingIndicesInOceanOutsideHull, ValidIndices, HullPointsCoordinatesList = \
            Geography.LocateMissingData( \
            Longitude, \
            Latitude, \
            LandIndices, \
            U_Original, \
            IncludeLandForHull, \
            UseConvexHull, \
            Alpha)

    # Create mask Info
    MaskInfo = Geography.CreateMaskInfo( \
            U_Original, \
            LandIndices, \
            MissingIndicesInOceanInsideHull, \
            MissingIndicesInOceanOutsideHull, \
            ValidIndices)

    # Set data on land to be zero (Note: This should be done after finding the convex hull)
    if hasattr(U_Original, 'mask'):
        U_Original.unshare_mask()

    if hasattr(V_Original, 'mask'):
        V_Original.unshare_mask()

    if numpy.any(numpy.isnan(LandIndices)) == False:
        for LandId in range(LandIndices.shape[0]):
            U_Original[LandIndices[LandId, 0], LandIndices[LandId, 1]] = 0.0
            V_Original[LandIndices[LandId, 0], LandIndices[LandId, 1]] = 0.0

    # Inpaint all missing points including inside and outside the domain
    U_InpaintedAllMissingPoints, V_InpaintedAllMissingPoints = DigitalImage.InpaintAllMissingPoints( \
            AllMissingIndicesInOcean, \
            LandIndices, \
            ValidIndices, \
            U_Original, \
            V_Original, \
            Diffusivity, \
            SweepAllDirections)

    # Use the inpainted point of missing points ONLY inside the domain to restore the data
    U_Inpainted_Masked, V_Inpainted_Masked = DigitalImage.RestoreMissingPointsInsideDomain( \
            MissingIndicesInOceanInsideHull, \
            MissingIndicesInOceanOutsideHull, \
            LandIndices, \
            U_Original, \
            V_Original, \
            U_InpaintedAllMissingPoints, \
            V_InpaintedAllMissingPoints)

    # Plot the grid and inpainted results
    if Plot == True:
        print("Plotting timeframe: %d ..."%TimeIndex)

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

    return TimeIndex, U_Inpainted_Masked, V_Inpainted_Masked, MaskInfo

# ============================
# Restore Ensemble Per Process
# ============================

def RestoreEnsemblePerProcess( \
        LandIndices, \
        AllMissingIndicesInOcean, \
        MissingIndicesInOceanInsideHull, \
        MissingIndicesInOceanOutsideHull, \
        ValidIndices, \
        U_AllEnsembles, \
        V_AllEnsembles, \
        Diffusivity, \
        SweepAllDirections, \
        EnsembleIndex):
    """
    Do all calculations for one time frame. This function is called from multiprocessing object. Each time frame is
    dispatched to a processor.
    """

    # Get one ensemble
    U_Ensemble = U_AllEnsembles[EnsembleIndex, :, :]
    V_Ensemble = V_AllEnsembles[EnsembleIndex, :, :]

    # Set data on land to be zero (Note: This should be done after finding the convex hull)
    if hasattr(U_Ensemble, 'mask'):
        U_Ensemble.unshare_mask()

    if hasattr(V_Ensemble, 'mask'):
        V_Ensemble.unshare_mask()

    if numpy.any(numpy.isnan(LandIndices)) == False:
        for LandId in range(LandIndices.shape[0]):
            U_Ensemble[LandIndices[LandId, 0], LandIndices[LandId, 1]] = 0.0
            V_Ensemble[LandIndices[LandId, 0], LandIndices[LandId, 1]] = 0.0

    # Inpaint all missing points including inside and outside the domain
    U_InpaintedAllMissingPoints, V_InpaintedAllMissingPoints = DigitalImage.InpaintAllMissingPoints( \
            AllMissingIndicesInOcean, \
            LandIndices, \
            ValidIndices, \
            U_Ensemble, \
            V_Ensemble, \
            Diffusivity, \
            SweepAllDirections)

    # Use the inpainted point of missing points ONLY inside the domain to restore the data
    U_Inpainted_Masked, V_Inpainted_Masked = DigitalImage.RestoreMissingPointsInsideDomain( \
            MissingIndicesInOceanInsideHull, \
            MissingIndicesInOceanOutsideHull, \
            LandIndices, \
            U_Ensemble, \
            V_Ensemble, \
            U_InpaintedAllMissingPoints, \
            V_InpaintedAllMissingPoints)

    return EnsembleIndex, U_Inpainted_Masked, V_Inpainted_Masked

# ====
# Main
# ====

def main(argv):
    """
    These parameters should be set for the opencv.inpaint method:

    Diffusivity: 
        (Default = 20) The diffusion coefficient

    SweepAllDirections:
        (Default to = True) If set to True, the inpaint is performed 4 times on the flipped left/right and up/down of the image.

    Notes on parallelization:
        - We have used multiprocessing.Pool.imap_unordered. Other options are apply, apply_async, map, imap, etc.
        - The imap_unordered can only accept functions with one argument, where the argument is the iterator of the parallelization.
        - In order to pass a multi-argument function, we have used functool.partial.
        - The imap_unordered distributes all tasks to processes by a chunk_size. Meaning that each process is assigned a chunk size
          number of iterators of tasks to do, before loads the next chunk size. By default the chunk size is 1. This causes many 
          function calls and slows down the parallelization. By setting the chunk_size=100, each process is assigned 100 iteration,
          with only 1 function call. So if we have 4 processors, each one perform 100 tasks. After each process is done with a 100
          task, it loads another 100 task from the pool of tasks in an unordered manner. The "map" in imap_unorderdd ensures that
          all processes are assigned a task without having an idle process.
    """

    # Parse arguments
    Arguments = InputOutput.ParseArguments(argv)

    # Get list of all separate input files to process
    FullPathInputFilenamesList, InputBaseFilenamesList = FileUtilities.GetFullPathInputFilenamesList( \
            Arguments['FullPathInputFilename'], \
            Arguments['ProcessMultipleFiles'], \
            Arguments['MultipleFilesMinIteratorString'], \
            Arguments['MultipleFilesMaxIteratorString'])

    # Get the list of all output files to be written to
    FullPathOutputFilenamesList = FileUtilities.GetFullPathOutputFilenamesList( \
            Arguments['FullPathOutputFilename'], \
            Arguments['ProcessMultipleFiles'], \
            Arguments['MultipleFilesMinIteratorString'], \
            Arguments['MultipleFilesMaxIteratorString'])

    NumberOfFiles = len(FullPathInputFilenamesList)

    # Iterate over multiple separate files
    for FileIndex in range(NumberOfFiles):

        # Open file
        agg = InputOutput.LoadDataset(FullPathInputFilenamesList[FileIndex])

        # Load variables
        DatetimeObject, LongitudeObject, LatitudeObject, EastVelocityObject, NorthVelocityObject, EastVelocityErrorObject, NorthVelocityErrorObject = \
                InputOutput.LoadVariables(agg)

        # To not issue error/warning when data has nan
        numpy.warnings.filterwarnings('ignore')

        # Get arrays
        Datetime = DatetimeObject[:]
        # Data_Longitude = LongitudeObject[:]
        # Data_Latitude = LatitudeObject[:]
        # Data_U_AllTimes = EastVelocityObject[:]
        # Data_V_AllTimes = NorthVelocityObject[:]
        Longitude = LongitudeObject[:]
        Latitude = LatitudeObject[:]
        U_AllTimes = EastVelocityObject[:]
        V_AllTimes = NorthVelocityObject[:]

        # Refinement
        # Longitude, Latitude, U_AllTimes, V_AllTimes = RefineGridByAddingMask(Arguments['RefinementLevel'], Data_Longitude, Data_Latitude, Data_U_AllTimes, Data_V_AllTimes)

        # Determine the land
        if Arguments['ExcludeLandFromOcean'] == 0:
            # Returns nan for Land indices, and returns all available indices for ocean.
            LandIndices, OceanIndices = Geography.DoNotFindLandAndOceanIndices(Longitude, Latitude)
        elif Arguments['ExcludeLandFromOcean'] == 1:
            # Separate land and ocean. Most accurate, very slow for points on land.
            LandIndices, OceanIndices = Geography.FindLandAndOceanIndices1(Longitude, Latitude)
        elif Arguments['ExcludeLandFromOcean'] == 2:
            # Separate land and ocean. Least accurate, very fast
            LandIndices, OceanIndices = Geography.FindLandAndOceanIndices2(Longitude, Latitude)
        elif Argumentsp['ExcludeLandFromOcean'] == 3:
            # Currently Not working well.
            LandIndices, OceanIndices = Geography.FindLandAndOceanIndices3(Longitude, Latitude)  # Not working (land are not detected)
        else:
            raise RuntimeError("ExcludeLandFromOcean option is invalid.")

        # If plotting, remove these files:
        if Arguments['Plot'] == True:
            # Remove ~/.Xauthority and ~/.ICEauthority
            import os.path
            HomeDir = os.path.expanduser("~")
            if os.path.isfile(HomeDir+'/.Xauthority'):
                os.remove(HomeDir+'/.Xauthority')
            if os.path.isfile(HomeDir+'/.ICEauthority'):
                os.remove(HomeDir+'/.ICEauthority')

        # Check whether to perform uncertainty quantification or not
        if Arguments['UncertaintyQuantification'] == True:

            # -----------------------------
            # 1. Uncertainty Quantification
            # -----------------------------

            # Time frame
            TimeFrame = Arguments['TimeFrame']
            if TimeFrame >= U_AllTimes.shape[0]:
                raise ValueError('Time frame is out of bound.')
            elif TimeFrame < 0:
                TimeFrame = -1

            # Get one time frame of velocities
            U_OneTime = MakeArrayMasked(U_AllTimes[TimeFrame, :, :])
            V_OneTime = MakeArrayMasked(V_AllTimes[TimeFrame, :, :])

            # Check if data has errors of velocities variable
            if (EastVelocityErrorObject is None):
                raise ValueError('Input netCDF data does not have East Velocity error, which is needed for uncertainty quantification.')
            if (NorthVelocityErrorObject is None):
                raise ValueError('Input netCDF data does not have North Velocity error, which is needed for uncertainty quantification.')

            # Make sure arrays are masked arrays
            Error_U_OneTime = MakeArrayMasked(EastVelocityErrorObject[TimeFrame, :, :])
            Error_V_OneTime = MakeArrayMasked(NorthVelocityErrorObject[TimeFrame, :, :])

            # Scale Errors
            Scale = 0.08 # m/s
            Error_U_OneTime *= Scale
            Error_V_OneTime *= Scale

            # Errors are usually squared. Take square root
            # Error_U_OneTime = numpy.ma.sqrt(Error_U_OneTime)
            # Error_V_OneTime = numpy.ma.sqrt(Error_V_OneTime)

            # Find indices of valid points, missing points inside and outside the domain
            # Note: In the following line, all indices outputs are Nx2, where the first column are latitude indices (not longitude)
            # and the second column indics are longitude indices (not latitude)
            AllMissingIndicesInOcean, MissingIndicesInOceanInsideHull, MissingIndicesInOceanOutsideHull, ValidIndices, HullPointsCoordinatesList = \
                    Geography.LocateMissingData( \
                    Longitude, \
                    Latitude, \
                    LandIndices, \
                    U_OneTime, \
                    Arguments['IncludeLandForHull'], \
                    Arguments['UseConvexHull'], \
                    Arguments['Alpha'])

            # Create mask Info
            MaskInfo = Geography.CreateMaskInfo( \
                    U_OneTime, \
                    LandIndices, \
                    MissingIndicesInOceanInsideHull, \
                    MissingIndicesInOceanOutsideHull, \
                    ValidIndices)

             
            # Generate Ensembles (Longitude and Latitude are not neede, but only used for plots if uncommented)
            NumModes = None  # None makes NumModes to be maximum number of possible modes
            U_AllEnsembles = UncertaintyQuantification.GenerateImageEnsembles(Longitude, Latitude, U_OneTime, Error_U_OneTime, ValidIndices, Arguments['NumEnsembles'], NumModes)
            V_AllEnsembles = UncertaintyQuantification.GenerateImageEnsembles(Longitude, Latitude, V_OneTime, Error_V_OneTime, ValidIndices, Arguments['NumEnsembles'], NumModes)

            # Create a partial function in order to pass a function with only one argument to the multiprocessor
            RestoreEnsemblePerProcess_PartialFunct = partial( \
                    RestoreEnsemblePerProcess, \
                    LandIndices, \
                    AllMissingIndicesInOcean, \
                    MissingIndicesInOceanInsideHull, \
                    MissingIndicesInOceanOutsideHull, \
                    ValidIndices, \
                    U_AllEnsembles, \
                    V_AllEnsembles, \
                    Arguments['Diffusivity'], \
                    Arguments['SweepAllDirections'])

            # Initialize Inpainted arrays
            FillValue = 999
            EnsembleIndices = range(U_AllEnsembles.shape[0])
            U_AllEnsembles_Inpainted = numpy.ma.empty(U_AllEnsembles.shape, dtype=float, fill_value=FillValue)
            V_AllEnsembles_Inpainted = numpy.ma.empty(V_AllEnsembles.shape, dtype=float, fill_value=FillValue)

            # Multiprocessing
            NumProcessors = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=NumProcessors)

            # Determine chunk size
            ChunkSize = int(U_AllEnsembles.shape[0] / NumProcessors)
            Ratio = 40.0
            ChunkSize = int(ChunkSize / Ratio)
            if ChunkSize > 50:
                ChunkSize = 50
            elif ChunkSize < 5:
                ChunkSize = 5

            # Parallel section
            Progress = 0
            print("Message: Restoring time frames ...")
            sys.stdout.flush()

            # Parallel section
            for EnsembleIndex, U_Inpainted_Masked, V_Inpainted_Masked in pool.imap_unordered(RestoreEnsemblePerProcess_PartialFunct, EnsembleIndices, chunksize=ChunkSize):

                # Set inpainted arrays
                U_AllEnsembles_Inpainted[EnsembleIndex, :] = U_Inpainted_Masked
                V_AllEnsembles_Inpainted[EnsembleIndex, :] = V_Inpainted_Masked

                Progress += 1
                print("Progress: %d/%d" %(Progress, U_AllEnsembles.shape[0]))
                sys.stdout.flush()

            # Get statistics of U inpainted ensembles
            U_AllEnsembles_Inpainted_Stats = UncertaintyQuantification.GetEnsemblesStatistics( \
                    LandIndices, \
                    ValidIndices, \
                    MissingIndicesInOceanInsideHull, \
                    MissingIndicesInOceanOutsideHull, \
                    U_OneTime, \
                    Error_U_OneTime, \
                    U_AllEnsembles_Inpainted, \
                    FillValue)

            # Get statistics of V inpainted ensembles
            V_AllEnsembles_Inpainted_Stats = UncertaintyQuantification.GetEnsemblesStatistics( \
                    LandIndices, \
                    ValidIndices, \
                    MissingIndicesInOceanInsideHull, \
                    MissingIndicesInOceanOutsideHull, \
                    V_OneTime, \
                    Error_V_OneTime, \
                    V_AllEnsembles_Inpainted, \
                    FillValue)

            # Add empty dimension to the begining of arrays dimensions for taking into account of time axis.
            U_AllEnsembles_Inpainted_Stats['CentralEnsemble'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['CentralEnsemble'], axis=0)
            V_AllEnsembles_Inpainted_Stats['CentralEnsemble'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['CentralEnsemble'], axis=0)
            U_AllEnsembles_Inpainted_Stats['Mean'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['Mean'], axis=0)
            V_AllEnsembles_Inpainted_Stats['Mean'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['Mean'], axis=0)
            U_AllEnsembles_Inpainted_Stats['STD'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['STD'], axis=0)
            V_AllEnsembles_Inpainted_Stats['STD'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['STD'], axis=0)
            U_AllEnsembles_Inpainted_Stats['RMSD'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['RMSD'], axis=0)
            V_AllEnsembles_Inpainted_Stats['RMSD'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['RMSD'], axis=0)
            U_AllEnsembles_Inpainted_Stats['NRMSD'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['NRMSD'], axis=0)
            V_AllEnsembles_Inpainted_Stats['ExNMSD'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['ExNMSD'], axis=0)
            U_AllEnsembles_Inpainted_Stats['ExNMSD'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['ExNMSD'], axis=0)
            V_AllEnsembles_Inpainted_Stats['NRMSD'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['NRMSD'], axis=0)
            U_AllEnsembles_Inpainted_Stats['Skewness'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['Skewness'], axis=0)
            V_AllEnsembles_Inpainted_Stats['Skewness'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['Skewness'], axis=0)
            U_AllEnsembles_Inpainted_Stats['ExKurtosis'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['ExKurtosis'], axis=0)
            V_AllEnsembles_Inpainted_Stats['ExKurtosis'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['ExKurtosis'], axis=0)
            U_AllEnsembles_Inpainted_Stats['Entropy'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['Entropy'], axis=0)
            V_AllEnsembles_Inpainted_Stats['Entropy'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['Entropy'], axis=0)
            U_AllEnsembles_Inpainted_Stats['RelativeEntropy'] = numpy.ma.expand_dims(U_AllEnsembles_Inpainted_Stats['RelativeEntropy'], axis=0)
            V_AllEnsembles_Inpainted_Stats['RelativeEntropy'] = numpy.ma.expand_dims(V_AllEnsembles_Inpainted_Stats['RelativeEntropy'], axis=0)
            MaskInfo = numpy.expand_dims(MaskInfo, axis=0)

            if Arguments['Plot'] == True:

                # ----------------
                # 1.1 Plot results
                # ----------------

                UncertaintyQuantification.PlotEnsemblesStatistics( \
                        Longitude, \
                        Latitude, \
                        ValidIndices, \
                        MissingIndicesInOceanInsideHull, \
                        U_OneTime, \
                        V_OneTime, \
                        Error_U_OneTime, \
                        Error_V_OneTime, \
                        U_AllEnsembles_Inpainted, \
                        V_AllEnsembles_Inpainted, \
                        U_AllEnsembles_Inpainted_Stats, \
                        V_AllEnsembles_Inpainted_Stats)

            else:

                # ---------------------------------------
                # 1.2 Write results to netcdf output file
                # ---------------------------------------

                InputOutput.WriteOutputFile( \
                        TimeFrame, 
                        DatetimeObject, \
                        Longitude, \
                        Latitude, \
                        MaskInfo, \
                        U_AllEnsembles_Inpainted_Stats['Mean'], \
                        V_AllEnsembles_Inpainted_Stats['Mean'], \
                        U_AllEnsembles_Inpainted_Stats['STD'], \
                        V_AllEnsembles_Inpainted_Stats['STD'], \
                        FillValue, \
                        FullPathOutputFilenamesList[FileIndex])

        else:

            # --------------------------------
            # 2. Restore With Central Ensemble (use original data, no uncertainty quantification)
            # --------------------------------

            # Create a partial function in order to pass a function with only one argument to the multiprocessor
            RestoreTimeFramePerProcess_PartialFunct = partial( \
                    RestoreTimeFramePerProcess, \
                    Longitude, \
                    Latitude, \
                    LandIndices, \
                    U_AllTimes, \
                    V_AllTimes, \
                    Arguments['Diffusivity'], \
                    Arguments['SweepAllDirections'], \
                    Arguments['Plot'], \
                    Arguments['IncludeLandForHull'], \
                    Arguments['UseConvexHull'], \
                    Arguments['Alpha'])

            # Do not perform uncertainty quantification.
            if Arguments['Plot'] == True:

                # --------------------------
                # 2.1 Plot of one time frame
                # --------------------------

                # Plot only one time frame
                TimeIndices = Arguments['TimeFrame']
                RestoreTimeFramePerProcess_PartialFunct(TimeIndices)

            else:

                # ----------------------------
                # 2.2 Restoration of All Times
                # ----------------------------

                # Do not plot, compute all time frames.

                # Inpaint all time frames
                TimeIndices = range(len(Datetime))

                # Initialize Inpainted arrays
                FillValue = 999
                ArrayShape = (len(TimeIndices), ) + U_AllTimes.shape[1:]
                U_AllTimes_Inpainted = numpy.ma.empty(ArrayShape, dtype=float, fill_value=FillValue)
                V_AllTimes_Inpainted = numpy.ma.empty(ArrayShape, dtype=float, fill_value=FillValue)
                MaskInfo_AllTimes = numpy.ma.empty(ArrayShape, dtype=float, fill_value=FillValue)

                # Multiprocessing
                NumProcessors = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(processes=NumProcessors)

                # Determine chunk size
                ChunkSize = int(len(TimeIndices) / NumProcessors)
                Ratio = 40.0
                ChunkSize = int(ChunkSize / Ratio)
                if ChunkSize > 50:
                    ChunkSize = 50
                elif ChunkSize < 5:
                    ChunkSize = 5

                # Parallel section
                Progress = 0
                print("Message: Restoring time frames ...")
                sys.stdout.flush()

                # Parallel section
                for TimeIndex, U_Inpainted_Masked, V_Inpainted_Masked, MaskInfo in pool.imap_unordered(RestoreTimeFramePerProcess_PartialFunct, TimeIndices, chunksize=ChunkSize):

                    # Set inpainted arrays
                    U_AllTimes_Inpainted[TimeIndex, :] = U_Inpainted_Masked
                    V_AllTimes_Inpainted[TimeIndex, :] = V_Inpainted_Masked
                    MaskInfo_AllTimes[TimeIndex, :] = MaskInfo

                    Progress += 1
                    print("Progress: %d/%d" %(Progress, len(TimeIndices)))
                    sys.stdout.flush()
                
                pool.terminate()

                # None arrays
                U_AllTimes_Inpainted_Error = None
                V_AllTimes_Inpainted_Error = None

                # Write results to netcdf output file
                InputOutput.WriteOutputFile( \
                        TimeIndices, 
                        DatetimeObject, \
                        Longitude, \
                        Latitude, \
                        MaskInfo_AllTimes, \
                        U_AllTimes_Inpainted, \
                        V_AllTimes_Inpainted, \
                        U_AllTimes_Inpainted_Error, \
                        V_AllTimes_Inpainted_Error, \
                        FillValue, \
                        FullPathOutputFilenamesList[FileIndex])

        agg.close()

    # End of loop over files

    # If there are multiple files, zip them are delete (clean) written files
    if Arguments['ProcessMultipleFiles'] == 1:
        FileUtilities.ArchiveMultipleFiles( \
                Arguments['FullPathOutputFilename'], \
                FullPathOutputFilenamesList, \
                InputBaseFilenamesList)

# ===========
# System Main
# ===========

if __name__ == "__main__":

    # Converting all warnings to error
    # warnings.simplefilter('error', UserWarning)
    warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Main function
    main(sys.argv)
