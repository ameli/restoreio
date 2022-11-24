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
import warnings

from restoreio.parser import parse_arguments
from restoreio.input_output import load_dataset, load_variables, \
         write_output_file
from restoreio.plots import plot_results
from restoreio.image import inpaint_all_missing_points, \
        restore_missing_points_inside_domain
from restoreio.geography import detect_land_ocean, locate_missing_data, \
        create_mask_info
from restoreio.uncertainty_quant import generate_image_ensembles, \
        get_ensembles_stat, plot_ensembles_stat
from restoreio.file_utilities import get_fullpath_input_filenames_list, \
        get_fullpath_output_filenames_list, archive_multiple_files


# ==========================
# Refine Grid By Adding mask
# ==========================

def refine_grid_by_adding_mask(
        refinement_level,
        data_lon,
        data_lat,
        data_U_all_times,
        data_V_all_times):
    """
    Increases the size of grid by the factor of refinement_level.
    The extra points on the grid will be numpy.ma.mask.

    Note that this does NOT refine the data. Rather, this just increases the
    size of grid. That is, between each two points we introduce a few grid
    points and we mask them. By masking these new points we will tend to
    restore them later.
    """

    # No refinement for level 1
    if refinement_level == 1:
        return data_lon, data_lat, data_U_all_times, data_V_all_times

    # lon
    lon = numpy.zeros(refinement_level*(data_lon.size-1)+1, dtype=float)
    for i in range(data_lon.size):

        # Data points
        lon[refinement_level*i] = data_lon[i]

        # Fill in extra points
        if i < data_lon.size - 1:
            for j in range(1, refinement_level):
                weight = float(j)/float(refinement_level)
                lon[refinement_level*i+j] = ((1.0-weight) * data_lon[i]) + \
                    (weight * data_lon[i+1])

    # lat
    lat = numpy.zeros(refinement_level*(data_lat.size-1)+1, dtype=float)
    for i in range(data_lat.size):

        # Data points
        lat[refinement_level*i] = data_lat[i]

        # Fill in extra points
        if i < data_lat.size - 1:
            for j in range(1, refinement_level):
                weight = float(j)/float(refinement_level)
                lat[refinement_level*i+j] = ((1.0-weight) * data_lat[i]) + \
                    (weight * data_lat[i+1])

    # East Velocity
    U_all_times = numpy.ma.masked_all(
            (data_U_all_times.shape[0],
             refinement_level*(data_U_all_times.shape[1]-1)+1,
             refinement_level*(data_U_all_times.shape[2]-1)+1),
            dtype=numpy.float64)

    U_all_times[:, ::refinement_level, ::refinement_level] = \
        data_U_all_times[:, :, :]

    # North Velocity
    V_all_times = numpy.ma.masked_all(
            (data_V_all_times.shape[0],
             refinement_level*(data_V_all_times.shape[1]-1)+1,
             refinement_level*(data_V_all_times.shape[2]-1)+1),
            dtype=numpy.float64)

    V_all_times[:, ::refinement_level, ::refinement_level] = \
        data_V_all_times[:, :, :]

    return lon, lat, U_all_times, V_all_times


# ============================
# Refine Grid By Interpolation
# ============================

def refine_grid_by_interpolation(
        refinement_level,
        data_lon,
        data_lat,
        data_U_all_times,
        data_V_all_times):
    """
    Refines grid by means of interpolation. Note that this actually
    interpolates the data which is in contrast to the previous function:
    "refine_grid_by_adding_mask"
    """

    # TODO
    print('hi')


# =================
# Make Array masked
# =================

def make_array_masked(array):
    """
    Often the array is not masked, but has nan or inf values. This function
    creates a masked array and mask nan and inf.

    Input:
        - array: is a 2D numpy array.
    Output:
        - array: is a 2D numpy.ma array.

    Note: array should be numpy object not netCDF object. So if you have a
          netCDF object, pass its numpy array with array[:] into this function.
    """

    if (not hasattr(array, 'mask')) or (array.mask.size == 1):
        if numpy.isnan(array).any() or numpy.isinf(array).any():
            # This array is not masked. Make a mask based no nan and inf
            mask_nan = numpy.isnan(array)
            mask_inf = numpy.isinf(array)
            mask = numpy.logical_or(mask_nan, mask_inf)
            array = numpy.ma.masked_array(array, mask=mask)
    else:
        # This array is masked. But check if any non-masked value is nan or inf
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array.mask[i, j] is False:
                    if numpy.isnan(array[i, j]) or numpy.isinf(array[i, j]):
                        array.mask[i, j] = True

    return array


# ==============================
# Restore Time Frame Per Process
# ==============================

def restore_timeframe_per_process(
        lon,
        lat,
        land_indices,
        U_all_times,
        V_all_times,
        diffusivity,
        sweep_all_directions,
        plot,
        include_land_for_hull,
        use_convex_hull,
        alpha,
        time_index):
    """
    Do all calculations for one time frame. This function is called from
    multiprocessing object. Each time frame is dispatched to a processor.
    """

    # Get one time frame of U and V velocities.
    U_original = U_all_times[time_index, :]
    V_original = V_all_times[time_index, :]

    # Make sure arrays are masked arrays
    U_original = make_array_masked(U_original)
    V_original = make_array_masked(V_original)

    # Find indices of valid points, missing points inside and outside the
    # domain. Note: In the following line, all indices outputs are Nx2, where
    # the first column are latitude indices (not longitude) and the second
    # column indices are longitude indices (not latitude)
    all_missing_indices_in_ocean, missing_indices_in_ocean_inside_hull, \
        missing_indices_in_ocean_outside_hull, valid_indices, \
        HullPointsCoordinatesList = locate_missing_data(
                lon,
                lat,
                land_indices,
                U_original,
                include_land_for_hull,
                use_convex_hull,
                alpha)

    # Create mask Info
    mask_info = create_mask_info(
            U_original,
            land_indices,
            missing_indices_in_ocean_inside_hull,
            missing_indices_in_ocean_outside_hull,
            valid_indices)

    # Set data on land to be zero (Note: This should be done after finding the
    # convex hull)
    if hasattr(U_original, 'mask'):
        U_original.unshare_mask()

    if hasattr(V_original, 'mask'):
        V_original.unshare_mask()

    if numpy.any(numpy.isnan(land_indices)) is False:
        for LandId in range(land_indices.shape[0]):
            U_original[land_indices[LandId, 0], land_indices[LandId, 1]] = 0.0
            V_original[land_indices[LandId, 0], land_indices[LandId, 1]] = 0.0

    # Inpaint all missing points including inside and outside the domain
    U_inpainted_all_missing_points, V_inpainted_all_missing_points = \
        inpaint_all_missing_points(
                all_missing_indices_in_ocean,
                land_indices,
                valid_indices,
                U_original,
                V_original,
                diffusivity,
                sweep_all_directions)

    # Use the inpainted point of missing points ONLY inside the domain to
    # restore the data
    U_inpainted_masked, V_inpainted_masked = \
        restore_missing_points_inside_domain(
                missing_indices_in_ocean_inside_hull,
                missing_indices_in_ocean_outside_hull,
                land_indices,
                U_original,
                V_original,
                U_inpainted_all_missing_points,
                V_inpainted_all_missing_points)

    # Plot the grid and inpainted results
    if plot is True:
        print("Plotting timeframe: %d ..." % time_index)

        plot_results(
                lon,
                lat,
                U_original,
                V_original,
                U_inpainted_masked,
                V_inpainted_masked,
                all_missing_indices_in_ocean,
                missing_indices_in_ocean_inside_hull,
                missing_indices_in_ocean_outside_hull,
                valid_indices,
                land_indices,
                HullPointsCoordinatesList)

        return

    return time_index, U_inpainted_masked, V_inpainted_masked, mask_info


# ============================
# Restore Ensemble Per Process
# ============================

def restore_ensemble_per_process(
        land_indices,
        all_missing_indices_in_ocean,
        missing_indices_in_ocean_inside_hull,
        missing_indices_in_ocean_outside_hull,
        valid_indices,
        U_all_ensembles,
        V_all_ensembles,
        diffusivity,
        sweep_all_directions,
        ensemble_index):
    """
    Do all calculations for one time frame. This function is called from
    multiprocessing object. Each time frame is dispatched to a processor.
    """

    # Get one ensemble
    U_Ensemble = U_all_ensembles[ensemble_index, :, :]
    V_Ensemble = V_all_ensembles[ensemble_index, :, :]

    # Set data on land to be zero (Note: This should be done after finding the
    # convex hull)
    if hasattr(U_Ensemble, 'mask'):
        U_Ensemble.unshare_mask()

    if hasattr(V_Ensemble, 'mask'):
        V_Ensemble.unshare_mask()

    if numpy.any(numpy.isnan(land_indices)) is False:
        for LandId in range(land_indices.shape[0]):
            U_Ensemble[land_indices[LandId, 0], land_indices[LandId, 1]] = 0.0
            V_Ensemble[land_indices[LandId, 0], land_indices[LandId, 1]] = 0.0

    # Inpaint all missing points including inside and outside the domain
    U_inpainted_all_missing_points, V_inpainted_all_missing_points = \
        inpaint_all_missing_points(
                all_missing_indices_in_ocean,
                land_indices,
                valid_indices,
                U_Ensemble,
                V_Ensemble,
                diffusivity,
                sweep_all_directions)

    # Use the inpainted point of missing points ONLY inside the domain to
    # restore the data
    U_inpainted_masked, V_inpainted_masked = \
        restore_missing_points_inside_domain(
                missing_indices_in_ocean_inside_hull,
                missing_indices_in_ocean_outside_hull,
                land_indices,
                U_Ensemble,
                V_Ensemble,
                U_inpainted_all_missing_points,
                V_inpainted_all_missing_points)

    return ensemble_index, U_inpainted_masked, V_inpainted_masked


# =======
# Restore
# =======

def restore(argv):
    """
    These parameters should be set for the opencv.inpaint method:

    diffusivity:
        (Default = 20) The diffusion coefficient

    sweep_all_directions:
        (Default to = True) If set to True, the inpaint is performed 4 times on
        the flipped left/right and up/down of the image.

    Notes on parallelization:
        - We have used multiprocessing.Pool.imap_unordered. Other options are
          apply, apply_async, map, imap, etc.
        - The imap_unordered can only accept functions with one argument, where
          the argument is the iterator of the parallelization.
        - In order to pass a multi-argument function, we have used
          functool.partial.
        - The imap_unordered distributes all tasks to processes by a
          chunk_size. Meaning that each process is assigned a chunk size number
          of iterators of tasks to do, before loads the next chunk size. By
          default the chunk size is 1. This causes many function calls and
          slows down the parallelization. By setting the chunk_size=100, each
          process is assigned 100 iteration, with only 1 function call. So if
          we have 4 processors, each one perform 100 tasks. After each process
          is done with a 100 task, it loads another 100 task from the pool of
          tasks in an unordered manner. The "map" in imap_unordered ensures
          that all processes are assigned a task without having an idle
          process.
    """

    # Parse arguments
    arguments = parse_arguments(argv)

    # Get list of all separate input files to process
    fullpath_input_filenames_list, input_base_filenames_list = \
        get_fullpath_input_filenames_list(
                arguments['fullpath_input_filename'],
                arguments['process_multiple_files'],
                arguments['multiple_files_min_iterator_string'],
                arguments['multiple_files_max_iterator_string'])

    # Get the list of all output files to be written to
    fullpath_output_filenames_list = get_fullpath_output_filenames_list(
            arguments['fullpath_output_filename'],
            arguments['process_multiple_files'],
            arguments['multiple_files_min_iterator_string'],
            arguments['multiple_files_max_iterator_string'])

    NumberOfFiles = len(fullpath_input_filenames_list)

    # Iterate over multiple separate files
    for file_index in range(NumberOfFiles):

        # Open file
        agg = load_dataset(fullpath_input_filenames_list[file_index])

        # Load variables
        datetime_obj, lon_obj, lat_obj, east_vel_obj, north_vel_obj, \
            east_vel_error_obj, north_vel_error_obj = load_variables(agg)

        # To not issue error/warning when data has nan
        numpy.warnings.filterwarnings('ignore')

        # Get arrays
        datetime = datetime_obj[:]
        # data_lon = lon_obj[:]
        # data_lat = lat_obj[:]
        # data_U_all_times = east_vel_obj[:]
        # data_V_all_times = north_vel_obj[:]
        lon = lon_obj[:]
        lat = lat_obj[:]
        U_all_times = east_vel_obj[:]
        V_all_times = north_vel_obj[:]

        # Refinement
        # lon, lat, U_all_times, V_all_times = refine_grid_by_adding_mask(
        #         arguments['refinement_level'],
        #         data_lon,
        #         data_lat,
        #         data_U_all_times,
        #         data_V_all_times)

        # Determine the land
        land_indices, ocean_indices = detect_land_ocean(
                lon, lat, arguments['exclude_land_from_ocean'])

        # If plotting, remove these files:
        if arguments['plot'] is True:
            # Remove ~/.Xauthority and ~/.ICEauthority
            import os.path
            HomeDir = os.path.expanduser("~")
            if os.path.isfile(HomeDir+'/.Xauthority'):
                os.remove(HomeDir+'/.Xauthority')
            if os.path.isfile(HomeDir+'/.ICEauthority'):
                os.remove(HomeDir+'/.ICEauthority')

        # Check whether to perform uncertainty quantification or not
        if arguments['uncertainty_quantification'] is True:

            # -----------------------------
            # 1. Uncertainty Quantification
            # -----------------------------

            # Time frame
            timeframe = arguments['timeframe']
            if timeframe >= U_all_times.shape[0]:
                raise ValueError('Time frame is out of bound.')
            elif timeframe < 0:
                timeframe = -1

            # Get one time frame of velocities
            U_one_time = make_array_masked(U_all_times[timeframe, :, :])
            V_one_time = make_array_masked(V_all_times[timeframe, :, :])

            # Check if data has errors of velocities variable
            if (east_vel_error_obj is None):
                raise ValueError('Input netCDF data does not have East ' +
                                 'Velocity error, which is needed for ' +
                                 'uncertainty quantification.')
            if (north_vel_error_obj is None):
                raise ValueError('Input netCDF data does not have North ' +
                                 'Velocity error, which is needed for ' +
                                 'uncertainty quantification.')

            # Make sure arrays are masked arrays
            error_U_one_time = make_array_masked(
                    east_vel_error_obj[timeframe, :, :])
            error_V_one_time = make_array_masked(
                    north_vel_error_obj[timeframe, :, :])

            # scale Errors (TODO)
            scale = 0.08  # m/s
            error_U_one_time *= scale
            error_V_one_time *= scale

            # Errors are usually squared. Take square root
            # error_U_one_time = numpy.ma.sqrt(error_U_one_time)
            # error_V_one_time = numpy.ma.sqrt(error_V_one_time)

            # Find indices of valid points, missing points inside and outside
            # the domain. Note: In the following line, all indices outputs are
            # Nx2, where the first column are latitude indices (not longitude)
            # and the second column indices are longitude indices (not
            # latitude)
            all_missing_indices_in_ocean, \
                missing_indices_in_ocean_inside_hull, \
                missing_indices_in_ocean_outside_hull, valid_indices, \
                HullPointsCoordinatesList = \
                locate_missing_data(
                            lon,
                            lat,
                            land_indices,
                            U_one_time,
                            arguments['include_land_for_hull'],
                            arguments['use_convex_hull'],
                            arguments['alpha'])

            # Create mask Info
            mask_info = create_mask_info(
                    U_one_time,
                    land_indices,
                    missing_indices_in_ocean_inside_hull,
                    missing_indices_in_ocean_outside_hull,
                    valid_indices)

            # Generate Ensembles (lon and lat are not needed, but only used for
            # plots if uncommented)
            num_modes = None  # None makes num_modes to use max possible modes
            U_all_ensembles = generate_image_ensembles(
                    lon, lat, U_one_time, error_U_one_time, valid_indices,
                    arguments['num_ensembles'], num_modes)
            V_all_ensembles = generate_image_ensembles(
                    lon, lat, V_one_time, error_V_one_time, valid_indices,
                    arguments['num_ensembles'], num_modes)

            # Create a partial function in order to pass a function with only
            # one argument to the multiprocessor
            restore_ensemble_per_process_partial_func = partial(
                    restore_ensemble_per_process,
                    land_indices,
                    all_missing_indices_in_ocean,
                    missing_indices_in_ocean_inside_hull,
                    missing_indices_in_ocean_outside_hull,
                    valid_indices,
                    U_all_ensembles,
                    V_all_ensembles,
                    arguments['diffusivity'],
                    arguments['sweep_all_directions'])

            # Initialize Inpainted arrays
            fill_value = 999
            EnsembleIndices = range(U_all_ensembles.shape[0])
            U_all_ensembles_inpainted = numpy.ma.empty(
                    U_all_ensembles.shape, dtype=float, fill_value=fill_value)
            V_all_ensembles_inpainted = numpy.ma.empty(
                    V_all_ensembles.shape, dtype=float, fill_value=fill_value)

            # Multiprocessing
            num_processors = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=num_processors)

            # Determine chunk size
            chunk_size = int(U_all_ensembles.shape[0] / num_processors)
            ratio = 40.0
            chunk_size = int(chunk_size / ratio)
            if chunk_size > 50:
                chunk_size = 50
            elif chunk_size < 5:
                chunk_size = 5

            # Parallel section
            Progress = 0
            print("Message: Restoring time frames ...")
            sys.stdout.flush()

            # Parallel section
            for ensemble_index, U_inpainted_masked, V_inpainted_masked in \
                    pool.imap_unordered(
                            restore_ensemble_per_process_partial_func,
                            EnsembleIndices,
                            chunksize=chunk_size):

                # Set inpainted arrays
                U_all_ensembles_inpainted[ensemble_index, :] = \
                        U_inpainted_masked
                V_all_ensembles_inpainted[ensemble_index, :] = \
                    V_inpainted_masked

                Progress += 1
                print("Progress: %d/%d" % (Progress, U_all_ensembles.shape[0]))
                sys.stdout.flush()

            # Get statistics of U inpainted ensembles
            U_all_ensembles_inpainted_stats = get_ensembles_stat(
                    land_indices,
                    valid_indices,
                    missing_indices_in_ocean_inside_hull,
                    missing_indices_in_ocean_outside_hull,
                    U_one_time,
                    error_U_one_time,
                    U_all_ensembles_inpainted,
                    fill_value)

            # Get statistics of V inpainted ensembles
            V_all_ensembles_inpainted_stats = get_ensembles_stat(
                    land_indices,
                    valid_indices,
                    missing_indices_in_ocean_inside_hull,
                    missing_indices_in_ocean_outside_hull,
                    V_one_time,
                    error_V_one_time,
                    V_all_ensembles_inpainted,
                    fill_value)

            # Add empty dimension to the beginning of arrays dimensions for
            # taking into account of time axis.
            U_all_ensembles_inpainted_stats['CentralEnsemble'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['CentralEnsemble'],
                        axis=0)
            V_all_ensembles_inpainted_stats['CentralEnsemble'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['CentralEnsemble'],
                        axis=0)
            U_all_ensembles_inpainted_stats['Mean'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['Mean'], axis=0)
            V_all_ensembles_inpainted_stats['Mean'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['Mean'], axis=0)
            U_all_ensembles_inpainted_stats['STD'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['STD'], axis=0)
            V_all_ensembles_inpainted_stats['STD'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['STD'], axis=0)
            U_all_ensembles_inpainted_stats['RMSD'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['RMSD'], axis=0)
            V_all_ensembles_inpainted_stats['RMSD'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['RMSD'], axis=0)
            U_all_ensembles_inpainted_stats['NRMSD'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['NRMSD'], axis=0)
            V_all_ensembles_inpainted_stats['ExNMSD'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['ExNMSD'], axis=0)
            U_all_ensembles_inpainted_stats['ExNMSD'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['ExNMSD'], axis=0)
            V_all_ensembles_inpainted_stats['NRMSD'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['NRMSD'], axis=0)
            U_all_ensembles_inpainted_stats['Skewness'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['Skewness'], axis=0)
            V_all_ensembles_inpainted_stats['Skewness'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['Skewness'], axis=0)
            U_all_ensembles_inpainted_stats['ExKurtosis'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['ExKurtosis'], axis=0)
            V_all_ensembles_inpainted_stats['ExKurtosis'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['ExKurtosis'], axis=0)
            U_all_ensembles_inpainted_stats['Entropy'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['Entropy'], axis=0)
            V_all_ensembles_inpainted_stats['Entropy'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['Entropy'], axis=0)
            U_all_ensembles_inpainted_stats['RelativeEntropy'] = \
                numpy.ma.expand_dims(
                        U_all_ensembles_inpainted_stats['RelativeEntropy'],
                        axis=0)
            V_all_ensembles_inpainted_stats['RelativeEntropy'] = \
                numpy.ma.expand_dims(
                        V_all_ensembles_inpainted_stats['RelativeEntropy'],
                        axis=0)
            mask_info = numpy.expand_dims(mask_info, axis=0)

            if arguments['plot'] is True:

                # ----------------
                # 1.1 Plot results
                # ----------------

                plot_ensembles_stat(
                        lon,
                        lat,
                        valid_indices,
                        missing_indices_in_ocean_inside_hull,
                        U_one_time,
                        V_one_time,
                        error_U_one_time,
                        error_V_one_time,
                        U_all_ensembles_inpainted,
                        V_all_ensembles_inpainted,
                        U_all_ensembles_inpainted_stats,
                        V_all_ensembles_inpainted_stats)

            else:

                # ---------------------------------------
                # 1.2 Write results to netcdf output file
                # ---------------------------------------

                write_output_file(
                        timeframe,
                        datetime_obj,
                        lon,
                        lat,
                        mask_info,
                        U_all_ensembles_inpainted_stats['Mean'],
                        V_all_ensembles_inpainted_stats['Mean'],
                        U_all_ensembles_inpainted_stats['STD'],
                        V_all_ensembles_inpainted_stats['STD'],
                        fill_value,
                        fullpath_output_filenames_list[file_index])

        else:

            # --------------------------------
            # 2. Restore With Central Ensemble
            # (use original data, no uncertainty quantification)
            # --------------------------------

            # Create a partial function in order to pass a function with only
            # one argument to the multiprocessor
            restore_timeframe_per_process_partial_func = partial(
                    restore_timeframe_per_process,
                    lon,
                    lat,
                    land_indices,
                    U_all_times,
                    V_all_times,
                    arguments['diffusivity'],
                    arguments['sweep_all_directions'],
                    arguments['plot'],
                    arguments['include_land_for_hull'],
                    arguments['use_convex_hull'],
                    arguments['alpha'])

            # Do not perform uncertainty quantification.
            if arguments['plot'] is True:

                # --------------------------
                # 2.1 Plot of one time frame
                # --------------------------

                # Plot only one time frame
                time_indices = arguments['timeframe']
                restore_timeframe_per_process_partial_func(time_indices)

            else:

                # ----------------------------
                # 2.2 Restoration of All Times
                # ----------------------------

                # Do not plot, compute all time frames.

                # Inpaint all time frames
                time_indices = range(len(datetime))

                # Initialize Inpainted arrays
                fill_value = 999
                array_shape = (len(time_indices), ) + U_all_times.shape[1:]
                U_all_times_inpainted = numpy.ma.empty(array_shape,
                                                       dtype=float,
                                                       fill_value=fill_value)
                V_all_times_inpainted = numpy.ma.empty(array_shape,
                                                       dtype=float,
                                                       fill_value=fill_value)
                mask_info_all_times = numpy.ma.empty(array_shape,
                                                     dtype=float,
                                                     fill_value=fill_value)

                # Multiprocessing
                num_processors = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(processes=num_processors)

                # Determine chunk size
                chunk_size = int(len(time_indices) / num_processors)
                ratio = 40.0
                chunk_size = int(chunk_size / ratio)
                if chunk_size > 50:
                    chunk_size = 50
                elif chunk_size < 5:
                    chunk_size = 5

                # Parallel section
                Progress = 0
                print("Message: Restoring time frames ...")
                sys.stdout.flush()

                # Parallel section
                for time_index, U_inpainted_masked, V_inpainted_masked, \
                        mask_info in pool.imap_unordered(
                                restore_timeframe_per_process_partial_func,
                                time_indices, chunksize=chunk_size):

                    # Set inpainted arrays
                    U_all_times_inpainted[time_index, :] = U_inpainted_masked
                    V_all_times_inpainted[time_index, :] = V_inpainted_masked
                    mask_info_all_times[time_index, :] = mask_info

                    Progress += 1
                    print("Progress: %d/%d" % (Progress, len(time_indices)))
                    sys.stdout.flush()

                pool.terminate()

                # None arrays
                U_all_times_inpainted_error = None
                V_all_times_inpainted_error = None

                # Write results to netcdf output file
                write_output_file(
                        time_indices,
                        datetime_obj,
                        lon,
                        lat,
                        mask_info_all_times,
                        U_all_times_inpainted,
                        V_all_times_inpainted,
                        U_all_times_inpainted_error,
                        V_all_times_inpainted_error,
                        fill_value,
                        fullpath_output_filenames_list[file_index])

        agg.close()

    # End of loop over files

    # If there are multiple files, zip them are delete (clean) written files
    if arguments['process_multiple_files'] == 1:
        archive_multiple_files(
                arguments['fullpath_output_filename'],
                fullpath_output_filenames_list,
                input_base_filenames_list)


# ====
# Main
# ====

def main():
    """
    Main function to be called when this script is called as an executable.
    """

    # Converting all warnings to error
    # warnings.simplefilter('error', UserWarning)
    warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Main function
    restore(sys.argv)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    main()
