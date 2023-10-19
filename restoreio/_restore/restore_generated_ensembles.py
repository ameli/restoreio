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

from .._inpaint import inpaint_missing_points_inside_domain
from .._geography import locate_missing_data, create_mask_info
from .._uncertainty_quant import generate_image_ensembles, \
        get_ensembles_stat
from .._plots_uq import plot_ensembles_stat, plot_convergence
from ._make_array_masked import make_array_masked
from .._server_utils import terminate_with_error

__all__ = ['restore_generated_ensembles']


# ============================
# Restore Ensemble Per Process
# ============================

def _restore_ensemble_per_process(
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

    # Use the inpainted point of missing points ONLY inside the domain to
    # restore the data
    U_inpainted, V_inpainted = \
        inpaint_missing_points_inside_domain(
                all_missing_indices_in_ocean,
                missing_indices_in_ocean_inside_hull,
                missing_indices_in_ocean_outside_hull,
                land_indices,
                valid_indices,
                U_Ensemble,
                V_Ensemble,
                diffusivity,
                sweep_all_directions)

    return ensemble_index, U_inpainted, V_inpainted


# ==========================
# Restore Generated Ensemble
# ==========================

def restore_generated_ensembles(
        diffusivity,
        sweep,
        fill_coast,
        alpha,
        convex_hull,
        num_samples,
        ratio_num_modes,
        kernel_width,
        scale_error,
        lon,
        lat,
        land_indices,
        U_all_times,
        V_all_times,
        U_error_all_times,
        V_error_all_times,
        fill_value,
        file_index,
        num_files,
        plot=False,
        save=True,
        verbose=False):
    """
    Restore all generated ensemble, and take their mean and std.

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

    Note: despite in the notations in this function U_all_times and V_all_times
    is used, these velocities and their error arrays indeed have only one time
    since in uncertainty quantification we limited the time index to only one.
    """

    # In UQ method, we only process one time index.
    time_index = 0

    # Get one time frame of velocities
    U_one_time = make_array_masked(U_all_times[time_index, :, :], fill_value)
    V_one_time = make_array_masked(V_all_times[time_index, :, :], fill_value)

    # Check if data has errors of velocities variable
    if (U_error_all_times is None):
        terminate_with_error('Input netCDF data does not have East ' +
                             'Velocity error, which is needed for ' +
                             'uncertainty quantification.')

    if (V_error_all_times is None):
        terminate_with_error('Input netCDF data does not have North ' +
                             'Velocity error, which is needed for ' +
                             'uncertainty quantification.')

    # Make sure arrays are masked arrays
    U_error_one_time = make_array_masked(U_error_all_times[time_index, :, :],
                                         fill_value)
    V_error_one_time = make_array_masked(V_error_all_times[time_index, :, :],
                                         fill_value)

    # scale Errors
    scale = scale_error  # in m/s unit
    U_error_one_time *= scale
    V_error_one_time *= scale

    # Errors are usually squared. Take square root
    # U_error_one_time = numpy.ma.sqrt(U_error_one_time)
    # V_error_one_time = numpy.ma.sqrt(V_error_one_time)

    # Find indices of valid points, missing points inside and outside
    # the domain. Note: In the following line, all indices outputs are
    # Nx2, where the first column are latitude indices (not longitude)
    # and the second column indices are longitude indices (not
    # latitude)
    all_missing_indices_in_ocean, \
        missing_indices_in_ocean_inside_hull, \
        missing_indices_in_ocean_outside_hull, valid_indices, \
        hull_points_coord_list = \
        locate_missing_data(
                    lon,
                    lat,
                    land_indices,
                    U_one_time,
                    fill_coast,
                    convex_hull,
                    alpha,
                    verbose=verbose)

    # Create mask Info
    mask_info = create_mask_info(
            U_one_time,
            land_indices,
            missing_indices_in_ocean_inside_hull,
            missing_indices_in_ocean_outside_hull,
            valid_indices)

    # Generate Ensemble (lon and lat are not needed, but only used for
    # plots if uncommented)
    U_all_ensembles = generate_image_ensembles(
            lon, lat, U_one_time, U_error_one_time, valid_indices,
            missing_indices_in_ocean_inside_hull, num_samples,
            ratio_num_modes, kernel_width, 'east', plot=plot, save=save,
            verbose=verbose)
    V_all_ensembles = generate_image_ensembles(
            lon, lat, V_one_time, V_error_one_time, valid_indices,
            missing_indices_in_ocean_inside_hull, num_samples,
            ratio_num_modes, kernel_width, 'north', plot=plot, save=save,
            verbose=verbose)

    # Create a partial function in order to pass a function with only
    # one argument to the multiprocessor
    restore_ensemble_per_process_partial_func = partial(
            _restore_ensemble_per_process,
            land_indices,
            all_missing_indices_in_ocean,
            missing_indices_in_ocean_inside_hull,
            missing_indices_in_ocean_outside_hull,
            valid_indices,
            U_all_ensembles,
            V_all_ensembles,
            diffusivity,
            sweep)

    # Initialize Inpainted arrays
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
    num_gen_ensembles = U_all_ensembles.shape[0]
    progress = file_index * num_gen_ensembles
    total_progress = num_files * num_gen_ensembles
    if verbose:
        print("Message: Restoring ensembles ...")
        sys.stdout.flush()

    # Parallel section
    for ensemble_index, U_inpainted, V_inpainted in \
            pool.imap_unordered(
                    restore_ensemble_per_process_partial_func,
                    EnsembleIndices,
                    chunksize=chunk_size):

        # Set inpainted arrays
        U_all_ensembles_inpainted[ensemble_index, :] = \
                U_inpainted
        V_all_ensembles_inpainted[ensemble_index, :] = \
            V_inpainted

        progress += 1
        if verbose:
            print("Progress: %d/%d" % (progress, total_progress))
            sys.stdout.flush()

    # Get statistics of U inpainted ensembles
    U_all_ensembles_inpainted_stats = get_ensembles_stat(
            land_indices,
            valid_indices,
            missing_indices_in_ocean_inside_hull,
            missing_indices_in_ocean_outside_hull,
            U_one_time,
            U_error_one_time,
            U_all_ensembles_inpainted,
            fill_value)

    # Get statistics of V inpainted ensembles
    V_all_ensembles_inpainted_stats = get_ensembles_stat(
            land_indices,
            valid_indices,
            missing_indices_in_ocean_inside_hull,
            missing_indices_in_ocean_outside_hull,
            V_one_time,
            V_error_one_time,
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
    U_all_ensembles_inpainted_stats['JSdistance'] = \
        numpy.ma.expand_dims(
                U_all_ensembles_inpainted_stats['JSdistance'],
                axis=0)
    V_all_ensembles_inpainted_stats['JSdistance'] = \
        numpy.ma.expand_dims(
                V_all_ensembles_inpainted_stats['JSdistance'],
                axis=0)
    mask_info = numpy.expand_dims(mask_info, axis=0)

    if plot is True:

        # Plot results
        plot_ensembles_stat(
                lon,
                lat,
                valid_indices,
                missing_indices_in_ocean_inside_hull,
                U_one_time,
                V_one_time,
                U_error_one_time,
                V_error_one_time,
                U_all_ensembles_inpainted,
                V_all_ensembles_inpainted,
                U_all_ensembles_inpainted_stats,
                V_all_ensembles_inpainted_stats,
                save=save,
                verbose=verbose)

    U_all_ensembles_inpainted_mean = U_all_ensembles_inpainted_stats['Mean']
    V_all_ensembles_inpainted_mean = V_all_ensembles_inpainted_stats['Mean']
    U_all_ensembles_inpainted_std = U_all_ensembles_inpainted_stats['STD']
    V_all_ensembles_inpainted_std = V_all_ensembles_inpainted_stats['STD']

    if plot is True:
        plot_convergence(
                missing_indices_in_ocean_inside_hull,
                U_all_ensembles_inpainted, V_all_ensembles_inpainted,
                U_all_ensembles_inpainted_stats,
                V_all_ensembles_inpainted_stats, save=save, verbose=verbose)

    return U_all_ensembles_inpainted, V_all_ensembles_inpainted, \
        U_all_ensembles_inpainted_mean, V_all_ensembles_inpainted_mean, \
        U_all_ensembles_inpainted_std, V_all_ensembles_inpainted_std, mask_info
