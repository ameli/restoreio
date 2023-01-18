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

from .._plots import plot_results
from .._inpaint import inpaint_missing_points_inside_domain
from .._geography import locate_missing_data, create_mask_info
from ._make_array_masked import make_array_masked

__all__ = ['restore_main_ensemble']


# ==============================
# Restore Time Frame Per Process
# ==============================

def _restore_timeframe_per_process(
        lon,
        lat,
        land_indices,
        U_all_times,
        V_all_times,
        diffusivity,
        sweep_all_directions,
        include_land_for_hull,
        convex_hull,
        alpha,
        plot,
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
    # domain. Note: In the following line, all indices outputs are array of the
    # size (N, a), where the first column are latitude indices (not longitude)
    # and the second column indices are longitude indices (not latitude)
    all_missing_indices_in_ocean, missing_indices_in_ocean_inside_hull, \
        missing_indices_in_ocean_outside_hull, valid_indices, \
        hull_points_coord_list = locate_missing_data(
                lon,
                lat,
                land_indices,
                U_original,
                include_land_for_hull,
                convex_hull,
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

    # Use the inpainted point of missing points ONLY inside the domain to
    # restore the data
    U_inpainted, V_inpainted = \
        inpaint_missing_points_inside_domain(
                all_missing_indices_in_ocean,
                missing_indices_in_ocean_inside_hull,
                missing_indices_in_ocean_outside_hull,
                land_indices,
                valid_indices,
                U_original,
                V_original,
                diffusivity,
                sweep_all_directions)

    # Output indices for plotting the grid
    plot_data = {}
    if plot:
        plot_data = {
            'all_missing_indices_in_ocean': all_missing_indices_in_ocean,
            'missing_indices_in_ocean_inside_hull':
                missing_indices_in_ocean_inside_hull,
            'missing_indices_in_ocean_outside_hull':
                missing_indices_in_ocean_outside_hull,
            'valid_indices': valid_indices,
            'hull_points_coord_list': hull_points_coord_list,
            'U_original': U_original,
            'V_original': V_original,
            'U_inpainted': U_inpainted,
            'V_inpainted': V_inpainted,
        }

    return time_index, U_inpainted, V_inpainted, mask_info, plot_data


# =====================
# Restore Main Ensemble
# =====================

def restore_main_ensemble(
        diffusivity,
        sweep,
        timeframe,
        fill_coast,
        alpha,
        convex_hull,
        datetime,
        lon,
        lat,
        land_indices,
        U_all_times,
        V_all_times,
        fill_value,
        plot):
    """
    Restore the given data (central ensemble).

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

    # Create a partial function in order to pass a function with only
    # one argument to the multiprocessor
    restore_timeframe_per_process_partial_func = partial(
            _restore_timeframe_per_process, lon, lat, land_indices,
            U_all_times, V_all_times, diffusivity, sweep, fill_coast,
            convex_hull, alpha, plot)

    # Restore one or all time frames
    if timeframe is not None:
        # Restore only one time frame
        time_indices = [timeframe]
    else:
        # Inpaint all time frames
        time_indices = range(len(datetime))

    # Initialize Inpainted arrays
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
    _plot_data = {}
    progress = 0
    print("Message: Restoring time frames ...")
    sys.stdout.flush()

    if plot is True:
        if timeframe is not None:
            plot_time_index = timeframe
        else:
            # If no timeframe is specified, use the last time for plot
            plot_time_index = time_indices[-1]

    # Parallel section
    for time_index, U_inpainted, V_inpainted, \
            mask_info, plot_data in pool.imap_unordered(
                    restore_timeframe_per_process_partial_func,
                    time_indices, chunksize=chunk_size):

        # Set index to zero when restoring a single time frame
        if timeframe is not None:
            array_time_index = 0
        else:
            array_time_index = time_index

        # Store plot_data for one time frame to be plotted later.
        if plot is True:
            if time_index == plot_time_index:
                _plot_data = plot_data

        # Set inpainted arrays
        U_all_times_inpainted[array_time_index, :] = U_inpainted
        V_all_times_inpainted[array_time_index, :] = V_inpainted
        mask_info_all_times[array_time_index, :] = mask_info

        progress += 1
        print("Progress: %d/%d" % (progress, len(time_indices)))
        sys.stdout.flush()

    pool.terminate()

    # Plotting a single time frame
    if plot is True:

        print("Plotting timeframe: %d ..." % plot_time_index)
        plot_results(
                lon,
                lat,
                land_indices,
                _plot_data['all_missing_indices_in_ocean'],
                _plot_data['missing_indices_in_ocean_inside_hull'],
                _plot_data['missing_indices_in_ocean_outside_hull'],
                _plot_data['valid_indices'],
                _plot_data['hull_points_coord_list'],
                _plot_data['U_original'],
                _plot_data['V_original'],
                _plot_data['U_inpainted'],
                _plot_data['V_inpainted'],
                save=True)

    # None arrays
    U_all_times_inpainted_error = None
    V_all_times_inpainted_error = None

    return time_indices, U_all_times_inpainted, V_all_times_inpainted, \
        U_all_times_inpainted_error, V_all_times_inpainted_error, \
        mask_info_all_times
