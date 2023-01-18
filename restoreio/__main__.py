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
import warnings

from ._parser import parse_arguments
from ._input_output import load_dataset, load_variables, write_output_file
from ._geography import detect_land_ocean
from ._file_utilities import get_fullpath_input_filenames_list, \
        get_fullpath_output_filenames_list, archive_multiple_files
from ._restore import restore_main_ensemble, restore_generated_ensembles
# from restoreio._restore import refine_grid

__all__ = ['restore']


# =================
# process arguments
# =================

def process_arguments(
        detect_land,
        min_file_index,
        max_file_index,
        fill_coast):
    """
    Parses the argument of the executable and obtains the filename.
    """

    # Check include land
    if detect_land == 0:
        fill_coast = False

    # Check Processing multiple file
    if ((min_file_index != '') and (max_file_index != '')):

        if ((min_file_index == '') or (max_file_index == '')):
            raise ValueError('To process multiple files, both min and max ' +
                             'file iterator should be specified.')

    return fill_coast


# =======
# Restore
# =======

def restore(
        input,
        min_file_index='',
        max_file_index='',
        output='',
        diffusivity=20,
        sweep=False,
        detect_land=2,
        fill_coast=False,
        convex_hull=False,
        alpha=20,
        refine_grid=1,
        timeframe=None,
        uncertainty_quant=False,
        num_ensembles=1000,
        num_modes=None,
        kernel_window=5,
        scale_error=0.08,
        plot=False):
    """
    Restore incomplete oceanographic dataset.

    Parameters
    ----------

    input : str
        Input filename. This can be either the path to a local file or the url
        to a remote dataset. The file extension should be ``.nc`` or ``.ncml``.

    min_file_index : str, default=''
        Start file iterator to be used for processing multiple input files. For
        Instance, setting ``input=input``, ``min_file_index=003``, and
        ``max_file_index=012`` means to read the series of input files with
        iterators ``input003.nc``, ``input004.nc``, to ``input012.nc``. If this
        option is used, the option ``max_file_index`` should also be given.

    max_file_index : str, default=''
        Start file iterator to be used for processing multiple input files. For
        Instance, setting ``input=input``, ``min_file_index=003``, and
        ``max_file_index=012`` means to read the series of input files with
        iterators ``input003.nc``, ``input004.nc``, to ``input012.nc``. If this
        option is used, the option ``min_file_index`` should also be given.

    output : str
        Output filename. This can be either the path to a local file or the url
        to a remote dataset. The file extension should be ``.nc`` or ``.ncml``
        only. If no output file is provided, the output filename is constructed
        by adding the word ``_restored`` at the end of the input filename.

    diffusivity : float, default=20
        Diffusivity of the PDE solver (real number). Large number leads to
        diffusion dominant solution. Small numbers leads to advection dominant
        solution.

    sweep : bool, default=False
        Sweeps the image data in all flipped directions. This ensures an even
        solution independent of direction.

    detect_land : int, default=2
        Detect land and exclude it from ocean's missing data points. This
        option should be an integer with the following values:

        * ``0``: Does not detect land from ocean. All land points are assumed
          to be a part of ocean's missing points.
        * ``1``: Detect land. Most accurate, slowest.
        * ``2``: Detect land. Less accurate, fastest (preferred method).
        * ``3``: Detect land. Currently this option is not fully implemented.

    fill_coast : bool, default=False
        Fills the gap the between the data in the ocean and between ocean and
        the coastline. This option is only effective if ``detect_land`` is not
        set to ``0``.

    convex_hull : bool, default=False
        Instead of using the concave hull (alpha shape) around the data points,
        this options uses convex hull of the area around the data points.

    alpha : float, default=20
        The alpha number for alpha shape. If not specified or a negative
        number, this value is computed automatically. This option is only
        relevant to concave shapes. This option is ignored if convex shape is
        used with ``convex_hull=True``.

    refine_grid : int, default=1
        Refines the grid by increasing the number of points on each axis by a
        multiple of a given integer. If this option is set to `1`, no
        refinement is performed. If set to integer n, the number of grid points
        is refined by :math:`n^2` times (that is, :math:`n` times on each
        axis).

    timeframe : int, default=None
        The time frame index in the dataset to process and to plot the
        uncertainty quantification. The index wraps around the total number of
        time frames. For instance, `-1` indicates the last time frame.

    uncertainty_quant : bool, default=False
        Performs uncertainty quantification on the data for the time frame
        given by ``timeframe`` option.

    num_ensembles : int, default=1000
        Number of ensembles used for uncertainty quantification. This option is
        relevant if ``uncertainty_quant`` is set to `True`.

    num_modes : int, default=None
        Number of eigen-modes used for uncertainty quantification (in
        KL-expansion). When not specified, maximum number of possible modes
        will be used, which is the number of ensembles minus one. This option
        is relevant if ``uncertainty_quant`` is set to `True`.

    kernel_window : int, default=5
        Window of the kernel to estimate covariance of data. The window length
        should be given by the integer number of pixels (data points). The
        non-zero extent of the kernel a square area with twice the window
        length in both longitude and latitude directions. This option is
        relevant if ``uncertainty_quant`` is set to `True`.

    scale_error : float, default=0.08
        Scale velocity error of the input data by a factor. Often, the input
        velocity error is the dimensionless GDOP which needs to be scaled by
        the standard deviation of the velocity error to represent the actual
        velocity error. This value scales the error. This option is relevant if
        ``uncertainty_quant`` is set to `True`.

    plot : bool, default=False
        Plots the results. In this case, instead of iterating through all time
        frames, only one time frame (given with option ``timeframe``) is
        restored and plotted. If in addition, the uncertainty quantification is
        enabled (with option ``uncertainty_quant=True``), the statistical
        analysis for the given time frame is also plotted.
    """

    save = True  # Test

    # Check arguments
    fill_coast = process_arguments(
            detect_land, min_file_index, max_file_index, fill_coast)

    # Get list of all separate input files to process
    fullpath_input_filenames_list, input_base_filenames_list = \
        get_fullpath_input_filenames_list(
                input, min_file_index, max_file_index)

    # Get the list of all output files to be written to
    fullpath_output_filenames_list = get_fullpath_output_filenames_list(
            output, min_file_index, max_file_index)

    num_files = len(fullpath_input_filenames_list)

    # Filling mas values
    fill_value = 999

    # Iterate over multiple separate files
    for file_index in range(num_files):

        # Open file
        agg = load_dataset(fullpath_input_filenames_list[file_index])

        # Load variables
        datetime_obj, lon_obj, lat_obj, east_vel_obj, north_vel_obj, \
            east_vel_error_obj, north_vel_error_obj = load_variables(agg)

        # To not issue error/warning when data has nan
        # numpy.warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore')

        # Get arrays
        datetime = datetime_obj[:]
        lon = lon_obj[:]
        lat = lat_obj[:]
        U_all_times = east_vel_obj[:]
        V_all_times = north_vel_obj[:]

        # Refinement
        # Do not use this, because (1) lon and lat for original and refined
        # grids will be different, hence the plot functions should be aware of
        # these two grids, and (2) inpainted results on refined grid is poor.
        # if refine_grid != 1:
        #     lon, lat, U_all_times, V_all_times = refine_grid(
        #             refine_grid, lon, lat, U_all_times, V_all_times)

        # Determine the land
        land_indices, ocean_indices = detect_land_ocean(lon, lat, detect_land)

        # If plotting, remove these files:
        if plot is True:
            # Remove ~/.Xauthority and ~/.ICEauthority
            import os.path
            home_dir = os.path.expanduser("~")
            if os.path.isfile(home_dir+'/.Xauthority'):
                os.remove(home_dir+'/.Xauthority')
            if os.path.isfile(home_dir+'/.ICEauthority'):
                os.remove(home_dir+'/.ICEauthority')

        # Check whether to perform uncertainty quantification or not
        if uncertainty_quant is True:

            # Restore all generated ensembles
            timeframe, U_all_ensembles_inpainted_mean, \
                V_all_ensembles_inpainted_mean, \
                U_all_ensembles_inpainted_std, \
                V_all_ensembles_inpainted_std, mask_info = \
                restore_generated_ensembles(
                        diffusivity, sweep, timeframe, fill_coast, alpha,
                        convex_hull, num_ensembles, num_modes,
                        kernel_window, scale_error, datetime, lon, lat,
                        land_indices, U_all_times, V_all_times,
                        east_vel_error_obj, north_vel_error_obj, fill_value,
                        plot, save=save)

            # Write results to netcdf output file
            write_output_file(
                    timeframe,
                    datetime_obj,
                    lon,
                    lat,
                    mask_info,
                    U_all_ensembles_inpainted_mean,
                    V_all_ensembles_inpainted_mean,
                    U_all_ensembles_inpainted_std,
                    V_all_ensembles_inpainted_std,
                    fill_value,
                    fullpath_output_filenames_list[file_index])

        else:

            # Restore With Central Ensemble (use original data, no uncertainty
            # quantification
            time_indices, U_all_times_inpainted, V_all_times_inpainted, \
                U_all_times_inpainted_error, V_all_times_inpainted_error, \
                mask_info_all_times = restore_main_ensemble(
                        diffusivity, sweep, timeframe, fill_coast, alpha,
                        convex_hull, datetime, lon, lat, land_indices,
                        U_all_times, V_all_times, fill_value, plot)

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
    if (min_file_index != '') or (max_file_index != ''):
        archive_multiple_files(output, fullpath_output_filenames_list,
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

    # Parse arguments
    arguments = parse_arguments(sys.argv)

    # Main function
    restore(**arguments)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    main()
