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

from restoreio._parser import parse_arguments
from restoreio._input_output import load_dataset, load_variables, \
         write_output_file
from restoreio._geography import detect_land_ocean
from restoreio._file_utilities import get_fullpath_input_filenames_list, \
        get_fullpath_output_filenames_list, archive_multiple_files
from restoreio._restore import restore_main_ensemble, \
        restore_generated_ensembles
# from restoreio._restore import refine_grid


# =======
# Restore
# =======

def restore(argv):
    """
    """

    # Parse arguments
    arguments = parse_arguments(argv)
    save = True  # Test

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
        # if arguments['refinement_level'] != 1:
        #     lon, lat, U_all_times, V_all_times = refine_grid(
        #             arguments['refinement_level'], lon, lat,
        #             U_all_times, V_all_times)

        # Determine the land
        land_indices, ocean_indices = detect_land_ocean(
                lon, lat, arguments['detect_land'])

        # If plotting, remove these files:
        if arguments['plot'] is True:
            # Remove ~/.Xauthority and ~/.ICEauthority
            import os.path
            home_dir = os.path.expanduser("~")
            if os.path.isfile(home_dir+'/.Xauthority'):
                os.remove(home_dir+'/.Xauthority')
            if os.path.isfile(home_dir+'/.ICEauthority'):
                os.remove(home_dir+'/.ICEauthority')

        # Check whether to perform uncertainty quantification or not
        if arguments['uncertainty_quantification'] is True:

            # Restore all generated ensembles
            timeframe, U_all_ensembles_inpainted_mean, \
                V_all_ensembles_inpainted_mean, \
                U_all_ensembles_inpainted_std, \
                V_all_ensembles_inpainted_std, mask_info = \
                restore_generated_ensembles(
                        arguments, datetime, lon, lat, land_indices,
                        U_all_times, V_all_times, east_vel_error_obj,
                        north_vel_error_obj, fill_value, save=save)

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
                        arguments, datetime, lon, lat, land_indices,
                        U_all_times, V_all_times, fill_value)

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
