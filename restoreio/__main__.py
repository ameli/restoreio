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
import warnings

from ._parser import parse_arguments
from ._subset import subset_domain, subset_datetime
from ._input_output import load_dataset, load_variables, get_datetime_info, \
        write_output_file
from ._geography import detect_land_ocean
from ._file_utilities import get_fullpath_input_filenames_list, \
        get_fullpath_output_filenames_list, archive_multiple_files
from ._restore import restore_main_ensemble, restore_generated_ensembles
from ._server_utils import globals, terminate_with_error

__all__ = ['restore']


# ==============
# get fill value
# ==============

def _get_fill_value(east_vel_obj, north_vel_obj):
    """
    Finds missing value (or fill value) from wither of east of north velocity
    objects.
    """

    # Missing Value
    if hasattr(east_vel_obj, '_FillValue') and \
            (not numpy.isnan(float(east_vel_obj._FillValue))):
        fill_value = numpy.fabs(float(east_vel_obj._FillValue))

    elif hasattr(north_vel_obj, '_FillValue') and \
            (not numpy.isnan(float(north_vel_obj._FillValue))):
        fill_value = numpy.fabs(float(north_vel_obj._FillValue))

    elif hasattr(east_vel_obj, 'missing_value') and \
            (not numpy.isnan(float(east_vel_obj.missing_value))):
        fill_value = numpy.fabs(float(east_vel_obj.missing_value))

    elif hasattr(north_vel_obj, 'missing_value') and \
            (not numpy.isnan(float(north_vel_obj.missing_value))):
        fill_value = numpy.fabs(float(north_vel_obj.missing_value))

    elif hasattr(east_vel_obj, 'fill_value') and \
            (not numpy.isnan(float(east_vel_obj.fill_value))):
        fill_value = numpy.fabs(float(east_vel_obj.fill_value))

    elif hasattr(north_vel_obj, 'fill_value') and \
            (not numpy.isnan(float(north_vel_obj.fill_value))):
        fill_value = numpy.fabs(float(north_vel_obj.fill_value))

    else:
        fill_value = 999.0

    return fill_value


# =================
# process arguments
# =================

def process_arguments(
        detect_land,
        min_file_index,
        max_file_index,
        fill_coast,
        time,
        min_time,
        max_time,
        uncertainty_quant,
        plot):
    """
    Parses the argument of the executable and obtains the filename.
    """

    # Check include land
    if detect_land == 0:
        fill_coast = False

    # Check Processing multiple file
    if ((min_file_index != '') and (max_file_index != '')):

        if ((min_file_index == '') or (max_file_index == '')):
            terminate_with_error(
                'To process multiple files, both min and max file iterator ' +
                'should be specified.')

    # A time interval and single time point cannot be both specified.
    if (min_time != '' or max_time != '') and (time != ''):
        terminate_with_error(
            'When "time" argument is specified, the other time arguments ' +
            '"min_time" and "max_time" cannot be specified and vice versa.')

    # When uncertainty quantification is used, only a single time point should
    # be given, not an interval of time.
    if (uncertainty_quant is True) and (min_time != '' or max_time != ''):
        terminate_with_error(
            'When uncertainty quantification is enabled, a time interval ' +
            'cannot be specified, rather a single time should be specified ' +
            'using "time" argument.')

    # When plotting, only a single time point can be plotted
    if (plot is True) and (((min_time != '') or (max_time != '')) or
       ((min_time == '') and (max_time == '') and (time == ''))):
        terminate_with_error(
            'When plotting is enabled, a time interval with "min_time" or ' +
            '"max_time" arguments should not be given. Rather, only a ' +
            'single time point can be plotted that is specified with ' +
            '"time" argument.')

    return fill_coast


# =======
# Restore
# =======

def restore(
        input,
        min_file_index='',
        max_file_index='',
        output='',
        min_lon=float('nan'),
        max_lon=float('nan'),
        min_lat=float('nan'),
        max_lat=float('nan'),
        min_time='',
        max_time='',
        time='',
        diffusivity=20,
        sweep=False,
        detect_land=True,
        fill_coast=False,
        convex_hull=False,
        alpha=20,
        refine_grid=1,
        uncertainty_quant=False,
        num_samples=1000,
        ratio_num_modes=1,
        kernel_width=5,
        scale_error=0.08,
        write_samples=False,
        plot=False,
        save=True,
        verbose=False,
        terminate=False):
    """
    Restore incomplete oceanographic dataset and generate data ensemble.

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

    min_lon : float, default=float('nan')
        Minimum longitude in the unit of degrees to subset the processing
        domain. If not provided or set to `float('nan')`, the minimum longitude
        of the input data is considered.

    max_lon : float, default=float('nan')
        Maximum longitude in the unit of degrees to subset the processing
        domain. If not provided or set to `float('nan')`, the maximum longitude
        of the input data is considered.

    min_lat : float, default=float('nan')
        Minimum latitude in the unit of degrees to subset the processing
        domain. If not provided or set to `float('nan')`, the minimum latitude
        of the input data is considered.

    max_lat : float, default: float('nan')
        Maximum latitude in the unit of degrees to subset the processing
        domain. If not provided or set to `float('nan')`, the maximum latitude
        of the input data is considered.

    min_time : str, default=''
        The start of the time interval within the dataset times to be
        processed. The time should be provided as a string with the format
        ``'yyyy-mm-ddTHH:MM:SS'`` where ``yyyy`` is year, ``mm`` is month,
        ``dd`` is day, ``HH`` is hour from `00` to `23`, ``MM`` is minutes and
        ``SS`` is seconds. If the given time does not exactly match any time in
        the dataset, the closest data time is used. If this argument is not
        given, the earliest available time in the dataset is used. Note that
        specifying a time interval cannot be used together with uncertainty
        quantification (using argument ``uncertainty_quant=True``). For this
        case, use ``time`` argument instead which specifies a single time
        point.

    max_time : str, default=''
        The end of the time interval within the dataset times to be
        processed. The time should be provided as a string with the format
        ``'yyyy-mm-ddTHH:MM:SS'`` where ``yyyy`` is year, ``mm`` is month,
        ``dd`` is day, ``HH`` is hour from `00` to `23`, ``MM`` is minutes and
        ``SS`` is seconds. If the given time does not exactly match any time in
        the dataset, the closest data time is used. If this argument is not
        given, the latest available time in the dataset is used. Note that
        specifying a time interval cannot be used together with uncertainty
        quantification (using argument ``uncertainty_quant=True``). For this
        case, use ``time`` argument instead which specifies a single time
        point.

    time : str, default=''
        Specify a single time point to process. The time should be provided as
        a string with the format ``'yyyy-mm-ddTHH:MM:SS'`` where ``yyyy`` is
        year, ``mm`` is month, ``dd`` is day, ``HH`` is hour from `00` to `23`,
        ``MM`` is minutes and ``SS`` is seconds. If the given time does not
        exactly match any time in the dataset, the closest data time is used.
        If this option is not given, the latest available time in the dataset
        is used. This option sets both ``min_time`` and ``max_time`` to this
        given time value. The argument is useful when performing uncertainty
        quantification (using argument ``uncertainty_quant=True``) or plotting
        (using argument ``plot=True``) as these require a single time, rather
        than a time interval. In contrary, to specify a time interval, use
        ``min_time`` and ``max_time`` arguments.

    diffusivity : float, default=20
        Diffusivity of the PDE solver (real number). Large number leads to
        diffusion dominant solution. Small numbers leads to advection dominant
        solution.

    sweep : bool, default=False
        Sweeps the image data in all flipped directions. This ensures an even
        solution independent of direction.

    detect_land : int, default=2
        Detect land and exclude it from ocean's missing data points. This
        option should be a boolean or an integer with the following values:

        * ``False``: Same as ``0``. See below.
        * ``True``: Same as ``2``. See below.
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

    uncertainty_quant : bool, default=False
        Performs uncertainty quantification on the data for the time frame
        given by ``time`` option.

    num_samples : int, default=1000
        Number of ensemble members used for uncertainty quantification. This
        option is relevant if ``uncertainty_quant`` is set to `True`.

    ratio_num_modes : int, default=1.0
        Ratio of the number of KL eigen-modes to be used in the truncation of
        the KL expansion. The ratio is defined by the number of modes to be
        used over the total number of modes. The ratio is a number between 0
        and 1. For instance, if set to 1, all modes are used, hence the KL
        expansion is not truncated. If set to 0.5, half of the number of modes
        are used. This option is relevant if ``uncertainty_quant`` is set to
        `True`.

    kernel_width : int, default=5
        Window of the kernel to estimate covariance of data. The window width
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

    write_samples : bool, default=False,
        If `True`, all generated ensemble will be written to the output file.
        This option is relevant if ``uncertainty_quant`` is set to `True`.

    plot : bool, default=False
        Plots the results. In this case, instead of iterating through all time
        frames, only one time frame (given with option ``timeframe``) is
        restored and plotted. If in addition, the uncertainty quantification is
        enabled (with option ``uncertainty_quant=True``), the statistical
        analysis for the given time frame is also plotted.

    save : bool, default=True
        If `True`, the plots are not displayed, rather are saved in the current
        directory as ``.pdf`` and ``.svg`` format. This option is useful when
        executing this script in an environment without display (such as remote
        cluster). If `False`, the generated plots will be displayed.

    verbose : bool, default=False
        If `True`, prints verbose information during the computation process.

    terminate ; bool, default=False
        If `True`, on encountering errors, the program both raises error and
        exists with code 1 with printing the message starting with the keyword
        ``ERROR``. This is useful when this package is executed on a server
        to pass exit signals to a Node application. On the downside, this
        option causes an interactive python environment to both terminate the
        script and the python environment itself. To avoid this, set this
        option to `False`. In this case, upon an error, the ``ValueError`` is
        raised, which cases the script to terminate, however, an interactive
        python environment will not be exited.

    See Also
    --------

    restoreio.scan

    Notes
    -----

    **Output File:**

    The output is a NetCDF file in ``.nc`` format containing a selection of the
    following variables, contingent on the chosen configuration:

    1. Mask
    2. Reconstructed East and North Velocities
    3. East and North Velocity Errors
    4. East and North Velocity Ensemble

    **1. Mask:**

    The mask variable is a three-dimensional array with dimensions for *time*,
    *longitude*, and *latitude*. This variable is stored under the name
    ``mask`` in the output file.

    **Interpreting Mask Variable over Segmented Domains:**

    The mask variable includes information about the result of domain
    segmentation. This array contains integer values ``-1``, ``0``, ``1``, and
    ``2`` that are interpreted as follows:

    * The value ``-1`` indicates the location is identified to be on the
      **land** domain :math:`\\Omega_l`. In these locations, the output
      velocity variable is masked.
    * The value ``0`` indicates the location is identified to be on the
      **known** domain :math:`\\Omega_k`. These locations have velocity data in
      the input file. The same velocity values are preserved in the output
      file.
    * The value ``1`` indicates the location is identified to be on the
      **missing** domain :math:`\\Omega_m`. These locations do not have a
      velocity data in the input file, but they do have a reconstructed
      velocity data on the output file.
    * The value ``2`` indicates the location is identified to be on the
      **ocean** domain :math:`\\Omega_o`. In these locations, the output
      velocity variable is masked.

    **2. Reconstructed East and North Velocities:**

    The reconstructed east and north velocity variables are stored in the
    output file under the names ``east_vel`` and ``north_vel``, respectively.
    These variables are three-dimensional arrays with dimensions for *time*,
    *longitude*, and *latitude*.

    **Interpreting Velocity Variables over Segmented Domains:**

    The velocity variables on each of the segmented domains are defined as
    follows:

    * On locations where the ``mask`` value is ``-1`` or ``2``, the output
      velocity variables are masked.
    * On locations where the ``mask`` value is ``0``, the output velocity
      variables have the same values as the corresponding variables in the
      input file.
    * On locations where the ``mask`` value is ``1``, the output velocity
      variables are reconstructed. If the ``uncertainty_quant`` is enabled,
      these output velocity variables are obtained by the *mean* of the
      velocity ensemble, where the missing domain of each ensemble is
      reconstructed.

    **3. East and North Velocity Errors:**

    If the ``uncertainty_quant`` option is enabled, the east and north velocity
    error variables will be included in the output file under the names
    ``east_err`` and ``north_err``, respectively. These variables are
    three-dimensional arrays with dimensions for *time*, *longitude*, and
    *latitude*.

    **Interpreting Velocity Error Variable over Segmented Domains:**

    The velocity error variables on each of the segmented domains are defined
    as follows:

    * On locations where the ``mask`` value is ``-1`` or ``2``, the output
      velocity error variables are masked.
    * On locations where the ``mask`` value is ``0``, the output velocity error
      variables are obtained from either the corresponding velocity error or
      GDOP variables in the input file scaled by the value of ``scale_error``.
    * On locations where the ``mask`` value is ``1``, the output velocity error
      variables are obtained from the *standard deviation* of the ensemble,
      where the missing domain of each ensemble is reconstructed.

    **4. East and North Velocity Ensemble:**

    When you activate the ``uncertainty_quant`` option, a collection of
    velocity field ensemble is created. Yet, by default, the output file only
    contains the mean and standard deviation of these ensemble. To incorporate
    all ensemble into the output file, you should additionally enable the
    ``write_samples`` option. This action saves the east and north velocity
    ensemble variables in the output file as ``east_vel_ensemble`` and
    ``north_vel_ensemble``, respectively. These variables are four-dimensional
    arrays with dimensions for *ensemble*, *time*, *longitude*, and
    *latitude*.

    The *ensemble* dimension of the array has the size :math:`s+1` where
    :math:`s` is the number of ensemble specified by ``num_samples``
    argument.. The first ensemble with the index :math:`0` corresponds to the
    original input dataset. The other ensemble with the indices
    :math:`1, \\dots, s` correspond to the generated ensemble.

    **Interpreting Velocity Ensemble Variables over Segmented Domains:**

    The velocity ensemble variables on each of the segmented domains are
    defined similar to those presented for velocity velocities. In particular,
    the missing domain of each ensemble is reconstructed independently.

    **Mean and Standard Deviation of Ensemble:**

    Note that the *mean* and *standard deviation* of the velocity ensemble
    arrays over the ensemble dimension yield the velocity and velocity error
    variables, respectively.

    Examples
    --------

    **Restoring Data:**

    The following code is a minimalistic example of restoring the missing data
    of an HF radar dataset:

    .. code-block:: python

        >>> # Import package
        >>> from restoreio import restore

        >>> # OpenDap URL of HF radar data, south side of Martha's Vineyard
        >>> input = 'https://transport.me.berkeley.edu/thredds/dodsC/' + \\
        ...         'root/WHOI-HFR/WHOI_HFR_2014_original.nc'

        >>> # Subsetting time
        >>> min_time = '2014-07-01T20:00:00'
        >>> max_time = '2014-07-03T20:00:00'

        >>> # Specify output
        >>> output = 'output.nc'

        >>> # Restore missing velocity data
        >>> restore(input, output=output, min_time=min_time, max_time=max_time,
        ...         detect_land=True, fill_coast=False, convex_hull=False,
        ...         alpha=20, plot=False, verbose=True)

    **Ensemble Generation:**

    The following code is an example of generating ensemble for an HF radar
    dataset:

    .. code-block:: python

        >>> # Import package
        >>> from restoreio import restore

        >>> # OpenDap URL of HF radar data, US west coast
        >>> url = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/' + \\
        ...       'hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hou' + \\
        ...       'rly_RTV_best.ncd'

        >>> # Subsetting spatial domain to the Monterey Bay region, California
        >>> min_lon = -122.344
        >>> max_lon = -121.781
        >>> min_lat = 36.507
        >>> max_lat = 36.992

        >>> # Time subsetting
        >>> time_point = '2017-01-25T03:00:00'

        >>> # Generate ensemble and reconstruct gaps
        >>> restore(input=url, output='output.nc', min_lon=min_lon,
        ...         max_lon=max_lon, min_lat=min_lat, max_lat=max_lat,
        ...         time=time_point, uncertainty_quant=True, plot=False,
        ...         num_samples=2000, ratio_num_modes=1, kernel_width=5,
        ...         scale_error=0.08, detect_land=True, fill_coast=True,
        ...         write_samples=True, verbose=True)
    """

    # Define global variable for terminate with error
    if terminate:
        globals.terminate = True
    else:
        globals.terminate = False

    # Check arguments
    fill_coast = process_arguments(
            detect_land, min_file_index, max_file_index, fill_coast,
            time, min_time, max_time, uncertainty_quant, plot)

    # Get list of all separate input files to process
    fullpath_input_filenames_list, input_base_filenames_list = \
        get_fullpath_input_filenames_list(
                input, min_file_index, max_file_index)

    # Get the list of all output files to be written to
    fullpath_output_filenames_list = get_fullpath_output_filenames_list(
            output, min_file_index, max_file_index)

    num_files = len(fullpath_input_filenames_list)

    # Iterate over multiple separate files
    for file_index in range(num_files):

        # Open file
        agg = load_dataset(fullpath_input_filenames_list[file_index],
                           verbose=verbose)

        # Load variables
        datetime_obj, lon_obj, lat_obj, east_vel_obj, north_vel_obj, \
            east_vel_error_obj, north_vel_error_obj = load_variables(agg)

        # To not issue error/warning when data has nan
        # numpy.warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore')

        # Fill value
        fill_value = _get_fill_value(east_vel_obj, north_vel_obj)

        # Get datetime info from datetime netcdf object
        datetime_info = get_datetime_info(datetime_obj)

        # Subset time
        min_datetime_index, max_datetime_index = subset_datetime(
            datetime_info, min_time, max_time, time)

        datetime_info['array'] = \
            datetime_info['array'][min_datetime_index:max_datetime_index+1]

        # Subset domain
        min_lon_index, max_lon_index, min_lat_index, max_lat_index = \
            subset_domain(lon_obj, lat_obj, min_lon, max_lon, min_lat, max_lat)
        lon = lon_obj[min_lon_index:max_lon_index+1]
        lat = lat_obj[min_lat_index:max_lat_index+1]

        # if the velocity arrays have depth dimension, use only the first index
        depth_index = 0
        vel_array_dim = len(east_vel_obj.shape)

        # Subset velocity arrays both in datetime and domain
        if vel_array_dim == 3:

            u_all_times = east_vel_obj[
                    min_datetime_index:max_datetime_index+1,
                    min_lat_index:max_lat_index+1,
                    min_lon_index:max_lon_index+1]

            v_all_times = north_vel_obj[
                    min_datetime_index:max_datetime_index+1,
                    min_lat_index:max_lat_index+1,
                    min_lon_index:max_lon_index+1]

        elif vel_array_dim == 4:

            u_all_times = east_vel_obj[
                    min_datetime_index:max_datetime_index+1,
                    depth_index,
                    min_lat_index:max_lat_index+1,
                    min_lon_index:max_lon_index+1]

            v_all_times = north_vel_obj[
                    min_datetime_index:max_datetime_index+1,
                    depth_index,
                    min_lat_index:max_lat_index+1,
                    min_lon_index:max_lon_index+1]

        else:
            terminate_with_error('Velocity arrays should have three or four' +
                                 'dimensions.')

        # Velocity error
        if east_vel_error_obj is None:
            # When None, no uncertainty quantification should be used.
            u_error_all_times = None
            v_error_all_times = None

        else:

            # if the velocity error has depth dimension, use only the first
            # index
            depth_index = 0
            error_array_dim = len(east_vel_obj.shape)

            # Subset velocity arrays both in datetime and domain
            if error_array_dim == 3:

                u_error_all_times = east_vel_error_obj[
                        min_datetime_index:max_datetime_index+1,
                        min_lat_index:max_lat_index+1,
                        min_lon_index:max_lon_index+1]

                v_error_all_times = north_vel_error_obj[
                        min_datetime_index:max_datetime_index+1,
                        min_lat_index:max_lat_index+1,
                        min_lon_index:max_lon_index+1]

            elif error_array_dim == 4:

                u_error_all_times = east_vel_obj[
                        min_datetime_index:max_datetime_index+1,
                        depth_index,
                        min_lat_index:max_lat_index+1,
                        min_lon_index:max_lon_index+1]

                v_error_all_times = north_vel_obj[
                        min_datetime_index:max_datetime_index+1,
                        depth_index,
                        min_lat_index:max_lat_index+1,
                        min_lon_index:max_lon_index+1]

            else:
                terminate_with_error(
                    'Velocity error or DGOP arrays should have three or ' +
                    'four dimensions.')

        # Refinement
        # Do not use this, because (1) lon and lat for original and refined
        # grids will be different, hence the plot functions should be aware of
        # these two grids, and (2) inpainted results on refined grid is poor.
        # if refine_grid != 1:
        #     lon, lat, u_all_times, v_all_times = refine_grid(
        #             refine_grid, lon, lat, u_all_times, v_all_times)

        # Determine the land
        land_indices, ocean_indices = detect_land_ocean(
                lon, lat, detect_land, verbose=verbose)

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

            # Restore all generated ensemble
            u_all_ensembles_inpainted, v_all_ensembles_inpainted, \
                u_all_ensembles_inpainted_mean, \
                v_all_ensembles_inpainted_mean, \
                u_all_ensembles_inpainted_std, v_all_ensembles_inpainted_std, \
                mask_info = restore_generated_ensembles(
                        diffusivity, sweep, fill_coast, alpha, convex_hull,
                        num_samples, ratio_num_modes, kernel_width,
                        scale_error, lon, lat, land_indices, u_all_times,
                        v_all_times, u_error_all_times, v_error_all_times,
                        fill_value, file_index, num_files, plot=plot,
                        save=save, verbose=verbose)

            # Write results to netcdf output file
            write_output_file(
                    fullpath_output_filenames_list[file_index],
                    datetime_info,
                    lon,
                    lat,
                    mask_info,
                    fill_value,
                    u_all_ensembles_inpainted_mean,
                    v_all_ensembles_inpainted_mean,
                    u_all_ensembles_inpainted_std,
                    v_all_ensembles_inpainted_std,
                    u_all_ensembles_inpainted,
                    v_all_ensembles_inpainted,
                    write_samples=write_samples,
                    verbose=verbose)

        else:

            if write_samples:
                terminate_with_error('Cannot write ensemble to output ' +
                                     'when uncertainty quantification is ' +
                                     'not enabled.')

            # Restore With Central Ensemble (use original data, no uncertainty
            # quantification
            u_all_times_inpainted, v_all_times_inpainted, \
                u_all_times_inpainted_error, v_all_times_inpainted_error, \
                mask_info_all_times = restore_main_ensemble(
                        diffusivity, sweep, fill_coast, alpha, convex_hull,
                        lon, lat, land_indices, u_all_times, v_all_times,
                        fill_value, file_index, num_files, plot=plot,
                        save=save, verbose=verbose)

            # Write results to netcdf output file
            write_output_file(
                    fullpath_output_filenames_list[file_index],
                    datetime_info,
                    lon,
                    lat,
                    mask_info_all_times,
                    fill_value,
                    u_all_times_inpainted,
                    v_all_times_inpainted,
                    u_all_times_inpainted_error,
                    v_all_times_inpainted_error,
                    u_all_ensembles_inpainted=None,
                    v_all_ensembles_inpainted=None,
                    write_samples=False,
                    verbose=verbose)

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
    arguments = parse_arguments()

    # Main function
    restore(**arguments)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    main()
