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

import argparse
from .formatter import WrappedNewlineFormatter
from .examples import examples
from ..__version__ import __version__

__all__ = ['parse_arguments']


# =============
# Create Parser
# ============

def create_parser():
    """
    Creates parser object.

    The parser object can be used in both parsing arguments and to generate
    Sphinx documentation for the command line interface (CLI).
    """

    # Instantiate the parser
    description = 'Restore incomplete oceanographic dataset. ' + \
        '"restore" is provided by "restoreio" python package.'
    epilog = examples
    # formatter_class = argparse.RawTextHelpFormatter
    # formatter_class = argparse.ArgumentDefaultsHelpFormatter
    # formatter_class = DescriptionWrappedNewlineFormatter
    formatter_class = WrappedNewlineFormatter

    parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                     formatter_class=formatter_class,
                                     add_help=False)

    # Manually create two groups of required and optional arguments
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Add back help
    optional.add_argument('-h', '--help', action='help',
                          default=argparse.SUPPRESS,
                          help='show this help message and exit')

    # Input filename
    help_input = """
    Input filename. This can be either the path to a local file or the URL to
    a remote dataset. The file or URL may or may not have a file extension.
    However, if the file does have an extension, the file extension should be
    either ``.nc``, ``.nc4``, ``.ncd``, ``.nc.gz``, ``.ncml``, or ``*.ncml.gz``
    only.
    """
    required.add_argument('-i', type=str, help=help_input, metavar='INPUT',
                          required=True)

    # Output filename
    help_output = """
    Output filename. This can be either the path to a local file or the URL to
    a remote dataset. The file extension should be ``.nc`` or ``.ncml`` only.
    If no output file is provided, the output filename is constructed by adding
    the word ``_restored`` at the end of the input filename.
    """
    required.add_argument('-o', type=str, default='', metavar='OUTPUT',
                          help=help_output, required=True)

    # Min Longitude
    help_min_lon = """
    Minimum longitude in the unit of degrees to subset the processing domain.
    If not provided or set to ``nan``, the minimum longitude of the input data
    is considered.
    """
    optional.add_argument('--min-lon', type=float, default=float('nan'),
                          metavar='MIN_LON', help=help_min_lon)

    # Mix Longitude
    help_max_lon = """
    Maximum longitude in the unit of degrees to subset the processing domain.
    If not provided or set to ``nan``, the maximum longitude of the input data
    is considered.
    """
    optional.add_argument('--max-lon', type=float, default=float('nan'),
                          metavar='MAX_LON', help=help_max_lon)

    # Min Latitude
    help_min_lat = """
    Minimum latitude in the unit of degrees to subset the processing domain.
    If not provided or set to ``nan``, the minimum latitude of the input data
    is considered.
    """
    optional.add_argument('--min-lat', type=float, default=float('nan'),
                          metavar='MIN_LAT', help=help_min_lat)

    # Mix Latitude
    help_max_lat = """
    Maximum latitude in the unit of degrees to subset the processing domain.
    If not provided or set to ``nan``, the maximum latitude of the input data
    is considered.
    """
    optional.add_argument('--max-lat', type=float, default=float('nan'),
                          metavar='MAX_LAT', help=help_max_lat)

    # Min time
    help_min_time = """
    The start of the time interval within the dataset times to be processed.
    The time should be provided as a string with the format yyyy-mm-ddTHH:MM:SS
    where yyyy is year, mm is month, dd is day, HH is hour from 00 to 23, MM is
    minutes and SS is seconds. If the given time does not exactly match any
    time in the dataset, the closest data time is used. If this argument is not
    given, the earliest available time in the dataset is used. Note that
    specifying a time interval cannot be used together with uncertainty
    quantification (using option '-u'). For this case, use '--time' argument
    instead which specifies a single time point.
    (default: %(default)s)
    """
    optional.add_argument('--min-time', type=str, default='',
                          metavar="MIN_TIME", help=help_min_time)

    # Max time
    help_max_time = """
    The end of the time interval within the dataset times to be processed. The
    time should be provided as a string with the format ``yyyy-mm-ddTHH:MM:SS``
    where ``yyyy`` is year, ``mm`` is month, ``dd`` is day, ``HH`` is hour from
    `00` to `23`, ``MM`` is minutes and ``SS`` is seconds. If the given time
    does not exactly match any time in the dataset, the closest data time is
    used. If this argument is not given, the latest available time in the
    dataset is used. Note that specifying a time interval cannot be used
    together with uncertainty quantification (using option ``-u``). For this
    case, use ``--time`` argument instead which specifies a single time point.
    (default: %(default)s)
    """
    optional.add_argument('--max-time', type=str, default='',
                          metavar="MAX_TIME", help=help_max_time)

    # time
    help_time = """
    Specify a single time point to process. The time should be provided as a
    string with the format ``yyyy-mm-ddTHH:MM:SS`` where ``yyyy`` is year,
    ``mm`` is month, ``dd`` is day, ``HH`` is hour from `00` to `23`, ``MM`` is
    minutes and ``SS`` is seconds. If the given time does not exactly match any
    time in the dataset, the closest data time is used. If this option is not
    given, the latest available time in the dataset is used. This option sets
    both ``--min-time`` and ``--max-time`` to this given time value. The
    argument is useful when performing uncertainty quantification (using
    argument ``-u``) or plotting (using argument ``-p``) as these require a
    single time, rather than a time interval. In contrary, to specify a time
    interval, use ``--min-time`` and ``--max-time`` arguments.
    (default: %(default)s)
    """
    optional.add_argument('-t', '--time', type=str, default='',
                          metavar="TIME", help=help_time)

    # diffusivity
    help_diffusivity = """
    Diffusivity of the PDE solver (real number). Large number leads to
    diffusion dominant solution. Small numbers leads to advection dominant
    solution.
    """ + "(default: %(default)s)"
    optional.add_argument('-d', type=float, default=20, metavar='DIFFUSIVITY',
                          help=help_diffusivity)

    # Sweep
    help_sweep = """
    Sweeps the image data in all flipped directions. This ensures an even
    solution independent of direction.
    """
    optional.add_argument('-s', action='store_true', help=help_sweep)

    # Plot
    help_plot = """
    Plots the results. In this case, instead of iterating through all time
    frames, only one time frame (given with option ``--time``) is restored and
    plotted. If in addition, the uncertainty quantification is enabled (with
    option ``-u``), the statistical analysis for the given time frame is also
    plotted.
    """
    optional.add_argument('-p', action='store_true', help=help_plot)

    # Save
    help_save = """
    If `True`, the plots are not displayed, rather are saved in the current
    directory as ``.pdf`` and ``.svg`` format. This option is useful when
    executing this script in an environment without display (such as remote
    cluster). If `False`, the generated plots will be displayed.
    """
    optional.add_argument('-S', action='store_true', help=help_save)

    # Detect land
    help_detect_land = """
    Detect land and exclude it from ocean's missing data points. This option
    should be an boolean or an integer with the following values
    (default: %(default)s):

    - ``False``: Same as ``0``. See below.

    - ``True``: Same as ``2``. See below.

    - ``0``: Does not detect land from ocean. All land points are assumed to be
        a part of ocean's missing points.

    - ``1``: Detect land. Most accurate, slowest.

    - ``2``: Detect land. Less accurate, fastest (preferred method).

    - ``3``: Detect land. Currently this option is not fully implemented.
    """
    optional.add_argument('-L', choices=[0, 1, 2, 3, False, True],
                          default=True, type=int, metavar='DETECT_LAND',
                          help=help_detect_land)

    # Include near shore
    help_fill_coast = """
    Fills the gap the between the data in the ocean and between ocean and the
    coastline. This option is only effective if ``L`` is not set to ``0``.
    """
    optional.add_argument('-l', action="store_true", help=help_fill_coast)

    # Convex Hull
    help_convex_hull = """
    Instead of using the concave hull (alpha shape) around the data points,
    this options uses convex hull of the area around the data points.
    """
    optional.add_argument('-c', action="store_true", help=help_convex_hull)

    # Alpha
    help_alpha = """
    The alpha number for alpha shape. If not specified or a negative number,
    this value is computed automatically. This option is only relevant to
    concave shapes. This option is ignored if convex shape is used with ``-c``
    option.
    """
    optional.add_argument('-a', default=-1, type=float, metavar="ALPHA",
                          help=help_alpha)

    # Refine
    help_refine_grid = """
    Refines the grid by increasing the number of points on each axis by a
    multiple of a given integer. If this option is set to 1, no refinement is
    performed. If set to integer n, the number of grid points is refined by
    `n^2` times (that is `n` times on each axis).
    (default: %(default)s)
    """
    optional.add_argument('-r', type=int, default=1, metavar="REFINE",
                          help=help_refine_grid)

    # Uncertainty quantification
    help_uncertainty_quant = """
    Enables uncertainty quantification for the time frame given in option
    ``--time``. This either produces results in output file as given in option
    ``-o``, or plots the results if the option ``-p`` is specified.
    """
    optional.add_argument('-u', action="store_true",
                          help=help_uncertainty_quant)

    # Number of ensemble
    help_num_samples = """
    Number of ensemble members used for uncertainty quantification. This option
    is only relevant to uncertainty quantification (when ``-u`` is used).
    (default: %(default)s)
    """
    optional.add_argument('-e', type=int, default=1000, metavar="NUM_ENSEMBLE",
                          help=help_num_samples)

    # Number of modes
    help_ratio_num_modes = """
    Ratio of the number of KL eigen-modes to be used in the truncation of the
    KL expansion. The ratio is defined by the number of modes to be used over
    the total number of modes. The ratio is a number between `0` and `1`. For
    instance, if set to `1`, all modes are used, hence the KL expansion is not
    truncated. If set to `0.5`, half of the number of modes are used. This
    option is only relevant to uncertainty quantification (when ``-u`` is
    used).
    """
    optional.add_argument('-m', type=float, default=1.0, metavar="NUM_MODES",
                          help=help_ratio_num_modes)

    # Kernel window
    help_kernel_width = """
    Window of the kernel to estimate covariance of data. The window width
    should be given by the integer number of pixels (data points). The non-zero
    extent of the kernel a square area with twice the window length in both
    longitude and latitude directions. This option is only relevant to
    uncertainty quantification (when ``-u`` is used).
    (default: %(default)s)
    """
    optional.add_argument('-w', default=5, type=int, metavar="WINDOW",
                          help=help_kernel_width)

    # Scale error
    help_scale_error = """
    Scale velocity error of the input data by a factor. Often, the input
    velocity error is the dimensionless GDOP which needs to be scaled by the
    standard deviation of the velocity error to represent the actual velocity
    error. This value scales the error. This option is only relevant to
    uncertainty quantification (when ``-u`` is used).
    (default: %(default)s)
    """
    optional.add_argument('-E', default=0.08, type=float,
                          metavar="SCALE_ERROR", help=help_scale_error)

    # Start file index
    help_min_file_index = """
    Start file iterator to be used for processing multiple input files. For
    Instance, ``-i input -I 003 -J 012`` means to read the series of input
    files with iterators ``input003.nc``, ``input004.nc``, to ``input012.nc``.
    If this option is used, the option ``-J`` should also be given.
    """
    optional.add_argument('-I', type=str, default='', metavar="START_FILE",
                          help=help_min_file_index)

    # End file index
    help_max_file_index = """
    Start file iterator to be used for processing multiple input files. For
    Instance, ``-i input -I 003 -J 012`` means to read the series of input
    files with iterators ``input003.nc``, ``input004.nc``, to ``input012.nc``.
    If this option is used, the option ``-I`` should also be given.
    """
    optional.add_argument('-J', type=str, default='', metavar="END_FILE",
                          help=help_max_file_index)

    # Write samples
    help_write_samples = """
    If `True`, all generated ensemble will be written to the output file. This
    option is relevant to uncertainty quantification (when ``-u`` is used).
    """
    optional.add_argument('-W', action='store_true', help=help_write_samples)

    # Verbose
    help_verbose = """
    Prints verbose information.
    """
    optional.add_argument('-v', action='store_true', help=help_verbose)

    # Terminate
    help_terminate = """
    If `True`, on encountering errors, the program both raises error and exists
    with code 1 with printing the message starting with the keyword
    ``ERROR``. This is useful when this package is executed on a server to
    pass exit signals to a Node application. On the downside, this option
    causes an interactive python environment to both terminate the script and
    the python environment itself. To avoid this, set this option to `False`.
    In this case, upon an error, the ``ValueError`` is raised, which cases the
    script to terminate, however, an interactive python environment will not be
    exited.
    """
    optional.add_argument('-T', action='store_true', help=help_terminate)

    # Version
    help_version = """
    Prints version.
    """
    version = '%(prog)s {version}'.format(version=__version__)
    parser.add_argument('-V', '--version', action='version', version=version,
                        help=help_version)

    return parser


# ===============
# Parse Arguments
# ===============

def parse_arguments():
    """
    Parse the user's command line arguments.
    """

    # Create parser object
    parser = create_parser()

    # Parse arguments. Here args is a namespace
    args = parser.parse_args()

    # Convert namespace to dictionary
    # args = vars(args)

    # Output dictionary
    arguments = {
        'input': args.i,
        'output': args.o,
        'min_lon': args.min_lon,
        'max_lon': args.max_lon,
        'min_lat': args.min_lat,
        'max_lat': args.max_lat,
        'min_time': args.min_time,
        'max_time': args.max_time,
        'time': args.time,
        'diffusivity': args.d,
        'sweep': args.s,
        'plot': args.p,
        'save': args.S,
        'detect_land': args.L,
        'fill_coast': args.l,
        'convex_hull': args.c,
        'alpha': args.a,
        'refine_grid': args.r,
        'uncertainty_quant': args.u,
        'num_samples': args.e,
        'ratio_num_modes': args.m,
        'kernel_width': args.w,
        'scale_error': args.E,
        "min_file_index": args.I,
        "max_file_index": args.J,
        "write_samples": args.W,
        "verbose": args.v,
        "terminate": args.T,
    }

    return arguments
