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


# ==========
# Parse Args
# ==========

def parse_arguments(argv):
    """
    Parse the user's command line arguments.
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
    Input filename. This can be either the path to a local file or the url to
    a remote dataset. The file extension should be *.nc or *.ncml only.
    """
    required.add_argument('-i', type=str, help=help_input, metavar='INPUT',
                          required=True)

    # Output filename
    help_output = """
    Output filename. This can be either the path to a local file or the url to
    a remote dataset. The file extension should be *.nc or *.ncml only. If
    no output file is provided, the output filename is constructed by adding
    the word '_restored' at the end of the input filename.
    """
    required.add_argument('-o', type=str, default='', metavar='OUTPUT',
                          help=help_output, required=True)

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
    frames, only one time frame (given with option -t) is restored and plotted.
    If in addition, the uncertainty quantification is enabled (with option -u),
    the statistical analysis for the given time frame is also plotted.
    """
    optional.add_argument('-p', action='store_true', help=help_plot)

    # Detect land
    help_detect_land = """
    Detect land and exclude it from ocean's missing data points. This option
    should be an integer with the following values (default: %(default)s):

    - 0: Does not detect land from ocean. All land points are assumed to be a
         part of ocean's missing points.

    - 1: Detect land. Most accurate, slowest.

    - 2: Detect land. Less accurate, fastest (preferred method).

    - 3: Detect land. Currently this option is not fully implemented.
    """
    optional.add_argument('-L', choices=[0, 1, 2, 3], default=0, type=int,
                          metavar='DETECT_LAND', help=help_detect_land)

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
    concave shapes. This option is ignored if convex shape is used with '-c'
    option.
    """
    optional.add_argument('-a', default=-1, type=float, metavar="ALPHA",
                          help=help_alpha)

    # Refine
    help_refine_grid = """
    Refines the grid by increasing the number of points on each axis by a
    multiple of a given integer. If this option is set to 1, no refinement is
    performed. If set to integer n, the number of grid points is refined by
    n^2 times (that is n times on each axis).
    (default: %(default)s)
    """
    optional.add_argument('-r', type=int, default=1, metavar="REFINE",
                          help=help_refine_grid)

    # Time frame
    help_timeframe = """
    The time frame index in the dataset to process and to plot the uncertainty
    quantification. The index wraps around the total number of time frames. For
    instance, -1 indicates the last time frame.
    (default: %(default)s)
    """
    optional.add_argument('-t', type=int, default=None, metavar="TIME_INDEX",
                          help=help_timeframe)

    # Uncertainty quantification
    help_uncertainty_quant = """
    Enables uncertainty quantification for the time frame given in option -t.
    This either produces results in output file as given in option -o, or plots
    the results if the option -p is specified.
    """
    optional.add_argument('-u', action="store_true",
                          help=help_uncertainty_quant)

    # Number of ensembles
    help_num_ensembles = """
    Number of ensembles used for uncertainty quantification. This option is
    only relevant to uncertainty quantification (when -u is used).
    (default: %(default)s)
    """
    optional.add_argument('-e', type=int, default=1000, metavar="NUM_ENSEMBLE",
                          help=help_num_ensembles)

    # Number of modes
    help_num_modes = """
    Number of eigen-modes used for uncertainty quantification (in
    KL-expansion). When not specified, maximum number of possible modes will be
    used, which is the number of ensembles minus one. This option is only
    relevant to uncertainty quantification (when -u is used).
    """
    optional.add_argument('-m', type=int, default=None, metavar="NUM_MODES",
                          help=help_num_modes)

    # Kernel window
    help_kernel_window = """
    Window of the kernel to estimate covariance of data. The window length
    should be given by the integer number of pixels (data points). The non-zero
    extent of the kernel a square area with twice the window length in both
    longitude and latitude directions. This option is only relevant to
    uncertainty quantification (when -u is used).
    (default: %(default)s)
    """
    optional.add_argument('-w', default=5, type=int, metavar="WINDOW",
                          help=help_kernel_window)

    # Scale error
    help_scale_error = """
    Scale velocity error of the input data by a factor. Often, the input
    velocity error is the dimensionless GDOP which needs to be scaled by the
    standard deviation of the velocity error to represent the actual velocity
    error. This value scales the error. This option is only relevant to
    uncertainty quantification (when -u is used).
    (default: %(default)s)
    """
    optional.add_argument('-S', default=0.08, type=float,
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

    # Version
    help_version = """
    Prints version.
    """
    version = '%(prog)s {version}'.format(version=__version__)
    parser.add_argument('-v', '--version', action='version', version=version,
                        help=help_version)

    # Parse arguments. Here args is a namespace
    args = parser.parse_args()

    # Convert namespace to dictionary
    # args = vars(args)

    # Output dictionary
    arguments = {
        'input': args.i,
        'output': args.o,
        'diffusivity': args.d,
        'sweep': args.s,
        'plot': args.p,
        'detect_land': args.L,
        'fill_coast': args.l,
        'convex_hull': args.c,
        'alpha': args.a,
        'refine_grid': args.r,
        'timeframe': args.t,
        'uncertainty_quant': args.u,
        'num_ensembles': args.e,
        'num_modes': args.m,
        'kernel_window': args.w,
        'scale_error': args.S,
        "min_file_index": args.I,
        "max_file_index": args.J,
    }

    return arguments
