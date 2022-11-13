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


# =================
# process arguments
# =================

def process_arguments(parser, args):
    """
    Parses the argument of the executable and obtains the filename.
    """

    # Initialize variables (defaults)
    arguments = {
        'FullPathInputFilename': args.i,
        'FullPathOutputFilename': args.o,
        'Diffusivity': args.d,
        'SweepAllDirections': args.s,
        'Plot': args.p,
        'ExcludeLandFromOcean': args.L,
        'IncludeLandForHull': args.l,
        'UseConvexHull': args.c,
        'Alpha': args.c,
        'RefinementLevel': args.r,
        'TimeFrame': args.t,
        'UncertaintyQuantification': args.u,
        'NumEnsembles': args.e,
        "ProcessMultipleFiles": False,
        "MultipleFilesMinIteratorString": args.m,
        "MultipleFilesMaxIteratorString": args.n,
    }

    # Check include land
    if arguments['ExcludeLandFromOcean'] == 0:
        arguments['IncludeLandForHull'] = False

    # Check Processing multiple file
    if ((arguments['MultipleFilesMinIteratorString'] != '') and
            (arguments['MultipleFilesMaxIteratorString'] != '')):

        if ((arguments['MultipleFilesMinIteratorString'] == '') or
                (arguments['MultipleFilesMaxIteratorString'] == '')):
            raise ValueError('To process multiple files, both min and max ' +
                             'file iterator should be specified.')
        else:
            arguments['ProcessMultipleFiles'] = True

    return arguments


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

    # Diffusivity
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

    # Exclude land
    help_exclude_land = """
    Determines whether to exclude land from ocean. The should be an integer
    with the following values (default: %(default)s):

    - 0: Does not exclude land from ocean. All data are treated as in the
         ocean.

    - 1: Excludes ocean and land. Most accurate, slowest.

    - 2: Excludes ocean and land. Less accurate, fastest.

    - 3: Excludes ocean and land. Currently this option is not working.
    """
    optional.add_argument('-L', choices=[0, 1, 2, 3], default=0,
                          metavar='EXCLUDE_LAND', help=help_exclude_land)

    # Include near shore
    help_include_shore = """
    Includes the ocean area between data domain (convex/concave hull) and the
    shore. This fills the gap up to the coast. This is only effective if '-L'
    is used so that the land is separated to the ocean.
    """
    optional.add_argument('-l', action="store_true", help=help_include_shore)

    # Convex
    help_convex = """
    Instead of using the concave hull (alpha shape) around the data points,
    this options uses convex hull of the area around the data points.
    """
    optional.add_argument('-c', action="store_true", help=help_convex)

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
    help_refine = """
    Refines the grid by increasing the number of points on each axis by a
    multiple of a given integer. If this option is set to 1, no refinement is
    performed. If set to integer n, the number of grid points is refined by
    n^2 times (n times in each axis).
    (default: %(default)s)
    """
    optional.add_argument('-r', type=int, default=1, metavar="REFINE",
                          help=help_refine)

    # Time frame
    help_time = """
    The time frame index in the dataset to process and to plot the uncertainty
    quantification. The index wraps around the total number of time frames. For
    instance, -1 indicates the last time frame.
    (default: %(default)s)
    """
    optional.add_argument('-t', type=int, default=-1, metavar="TIME_INDEX",
                          help=help_time)

    # Uncertainty quantification
    help_uncertainty = """
    Enables uncertainty quantification for the time frame given in option -t.
    This either produces results in output file as given in option -o, or plots
    the results if the option -p is specified.
    """
    optional.add_argument('-u', action="store_true", help=help_uncertainty)

    # Number of ensembles
    help_num_ensembles = """
    Number of ensembles used for uncertainty quantification.
    (default: %(default)s)
    """
    optional.add_argument('-e', type=int, default=1000, metavar="NUM_ENSEMBLE",
                          help=help_num_ensembles)

    # Start file index
    help_start_file = """
    Start file iterator to be used for processing multiple input files. For
    Instance, '-m 003 -n 012' means to read the series of input files with
    iterators 003, 004, ..., 012. If this option is used, the option -n should
    also be given.
    """
    optional.add_argument('-m', type=str, default='', metavar="START_FILE",
                          help=help_start_file)

    # End file index
    help_end_file = """
    End file iterator to be used for processing multiple input files. For
    Instance, '-m 003 -n 012' means to read the series of input files with
    iterators 003, 004, ..., 012. If this option is used, the option -m should
    also be given.
    """
    optional.add_argument('-n', type=str, default='', metavar="END_FILE",
                          help=help_end_file)

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

    # Process arguments
    arguments = process_arguments(parser, args)

    return arguments
