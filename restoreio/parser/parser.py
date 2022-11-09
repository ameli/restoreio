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

import getopt
import sys
from ..__version__ import __version__

__all__ = ['parse_arguments']


# ===========
# Print Usage
# ===========

def print_usage(exac_name):
    """
    Prints usage.
    """

    usage_string = "Usage: " + exac_name + \
        " -i <input.{nc, ncml}> -o <output.nc> [options]"

    # Options file
    options_filename = "options.txt"
    with open(options_filename, "r") as file:
        options_string = file.read()

    # Example file
    options_filename = "examples.txt"
    with open(options_filename, "r") as file:
        examples_string = file.read()
    examples_string = examples_string % (exac_name, exac_name, exac_name,
                                         exac_name, exac_name, exac_name,
                                         exac_name, exac_name)

    # Print
    print(usage_string)
    print(options_string)
    print(examples_string)
    print_version()


# =============
# Print Version
# =============

def print_version():

    version_string = "Version: " % __version__
    print(version_string)


# ===============
# Parse arguments
# ===============

def parse_arguments(argv):
    """
    Parses the argument of the executable and obtains the filename.
    """

    # Initialize variables (defaults)
    arguments = {
        'FullPathInputFilename': '',
        'FullPathOutputFilename': '',
        'Diffusivity': 20,
        'SweepAllDirections': False,
        'Plot': False,
        'ExcludeLandFromOcean': 0,
        'IncludeLandForHull': False,
        'UseConvexHull': False,
        'Alpha': -1,
        'RefinementLevel': 1,
        'TimeFrame': -1,
        'UncertaintyQuantification': False,
        'NumEnsembles': 1000,
        "ProcessMultipleFiles": False,
        "MultipleFilesMinIteratorString": '',
        "MultipleFilesMaxIteratorString": ''
    }

    # Get options
    try:
        opts, args = getopt.getopt(
                argv[1:], "hvi:o:d:psL:lca:r:ut:e:m:n:",
                ["help", "version", "input=", "output=", "diffusivity=",
                 "plot", "sweep", "exclude-land=", "include-nearshore",
                 "convex", "alpha=", "refine=", "uncertainty", "time-frame=",
                 "num-ensembles=", "min-file=", "max-file="])

    except getopt.GetoptError:
        print_usage(argv[0])
        sys.exit(2)

    # Assign options
    for opt, arg in opts:

        if opt in ('-h', '--help'):
            print_usage(argv[0])
            sys.exit()
        elif opt in ('-v', '--version'):
            print_version()
            sys.exit()
        elif opt in ("-i", "--input"):
            arguments['FullPathInputFilename'] = arg
        elif opt in ("-o", "--output"):
            arguments['FullPathOutputFilename'] = arg
        elif opt in ('-d', '--diffusivity'):
            arguments['Diffusivity'] = float(arg)
        elif opt in ('-s', '--sweep'):
            arguments['SweepAllDirections'] = True
        elif opt in ('-p', '--plot'):
            arguments['Plot'] = True
        elif opt in ('-L', '--exclude-land'):
            arguments['ExcludeLandFromOcean'] = int(arg)
        elif opt in ('-l', '--include-nearshore'):
            arguments['IncludeLandForHull'] = True
        elif opt in ('-c', '--convex'):
            arguments['UseConvexHull'] = True
        elif opt in ('-a', '--alpha'):
            arguments['Alpha'] = float(arg)
        elif opt in ('-r', '--refine'):
            arguments['RefinementLevel'] = int(arg)
        elif opt in ('-u', '--uncertainty'):
            arguments['UncertaintyQuantification'] = True
        elif opt in ('-t', '--time-frame'):
            arguments['TimeFrame'] = int(arg)
        elif opt in ('-e', '--num-ensembles'):
            arguments['NumEnsembles'] = int(arg)
        elif opt in ('-m', '--min-file'):
            arguments['MultipleFilesMinIteratorString'] = arg
        elif opt in ('-n', '--max-file'):
            arguments['MultipleFilesMaxIteratorString'] = arg

    # Check arguments
    if len(argv) < 2:
        print_usage(argv[0])
        sys.exit()

    # Check InputFilename
    if (arguments['FullPathInputFilename'] == ''):
        print_usage(argv[0])
        print(' ')
        print('Error: No input file is selected with option (-i).')
        sys.exit(2)

    # We can not have empty outputfilename and not plotting.
    if ((arguments['FullPathOutputFilename'] == '') and
            (arguments['Plot'] is False)):
        print_usage(argv[0])
        print(' ')
        print('ERROR: Either the output file should be specified (-o) or ' +
              'the plot option should be enabled (-p).')
        sys.exit(2)

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
