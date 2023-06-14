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

import sys
from . import globals


# ====================
# Terminate With Error
# ====================

def terminate_with_error(message):
    """
    Terminate gracefully with exit code 1. This is used to print error message
    in Restore website.

    If terminate is True, the python program is exited with code 1. If False,
    only a ValueError is raised, but an interactive python environment is not
    exited.
    """

    if globals.terminate is True:
        # Added the keyword "ERROR: " in the beginning of error message to
        # signal the server to catch the error.
        print('ERROR: ' + message)

        if globals.raise_error is True:
            # In the server, this also terminate the program with a nonzero
            # exit code., but also leaves a trace back of the error.
            raise ValueError(message)
        else:
            # This does not leave a trace back of the error.
            sys.stdout.flush()
            sys.exit(1)
    else:
        raise ValueError(message)
