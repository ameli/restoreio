# SPDX-FileCopyrightText: Copyright 2016, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# ================
# Global Variables
# ================

# To use these variables across other modules in the same directory:
#   from . import globals
# To use in other directories:
#   from ._server_utils import globals
#
# Use as: globals.terminate = True, etc.

terminate = False   # This variable can be changed depending on -T CLI argument
raise_error = True
