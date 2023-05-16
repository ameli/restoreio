# SPDX-FileCopyrightText: Copyright 2016, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


examples = """
Examples:

    1. Using a url on a remote machine. This will not scan the velocities.
       $ %s -T -i <url>

    2. Using a filename on the local machine. This will not scan the velocities
       $ %s -T -i <filename>

    3. Scan velocities form a url or filename:
       $ %s -T -V -i <url or filename>
"""
