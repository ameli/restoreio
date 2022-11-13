# SPDX-FileCopyrightText: Copyright 2016, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

examples = """
Examples:

1. This example reads a local file and restores all time frames in it. A convex
   hull around data points as the working domain.

   $ %s -i input.ncml -o output.nc -d 20 -s -c

2. In the next example, only the time frame 20 is processed and the results
   will be also plotted.

   $ %s -i input.ncml -o output.nc -d 20 -s -c -t 20 -p

3. In the previous examples, the restoration domain of was inside the convex
   hull around the known data points. In the following example, a concave hull
   is used instead. The shape of the convex hull is controlled by the parameter
   alpha (given by the option -a). Here, we no longer specify the option (-c)
   which was for the convex hull.

   $ %s -i input.ncml -o output.nc -d 20 -s -t 20 -p -a 10

4. In the following example, we exclude the land from the ocean. That is, if a
   part of the concave hull (domain of restoration) intersects land, we exclude
   it. This is done by -L option followed by an integer. Here, -L 2 is the
   fastest method to detect land (but also the least accurate).

   $ %s -i input.ncml -o output.nc -d 20 -s -t 20 -p -a 10 -L 2

5. There might be a gap area between the domain of restoration and the land
   area. By providing -l, this area can be filled.

   $ %s -i input.ncml -o output.nc -d 20 -s -t 20 -p -a 10 -L 2 -l

6. The following example performs uncertainty quantification with 2000
   ensembles for restoring the time frame 20

   $ %s -i input.ncml -o output.nc -d 20 -s -t 20 -p -a 10 -L 2 -l -u -e 2000

7. The input data might be stored in a series of files. In the following
   example, among the files: input001.nc, input002.nc, ..., input012.nc, we
   want to read the third to tenth file, in each, read the time frame 2-, then
   store a single output file as 'output.zip'. To do so, specify one of the
   files names (such as the first file) by the -i option, and specify the
   start and end iterators of the file names with -m and -n, such as by

   $ %s -i input001.nc -o output.zip -d 20 -s -L 1 -l -a 10 -t 20 -u -e 2000 \
           -m 003 -n 010
"""
