# SPDX-FileCopyrightText: Copyright 2016, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

examples = """
Examples:

    1. Read a local file. Use diffusivity 20, a convex hull around data points,
       without detecting the land points. Since a convex hull is used, there is
       no need to specify the alpha parameter. The output file contains all
       time frames.

       $ %s -i input.ncml -o output.nc -d 20 -s -c -L 0

    2. Same setting as above, except we only plot the time frame 20 without
       going through all time frames. No output file is written, only the
       results are plotted.

       $ %s -i input.ncml -d 20 -s -c -L 0 -p -t 20

    3. Monterey Bay dataset, one *.nc input file, using concave hull with alpha
       10, diffusivity 20 and sweep. We separate ocean with the land (if land
       exists) and only inpaint areas in ocean by using option '-L'. We only
       plot one time frame at frame 102 without processing other time frames.

       $ %s -i /home/user/input.nc -d 20 -s -L 1 -a 10 -p -t 102

    4. Same as above. But we not only exclude the land from the ocean (option
       -L), also we extend the inpainting up to the  coast line by including
       the land to the concave hull (option -l)

       $ %s -i /home/user/input.nc -d 20 -s -L 1 -l -a 10 -p -t 102

    5. Same as above without plotting, but going through all time steps and
       write to output *.nc file, also with refinement

       $ %s -i /home/user/input.nc -o /home/user/output.nc -d 20 -s -L 1 -l \
            -a 10 -r 2

    6. Uncertainty quantification with 2000 ensembles, plotting (no output
       file), at time frame 102

       $ %s -i /home/user/input.nc -d 20 -s -L 1 -l -a 10 -t 102 -u -e 2000 -p

    7. Processing multiple separate files. Suppose we have these input files:
            /home/user/input001.nc
            /home/user/input002.nc
            ...
            /home/user/input012.nc

       and we want to store them all in this output file:
            /home/user/output.zip

       For uncertainty quantification: 
       $ %s -i /home/user/input001.nc -o /home/user/output.zip -d 20 -s -L 1 \
            -l -a 10 -t 102 -u -e 2000 -m 001 -n 012

       For restoration
       $ %s -i /home/user/input001.nc -o /home/user/output.zip -d 20 -s -L 1 \
            -l -a 10 -t 102 -m 001 -n 012
"""
