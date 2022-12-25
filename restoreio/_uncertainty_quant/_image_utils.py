# SPDX-FileCopyrightText: Copyright 2016, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# ======
# Import
# ======

import numpy

__all__ = ['convert_image_to_valid_vector', 'convert_valid_vector_to_image']


# =============================
# Convert Image To Valid Vector
# =============================

def convert_image_to_valid_vector(image, valid_indices):
    """
    - image:
      (n, m) 2D array. n is the lat size and m is lon size. Image is
      masked everywhere except where data is valid.

    - valid_indices:
      (num_valid, 2) array where first column is lats indices and second
      column is lon indices. Here num_valid is number of valid points.

    - valid_vector:
      (V_valid, ) vector.
    """

    valid_vector = image[valid_indices[:, 0], valid_indices[:, 1]]
    return valid_vector


# =============================
# Convert Valid Vector To Image
# =============================

def convert_valid_vector_to_image(valid_vector, valid_indices, image_shape):
    """
    - valid_vector:
      (num_valid) is the vector of all valid data values that is vectorized.

    - valid_indices:
      (num_valid, 2) array where first column is lats indices and second
      column is lon indices. Here num_valid is number of valid points.

    - Image:
      (n, m) 2D masked array, n is lat size, and m is lon size.
      Image is masked everywhere except where data is valid.
    """

    image = numpy.ma.masked_all(image_shape, dtype=float)
    for i in range(valid_indices.shape[0]):
        image[valid_indices[i, 0], valid_indices[i, 1]] = valid_vector[i]
    return image
