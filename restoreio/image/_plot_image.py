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

import matplotlib.pyplot as plt

# Change font family
plt.rc('font', family='serif')

__all__ = ['plot_color_and_grayscale_images']


# ===============================
# Plot Color And Grayscale Images
# ===============================

def plot_color_and_grayscale_images(
        grayscale_image_U,
        grayscale_image_V,
        color_image):
    """
    Plots the followings:
        - Grayscale image for east velocity U
        - Grayscale image for north velocity V
        - Color image with depths U, V and zero.
    """

    fig, ax = plt.subplots(figsize=(19, 6), ncols=3)

    # Grayscale U
    ax[0].imshow(grayscale_image_U, cmap='gray', interpolation='nearest',
                 origin='lower')
    ax[0].title("8-bit grayscale image of East velocity")

    # Grayscale V
    ax[1].imshow(grayscale_image_V, cmap='gray', interpolation='nearest',
                 origin='lower')
    ax[2].title("8-bit grayscale image of north velocity")

    # Color image
    ax[2].imshow(color_image, interpolation='nearest', origin='lower')
    ax[2].title('8-bit 3-channel color image of velocity vector')

    plt.savefig('color_image.png', transparent=True)
    print("color_image.png saved to disk.")
    plt.show()
