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

import numpy
from .._plots._plot_utilities import save_plot, plt, load_plot_settings, \
        make_axes_locatable

__all__ = ['plot_rbf_kernel']


# ===============
# Plot RBF Kernel
# ===============

def plot_rbf_kernel(
        quadratic_form,
        kernel_average,
        X,
        window_lon,
        window_lat,
        vel_component,
        save=True):
    """
    Plots both the averaged kernel and its analytic exponential estimate.
    """

    # Print characteristic length scales
    E, V = numpy.linalg.eigh(quadratic_form)
    print('RBF Kernel characteristic length scales: %f, %f'
          % (numpy.sqrt(1.0/E[0]), numpy.sqrt(1.0/E[1])))

    load_plot_settings()

    # fig1, ax1 = plt.subplots(figsize=(6, 5))
    fig1, ax1 = plt.subplots(figsize=(5, 4.17))
    ax1.set_rasterization_zorder(0)

    # Plot kernel with analytic exponential function
    xc = numpy.linspace(-window_lon, window_lon, 1000)
    yc = numpy.linspace(-window_lat, window_lat, 1000)
    xxc, yyc = numpy.meshgrid(xc, yc)
    z = numpy.exp(-0.5*numpy.sqrt(
        X[0]*xxc**2+2.0*X[1]*xxc*yyc+X[2]*yyc**2))
    levels_1 = numpy.linspace(0.0, 1.0, 6)
    cs = ax1.contour(xxc, yyc, z, levels=levels_1, cmap=plt.cm.Greys,
                     vmin=0.0, vmax=1.0)
    ax1.clabel(cs, inline=1, fontsize=13, colors='black')

    # Flip vectors if located on left half-plane
    W = numpy.copy(V)
    if W[0, 0] < 0.0:
        W[:, 0] = -W[:, 0]
    if W[0, 1] < 0.0:
        W[:, 1] = -W[:, 1]

    # Plot arrows annotation
    d = 0.085  # delta
    s = 0.6 * numpy.min([window_lon, window_lat])  # scale
    ax1.arrow(0, 0, s, 0.0, fc='black', ec='black', head_width=0.22,
              head_length=0.22, zorder=10)
    ax1.arrow(0, 0, 0.0, s, fc='black', ec='black', head_width=0.22,
              head_length=0.22, zorder=10)
    ax1.arrow(0, 0, s*W[0, 0], s*W[1, 0], fc='white', ec='white',
              head_width=0.22, head_length=0.22, zorder=10)
    ax1.arrow(0, 0, s*W[0, 1], s*W[1, 1], fc='white', ec='white',
              head_width=0.22, head_length=0.22, zorder=10)
    ax1.text(s*(1+d), s*(0-d), r'$x_1$', color='black', fontsize=15)
    ax1.text(s*(0+d), s*(1-d), r'$x_2$', color='black', fontsize=15)
    ax1.text(s*(W[0, 0]+d), s*(W[1, 0]-d), r'$y_1$', color='white',
             fontsize=15)
    ax1.text(s*(W[0, 1]+d), s*(W[1, 1]-d), r'$y_2$', color='white',
             fontsize=15)

    # Plot kernel function with statistical correlations that we found
    x = numpy.arange(-window_lon, window_lon+1)
    y = numpy.arange(-window_lat, window_lat+1)
    xx, yy = numpy.meshgrid(x, y)
    levels_2 = numpy.linspace(0, 1, 200)
    p = ax1.contourf(xx, yy, kernel_average, levels=levels_2,
                     cmap=plt.cm.Reds, vmin=0.0, vmax=1.0, zorder=-1)
    # Create axes for colorbar that is the same size as the plot axes
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.09)

    fig1.colorbar(p, cax=cax, ticks=[0, 0.5, 1])
    p.set_clim(0, 1)
    ax1.set_xticks([-window_lon, 0, window_lon], fontsize=13)
    ax1.set_yticks([-window_lat, 0, window_lat], fontsize=13)
    ax1.axis('equal')

    ax1.set_xlabel('Longitude Offset', fontsize=14)
    ax1.set_ylabel('Latitude Offset', fontsize=14)

    title = '%s Velocity Data' % vel_component.capitalize()
    if vel_component == 'east':
        title = '(a) ' + title
    else:
        title = '(b) ' + title
    ax1.set_title(title)

    fig1.set_tight_layout(True)
    fig1.patch.set_alpha(0)

    # Save plot
    if save:
        filename = 'rbf_kernel_2d_' + vel_component
        save_plot(filename, transparent_background=False, pdf=True,
                  bbox_extra_artists=None, verbose=True)

    # Plot 3D
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(xx, yy, kernel_average, antialiased=False)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax2.set_title(title, fontsize=16)

    fig2.set_tight_layout(True)
    # fig2.patch.set_alpha(0)

    # Save plot
    if save:
        filename = 'rbf_kernel_3d_' + vel_component
        save_plot(filename, transparent_background=True, pdf=True,
                  bbox_extra_artists=None, verbose=True)
