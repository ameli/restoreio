#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

"""
Before running this script, install distfit: pip install distfit
"""

import netCDF4
import numpy
import scipy
import os
import sys
from distfit import distfit

from _utils._plot_utils._plot_utilities import plt, save_plot, \
    load_plot_settings
from _utils._load_variables import get_datetime_info, get_fill_value, \
    make_array_masked
from _utils._subset import subset_domain, subset_datetime


# ================
# Load Remote Data
# ================

def load_remote_data():
    """
    Loads the u and v velocity data from remote THREDDs
    """

    # Data is obtained from http://hfrnet-tds.ucsd.edu/thredds. In this thredds
    # website, navigate to "HF RADAR, US West Coast", click on
    # "HFRADAR US West Coast 2km Resolution Hourly RTV" and then
    # "Best Time Series". From "Access" methods, select "NetcdfSubset", then
    # extract a subset of data between -122.843 to -121.698 longitudes and
    # 36.3992 to 37.2802 latitudes, from 2017-01-01 00:00:00Z to 2017-02-01
    # 00:00:00Z. After download, rename to the file below.
    # filename = '../files/Monterey_Large_2km_Hourly_2017_01.nc'

    # OpenDap URL of the remote netCDF data (SETTING)
    url = 'http://hfrnet-tds.ucsd.edu/thredds/' + \
          'dodsC/HFR/USWC/2km/hourly/RTV/HFRAD' + \
          'AR_US_West_Coast_2km_Resolution_Hou' + \
          'rly_RTV_best.ncd'

    nc = netCDF4.Dataset(url)
    datetime_obj = nc.variables['time']
    lon_obj = nc.variables['lon']
    lat_obj = nc.variables['lat']
    east_vel_obj = nc.variables['u']
    north_vel_obj = nc.variables['v']

    # Get fill value of the east and north velocity objects
    fill_value = get_fill_value(east_vel_obj, north_vel_obj)

    # Get datetime info from datetime netcdf object
    datetime_info = get_datetime_info(datetime_obj)

    # Subset settings (SETTING)
    time = ''
    min_time = '2017-01-01T00:00:00'
    max_time = '2017-02-01T00:00:00'

    # Monterey Bay domain (SETTING)
    min_lon = -122.344
    max_lon = -121.781
    min_lat = 36.507
    max_lat = 36.992

    # Subset time
    min_datetime_index, max_datetime_index = subset_datetime(
        datetime_info, min_time, max_time, time)

    # Subset domain
    min_lon_index, max_lon_index, min_lat_index, max_lat_index = \
        subset_domain(lon_obj, lat_obj, min_lon, max_lon, min_lat, max_lat)

    # Subset velocity
    u = east_vel_obj[
            min_datetime_index:max_datetime_index+1,
            min_lat_index:max_lat_index+1,
            min_lon_index:max_lon_index+1]
    v = north_vel_obj[
            min_datetime_index:max_datetime_index+1,
            min_lat_index:max_lat_index+1,
            min_lon_index:max_lon_index+1]

    # Make velocity arrays masked
    u = make_array_masked(u, fill_value)
    v = make_array_masked(v, fill_value)

    return u, v


# ===============
# Load Local Data
# ===============

def load_local_data():
    """
    Loads u and v velocity data from local file.
    """

    dir = '../data/'
    file = 'Monterey_Small_2km_Hourly_2017_01.nc'
    filename = os.path.join(dir, file)
    nc = netCDF4.Dataset(filename)
    u = nc.variables['u'][:, :, :]
    v = nc.variables['v'][:, :, :]

    return u, v


# =======
# Laplace
# =======

def laplace(x, a):
    """
    Laplace distribution.

    This function returns the logarithm of the distribution. This is useful to
    fit the tails of the distribution as effective as the center of the
    distribution. Without the log of the distribution, the tail (which are
    very close to zero value) do not produce much residue in the least squares
    method, hence the curve fitting do not fit the tails well.
    """

    f_ = (1.0 / (2.0 * a)) * numpy.exp(-numpy.abs(x) / a)
    return numpy.log(f_)


# ==============
# Variance Gamma
# ==============

def variance_gamma(x, a, la=0.5):
    """
    Variance-Gamma distribution.
    https://en.wikipedia.org/wiki/Variance-gamma_distribution

    The arguments beta (b) and mu (mean) are assumed to be zero. Also, the
    variable la is lambda, and g is gamma.

    This function returns the logarithm of the distribution. This is useful to
    fit the tails of the distribution as effective as the center of the
    distribution. Without the log of the distribution, the tail (which are
    very close to zero value) do not produce much residue in the least squares
    method, hence the curve fitting do not fit the tails well.
    """

    b = 0.0
    g = numpy.sqrt(a**2 - b**2)
    coeff_num = (g**(2.0*la)) * (numpy.abs(x)**(la-0.5))
    coeff_den = numpy.sqrt(numpy.pi) * scipy.special.gamma(la) * \
        (2.0*a)**(la-0.5)
    f_ = (coeff_num / coeff_den) * scipy.special.kv(la-0.5, a*numpy.abs(x))

    return numpy.log(f_)


# =================
# Generalized Gamma
# =================

def generalized_gamma(x, a, b, d):
    """
    Generalized Gamma distribution.
    https://en.wikipedia.org/wiki/Generalized_gamma_distribution

    This function returns the logarithm of the distribution. This is useful to
    fit the tails of the distribution as effective as the center of the
    distribution. Without the log of the distribution, the tail (which are
    very close to zero value) do not produce much residue in the least squares
    method, hence the curve fitting do not fit the tails well.
    """

    a = a * numpy.sqrt(2)
    f_ = 0.5 * (b/(a**d)) * (numpy.abs(x)**(d-1.0)) * \
        numpy.exp(-(numpy.abs(x)/a)**b) / scipy.special.gamma(d/b)

    return numpy.log(f_)


# ==================
# Generalized Normal
# ==================

def generalized_normal(x, a, b):
    """
    Generalized Normal distribution.
    https://en.wikipedia.org/wiki/Generalized_normal_distribution

    This function returns the logarithm of the distribution. This is useful to
    fit the tails of the distribution as effective as the center of the
    distribution. Without the log of the distribution, the tail (which are
    very close to zero value) do not produce much residue in the least squares
    method, hence the curve fitting do not fit the tails well.
    """

    a = a * numpy.sqrt(2)
    f_ = (b / (2 * a * scipy.special.gamma(1.0/b))) * \
        numpy.exp(-(numpy.abs(x)/a)**b)

    return numpy.log(f_)


# =========
# Student T
# =========

def student_t(x, nu):
    """
    Student T distribution.

    This function returns the logarithm of the distribution. This is useful to
    fit the tails of the distribution as effective as the center of the
    distribution. Without the log of the distribution, the tail (which are
    very close to zero value) do not produce much residue in the least squares
    method, hence the curve fitting do not fit the tails well.
    """

    coeff_num = scipy.special.gamma((nu + 1.0) / 2.0)
    coeff_den = numpy.sqrt(nu * numpy.pi) * scipy.special.gamma(nu / 2.0)
    f_ = (coeff_num / coeff_den) * (1.0 + x**2 / nu)**(-(nu + 1.0) / 2.0)

    return numpy.log(f_)


# ==========
# hyperbolic
# ==========

def hyperbolic(x, a, d):
    """
    Hyperbolic distribution.

    This function returns the logarithm of the distribution. This is useful to
    fit the tails of the distribution as effective as the center of the
    distribution. Without the log of the distribution, the tail (which are
    very close to zero value) do not produce much residue in the least squares
    method, hence the curve fitting do not fit the tails well.
    """

    b = 0.0
    c = numpy.sqrt(a**2 - b**2)

    coeff = c / (2.0 * a * d * scipy.special.kv(1, c * d))
    exponent = -a * numpy.sqrt(d**2 + x**2) + b * x
    f_ = coeff * numpy.exp(exponent)

    return numpy.log(f_)


# ============================
# Fit Generalized Distribution
# ============================

def fit_generalized_distribution(
        data,
        ax,
        vel_component,
        dist='generalized_normal',
        threshold_fit=2.58,
        threshold_plot=2.58):
    """
    Fits either the generalized normal or generalized Gamma distribution.
    To run this function, install distfit package: pip install distfit

    The threshold_fit causes to fit only tail of data outside of a z-score.
    Set threshold_fit to 2.58 to use the data in the domain |x| > 2.58, which
    corresponds to the region outside the 99% confidence interval of the
    normal distribution. If set to zero, all data is used to fit the
    distribution.

    The threshold_plot is restricts the domain of the plot of the fitted
    distribution. If set to zero, the domain is all the real line. If set
    to 2.58, the domain of plot is |x| > 2.58, corresponding to the outside
    region of the 99% confidence interval of the normal distribution.

    The 'dist' argument can be either 'generalized_normal', 'variance_gamma',
    'generalized_gamma', 'student_t', 'hyperbolic', or 'laplace'.
    """

    dfit = distfit(smooth=10)
    bins, density = dfit.density(data)

    density_zeros = density == 0.0
    bins_hist = bins[~density_zeros]
    density_hist = density[~density_zeros]

    # Compute the integral of the density function (should be one)
    integral = scipy.integrate.cumtrapz(density_hist, bins_hist, initial=0)[-1]

    # Normalize the density
    density_hist /= integral

    if dist == 'generalized_normal':
        p0 = [1, 2]
        tail_index_pos = 1
        func = generalized_normal
    elif dist == 'variance_gamma':
        p0 = [1, 0.5]
        tail_index_pos = 1
        func = variance_gamma
    elif dist == 'generalized_gamma':
        p0 = [1, 2, 1]
        tail_index_pos = 1
        func = generalized_gamma
    elif dist == 'student_t':
        p0 = [20]
        tail_index_pos = 0
        func = student_t
    elif dist == 'hyperbolic':
        p0 = [0.5, 1.0]
        tail_index_pos = 1
        func = hyperbolic
    elif dist == 'laplace':
        p0 = [1.0]
        tail_index_pos = 0
        func = laplace
    else:
        raise ValueError('"dist" should be either "generalized_normal", ' +
                         '"variance_gamma", "generalized_gamma", ' +
                         '"studnet_t", "hyperbolic", or "laplace".')

    subset = numpy.abs(bins_hist) > threshold_fit
    opt, _ = scipy.optimize.curve_fit(func, bins_hist[subset],
                                      numpy.log(density_hist[subset]), p0=p0,
                                      method='trf', max_nfev=10000)
    print('Parameters: ', end='')
    print(opt)

    # Tail index
    tail_index = int(100*opt[tail_index_pos])/100

    range = numpy.linspace(-6, 6, 1000)

    # The threshold_plot causes the fitted PDF to be plotted outside a z-score
    left = range > threshold_plot
    right = range < -threshold_plot

    # Use numpy.exp to produce the correct density function, since the 'func'
    # functions return the logarithm of the density.
    density_normal = numpy.exp(generalized_normal(range, 1, 2))
    density_fit = numpy.exp(func(range, *opt))

    title_fontsize = 11
    label_fontsize = 10

    ax.plot(bins_hist, density_hist, '.', color='darkgrey', label='Empirical')
    ax.plot(range, density_normal, '-', color='black', label='Gaussian')
    # ax.plot(range, density_fit, '--', color='black', label='Hyperbolic')
    ax.plot(range[left], density_fit[left], '--', color='black',
            label='GGD ($\\beta=' + str(tail_index) + '$)')
    ax.plot(range[right], density_fit[right], '--', color='black')
    ax.legend(fontsize='xx-small', handlelength=1.6)
    ax.grid(True)
    ax.set_xlim([-6, 6])
    ax.set_ylim([1e-4, 1])
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_yscale('log')

    if vel_component == 'east':
        title = '(a) Eastward Velocity PDF'
        xlabel = '$\\hat{v}_e$'
        ylabel = '$p(\\hat{v}_e)$'
    elif vel_component == 'north':
        title = '(b) Northward Velocity PDF'
        xlabel = '$\\hat{v}_n$'
        ylabel = '$p(\\hat{v}_n)$'
    else:
        raise ValueError('"vel_component" should be either "east" or "north".')

    ax.set_title(title, fontdict={'fontsize': title_fontsize})
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(labelsize=label_fontsize)

    ticks = ax.get_xticks()
    new_labels = [f'{int(t)}$\\sigma$' if t != 0 else '0' for t in ticks]
    ax.set_xticks(ticks, new_labels)

    return bins_hist


# ====
# Main
# ====

def main():
    """
    Plots velocity PDF.
    """

    # Load data (choose one, either remote or local data)
    u, v = load_remote_data()
    # u, v = load_local_data()

    # Vectorzize
    uu = u.ravel()
    vv = v.ravel()

    # Remove masked data
    if hasattr(uu, 'mask'):
        uu = uu[~uu.mask].data
    if hasattr(vv, 'mask'):
        vv = vv[~vv.mask].data

    # Remove nans
    uu = uu[~numpy.isnan(uu)]
    vv = vv[~numpy.isnan(vv)]

    # Symmetrize data. This symmetry is not exact, rather even and odd indices
    # are assigned opposite signs, or make the data more symmetrized. (SETTING)
    uu = numpy.r_[uu[::2], -uu[1::2]]
    vv = numpy.r_[vv[::2], -vv[1::2]]

    # Find the angle of the principal direction of the u and v population
    # X = numpy.c_[uu, vv]
    # Cov = (X.T @ X) / X.shape[0]
    # eig_val, eig_vec = numpy.linalg.eigh(Cov)
    # radian = numpy.arctan2(eig_vec[0, 1], eig_vec[0, 0])

    # Rotate (SETTING)
    angle = 0.0
    radian = (angle/180) * numpy.pi
    R = numpy.array([[numpy.cos(radian), -numpy.sin(radian)],
                     [numpy.sin(radian),  numpy.cos(radian)]])
    X = numpy.c_[uu, vv]
    Y = R @ X.T
    uu = Y.T[:, 0]
    vv = Y.T[:, 1]

    # Save remote data to avoid slow connection to THREDDS in repetitive runs
    # with open('vel_data.npy', 'wb') as f:
    #     numpy.save(f, uu)
    #     numpy.save(f, vv)

    # Load (if already saved previously)
    # with open('vel_data.npy', 'rb') as f:
    #     uu = numpy.load(f)
    #     vv = numpy.load(f)

    # Normalize east velocity
    uu_m = numpy.nanmean(uu)
    uu_s = numpy.nanstd(uu)
    uu = (uu - uu_m) / uu_s

    # Normalize north velocity
    vv_m = numpy.nanmean(vv)
    vv_s = numpy.nanstd(vv)
    vv = (vv - vv_m) / vv_s

    load_plot_settings()
    figsize = (7, 3.45)
    fig, ax = plt.subplots(figsize=figsize, ncols=2)

    # Choose a distribution to fit data to it (SETTING)
    dist = 'generalized_normal'   # Best tail fit, almost close to Laplace
    # dist = 'laplace'            # Similar to GGD, with with tail index 1
    # dist = 'variance_gamma'
    # dist = 'generalized_gamma'  # Almost x^p part vanishes, only exp remains
    # dist = 'student_t'          # very similar to normal with high d.o.f
    # dist = 'hyperbolic'         # Very well fit for both tail and body

    # Threshold to restrict the z-score of the interval to fit data (SETTING)
    threshold_fit = 2.58

    # Threshold of the z-score to plot the fitted tail (SETTING)
    threshold_plot = 2.58

    # Fir the data t a distribution
    fit_generalized_distribution(uu, ax[0], vel_component='east', dist=dist,
                                 threshold_fit=threshold_fit,
                                 threshold_plot=threshold_plot)
    fit_generalized_distribution(vv, ax[1], vel_component='north', dist=dist,
                                 threshold_fit=threshold_fit,
                                 threshold_plot=threshold_plot)

    plt.tight_layout()
    fig.patch.set_alpha(0)

    filename = 'vel_distribution'
    save_plot(filename, dpi=200, transparent_background=False, pdf=True,
              bbox_extra_artists=None, verbose=False)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
