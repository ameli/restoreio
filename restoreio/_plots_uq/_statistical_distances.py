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
import scipy.stats

__all__ = ['kl_distance', 'symmetric_kl_distance', 'js_distance']


# ===========
# KL Distance
# ===========

def kl_distance(
        mean_1,
        mean_2,
        sigma_1,
        sigma_2):
    """
    KL distance of two normal distributions.
    """

    kld = numpy.log(sigma_2/sigma_1) + 0.5 * (sigma_1/sigma_2)**2 + \
        ((mean_1-mean_2)**2)/(2.0*sigma_2**2) - 0.5

    return kld


# =====================
# Symmetric KL Distance
# =====================

def symmetric_kl_distance(
        mean_1,
        mean_2,
        sigma_1,
        sigma_2):
    """
    Symmetric KL distance of two normal distributions.
    """

    skld = 0.5 * (kl_distance(mean_1, mean_2, sigma_1, sigma_2) +
                  kl_distance(mean_2, mean_1, sigma_2, sigma_1))

    return skld


# ===========
# JS Distance
# ===========

def js_distance(
        field_mean_1,
        field_mean_2,
        field_sigma_1,
        field_sigma_2):
    """
    JS distance of two normal distributions.
    """

    field_js_metric = numpy.ma.masked_all(
            field_mean_1.shape, dtype=float)

    for i in range(field_mean_1.shape[0]):
        for j in range(field_mean_1.shape[1]):

            if bool(field_mean_1.mask[i, j]) is False:
                mean_1 = field_mean_1[i, j]
                mean_2 = field_mean_2[i, j]
                sigma_1 = numpy.abs(field_sigma_1[i, j])
                sigma_2 = numpy.abs(field_sigma_2[i, j])

                x = numpy.linspace(numpy.min([mean_1-6*sigma_1,
                                             mean_2-6*sigma_2]),
                                   numpy.max([mean_1+6*sigma_1,
                                             mean_2+6*sigma_2]), 10000)

                norm_1 = scipy.stats.norm.pdf(x, loc=mean_1, scale=sigma_1)
                norm_2 = scipy.stats.norm.pdf(x, loc=mean_2, scale=sigma_2)
                norm_12 = 0.5*(norm_1+norm_2)

                jsd = 0.5 * (scipy.stats.entropy(norm_1, norm_12, base=2) +
                             scipy.stats.entropy(norm_2, norm_12, base=2))

                if jsd < 0.0:
                    if jsd > -1e-8:
                        jsd = 0.0
                    else:
                        print('WARNING: Negative JS distance: %f' % jsd)

                field_js_metric[i, j] = numpy.sqrt(jsd)

    return field_js_metric
