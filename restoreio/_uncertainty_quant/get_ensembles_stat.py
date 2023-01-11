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

__all__ = ['get_ensembles_stat']


# ========================
# Get Ensembles Statistics
# ========================

def get_ensembles_stat(
            land_indices,
            valid_indices,
            missing_indices_in_ocean_inside_hull,
            missing_indices_in_ocean_outside_hull,
            vel_one_time,
            error_vel_one_time,
            vel_all_ensembles_inpainted,
            fill_value):
    """
    Gets the mean and std of all inpainted ensembles in regions where
    inpainted.

    Inputs:

    - vel_one_time:
        The original velocity that is not inpainted, but ony one time-frame of
        it. This is used for its shape and mask, but not its data.

    - Velocity_all_ensembles_inpainted:
        This is the array that we need its data. Ensembles are iterated in the
        first index, i.e

            vel_all_ensembles_inpainted[ensemble_id, lat_index, lon_index]

        The first index vel_all_ensembles_inpainted[0, :] is the central
        ensemble, which is the actual inpainted velocity data in that specific
        timeframe without perturbation.
    """

    # Create a mask for the masked array
    mask = numpy.zeros(vel_one_time.shape, dtype=bool)

    # mask missing points in ocean outside hull
    for i in range(missing_indices_in_ocean_outside_hull.shape[0]):
        mask[missing_indices_in_ocean_outside_hull[i, 0],
             missing_indices_in_ocean_outside_hull[i, 1]] = True

    # mask missing or even valid points on land
    if bool(numpy.any(numpy.isnan(land_indices))) is False:
        for i in range(land_indices.shape[0]):
            mask[land_indices[i, 0], land_indices[i, 1]] = True

    # mask points on land even if they have valid values
    if bool(numpy.any(numpy.isnan(land_indices))) is False:
        for i in range(land_indices.shape[0]):

            # Velocities
            vel_one_time[land_indices[i, 0], land_indices[i, 1]] = \
                    numpy.ma.masked

            # Velocities Errors
            error_vel_one_time[land_indices[i, 0], land_indices[i, 1]] = \
                numpy.ma.masked

    # Initialize Outputs
    vel_one_time_inpainted_stats = {
        'CentralEnsemble': vel_all_ensembles_inpainted[0, :],
        'Mean': numpy.ma.masked_array(vel_one_time, mask=mask,
                                      fill_value=fill_value),
        'AbsoluteError': numpy.ma.masked_array(error_vel_one_time, mask=mask,
                                               fill_value=fill_value),
        'STD': numpy.ma.masked_array(error_vel_one_time, mask=mask,
                                     fill_value=fill_value),
        'RMSD': numpy.ma.masked_all(error_vel_one_time.shape, dtype=float),
        'NRMSD': numpy.ma.masked_all(error_vel_one_time.shape, dtype=float),
        'ExNMSD': numpy.ma.masked_all(error_vel_one_time.shape, dtype=float),
        'Skewness': numpy.ma.masked_all(vel_one_time.shape, dtype=float),
        'ExKurtosis': numpy.ma.masked_all(vel_one_time.shape, dtype=float),
        'Entropy': numpy.ma.masked_all(vel_one_time.shape, dtype=float),
        'RelativeEntropy': numpy.ma.masked_all(vel_one_time.shape, dtype=float)
    }

    # Fill outputs with statistics only at missing_indices_in_ocean_inside_hull
    # or all_missing_indices_in_ocean Note: We exclude the first ensemble since
    # it is the central ensemble and is not coming from the Gaussian
    # distribution. Hence, this preserves the mean, std exactly as it was
    # described for the random Gaussian distribution.
    # Indices = missing_indices_in_ocean_inside_hull
    Indices = numpy.vstack(
            (valid_indices, missing_indices_in_ocean_inside_hull))
    for id in range(Indices.shape[0]):

        # Point Id to Point index
        i, j = Indices[id, :]

        # All ensembles of the point (i, j)
        data_ensembles = vel_all_ensembles_inpainted[1:, i, j]

        # Central ensemble
        central_data = vel_all_ensembles_inpainted[0, i, j]

        # Mean of Velocity
        vel_one_time_inpainted_stats['Mean'][i, j] = \
            numpy.mean(data_ensembles)

        # Absolute Error
        vel_one_time_inpainted_stats['AbsoluteError'][i, j] = \
            error_vel_one_time[i, j]

        # STD of Velocity (Error)
        vel_one_time_inpainted_stats['STD'][i, j] = numpy.std(data_ensembles)

        # Root Mean Square Deviation
        vel_one_time_inpainted_stats['RMSD'][i, j] = \
            numpy.sqrt(numpy.mean((data_ensembles[:] - central_data)**2))

        # Normalized Root Mean Square Deviation
        # vel_one_time_inpainted_stats['NRMSD'][i, j] = numpy.ma.abs(
        #         vel_one_time_inpainted_stats['RMSD'][i, j] / \
        #                 (numpy.fabs(central_data)+1e-10))
        vel_one_time_inpainted_stats['NRMSD'][i, j] = \
            numpy.ma.abs(vel_one_time_inpainted_stats['RMSD'][i, j] /
                         vel_one_time_inpainted_stats['STD'][i, j])

        # Excess Normalized Mean Square Deviation (Ex NMSD)
        vel_one_time_inpainted_stats['ExNMSD'][i, j] = \
            numpy.mean(((data_ensembles[:] - central_data) /
                       vel_one_time_inpainted_stats['STD'][i, j])**2) - 1.0
        # vel_one_time_inpainted_stats['ExNMSD'][i, j] = \
        #         numpy.sqrt(numpy.mean(((data_ensembles[:] - central_data) / \
        #         vel_one_time_inpainted_stats['STD'][i, j])**2)) - 1.0

        # Skewness of Velocity (Error)
        # vel_one_time_inpainted_stats['Skewness'][i, j] = \
        #     scipy.stats.skew(data_ensembles)
        # vel_one_time_inpainted_stats['Skewness'][i, j] = \
        #     numpy.mean(((data_ensembles[:] - \
        #     vel_one_time_inpainted_stats['Mean'][i, j]) / \
        #     vel_one_time_inpainted_stats['STD'][i, j])**3)
        vel_one_time_inpainted_stats['Skewness'][i, j] = \
            numpy.mean(((data_ensembles[:] - central_data) /
                       vel_one_time_inpainted_stats['STD'][i, j])**3)

        # Excess Kurtosis of Velocity (Error) according to Fisher definition
        # (3.0 is subtracted)
        # vel_one_time_inpainted_stats['ExKurtosis'][i, j] = \
        #     scipy.stats.kurtosis(data_ensembles, fisher=True)
        # vel_one_time_inpainted_stats['ExKurtosis'][i, j] = \
        #     numpy.mean(((data_ensembles[:] - \
        #                 vel_one_time_inpainted_stats['Mean'][i, j]) / \
        #                 vel_one_time_inpainted_stats['STD'][i, j])**4) - 3.0
        vel_one_time_inpainted_stats['ExKurtosis'][i, j] = \
            numpy.mean(((data_ensembles[:] - central_data) /
                       vel_one_time_inpainted_stats['STD'][i, j])**4)-3.0

        # Entropy
        # num_bins = 21
        # Counts, Bins = numpy.histogram(data_ensembles, bins=num_bins)
        # PDF = Counts / numpy.sum(Counts, dtype=float)
        # vel_one_time_inpainted_stats['Entropy'][i, j] = \
        #         scipy.stats.entropy(PDF)
        # Only for normal distribution
        vel_one_time_inpainted_stats['Entropy'][i, j] = \
            numpy.log(numpy.std(data_ensembles) *
                      numpy.sqrt(2.0 * numpy.pi * numpy.exp(1)))

        # Relative Entropy
        # Normal = scipy.stats.norm(vel_one_time[i, j],
        #                           error_vel_one_time[i, j])
        # Normal = scipy.stats.norm(numpy.mean(data_ensembles),
        #                           numpy.std(data_ensembles))
        # Normal = scipy.stats.norm(central_data, numpy.std(data_ensembles))
        # Discrete_Normal_PDF = numpy.diff(Normal.cdf(Bins))
        # vel_one_time_inpainted_stats['RelativeEntropy'][i, j] = \
        #         scipy.stats.entropy(PDF, Discrete_Normal_PDF)

        # Only for two normal dist with the same std
        vel_one_time_inpainted_stats['RelativeEntropy'][i, j] = \
            0.5 * ((central_data - numpy.mean(data_ensembles)) /
                   numpy.std(data_ensembles))**2

        # mask zeros
        if numpy.fabs(vel_one_time_inpainted_stats['RMSD'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['RMSD'][i, j] = numpy.ma.masked
        if numpy.fabs(vel_one_time_inpainted_stats['NRMSD'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['NRMSD'][i, j] = numpy.ma.masked
        if numpy.fabs(vel_one_time_inpainted_stats['ExNMSD'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['ExNMSD'][i, j] = numpy.ma.masked
        if numpy.fabs(vel_one_time_inpainted_stats['Skewness'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['Skewness'][i, j] = numpy.ma.masked
        if numpy.fabs(
                vel_one_time_inpainted_stats['RelativeEntropy'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['RelativeEntropy'][i, j] = \
                    numpy.ma.masked

    return vel_one_time_inpainted_stats
