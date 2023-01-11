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
import pyDOE
from ._compute_correlation_matrix import compute_correlation_matrix
from .._plots_uq import plot_cor_cov, plot_valid_vector_ensembles_stat

__all__ = ['generate_valid_vector_ensembles']


# =========================
# generate Samples On Plane
# =========================

def _generate_samples_on_plane(num_ensembles):
    """
    Number of ensembles is modified to be the closest square number.
    """

    num_ensembles_square_root = int(numpy.sqrt(num_ensembles))
    num_ensembles = num_ensembles_square_root**2

    counter = 0
    samples_on_plane = numpy.empty((num_ensembles, 2), dtype=int)
    for i in range(num_ensembles_square_root):
        for j in range(num_ensembles_square_root):
            samples_on_plane[counter, 0] = num_ensembles_square_root*i + j
            samples_on_plane[counter, 1] = \
                num_ensembles_square_root*(j+1) - (i+1)
            counter += 1

    sorting_index = numpy.argsort(samples_on_plane[:, 0])
    samples_on_plane = samples_on_plane[sorting_index, :]
    permutation = samples_on_plane[sorting_index, 1]

    return permutation


# ==============================================
# Generate Symmetric Mean Latin Hypercube Design
# ==============================================

def _generate_symmetric_mean_latin_hypercube_design(num_modes, num_ensembles):
    """
    Symmetric means it preserves Skewness during isometric rotations.

    Pros:

    1. Makes skewness to be exactly zero, since samples are symmetric around
       zero.
    2. Latin hypercube could have faster convergence. But this can be seem only
       at very large number of samples.

    Cons:

    1. With this implementation, the rank of the matrix of random vectors is
       ONE! Hence, the eigenvalues of the covariance matrix of the random
       vectors is all zero, except one element is nonzero. This is a problem
       if we want to whiten the random vectors with either PCA or ZCA.

    Overall, if you want to use whitening, DO NOT USE THIS METHOD.
    """

    permutation = _generate_samples_on_plane(num_ensembles)
    num_ensembles = permutation.size

    samples_on_hypercube = numpy.empty((num_ensembles, num_modes), dtype=int)
    samples_on_hypercube[:, 0] = numpy.arange(num_ensembles)

    for mode_id in range(1, num_modes):
        samples_on_hypercube[:, mode_id] = permutation[
                samples_on_hypercube[:, mode_id-1]]

    # Values
    sample_uniform_values = 1.0 / (num_ensembles * 2.0) + \
        numpy.linspace(0.0, 1.0, num_ensembles, endpoint=False)
    sample_normal_values = \
        scipy.stats.distributions.norm(loc=0.0, scale=1.0).ppf(
                sample_uniform_values)

    # Values on Hypercube
    sample_normal_values_on_hypercube = numpy.zeros((num_ensembles, num_modes),
                                                    dtype=float)
    for mode_id in range(num_modes):
        sample_normal_values_on_hypercube[:, mode_id] = \
                sample_normal_values[samples_on_hypercube[:, mode_id]]

    return sample_normal_values_on_hypercube.transpose()


# ====================================
# Generate Mean Latin Hypercube Design
# ====================================

def _generate_mean_latin_hypercube_design(num_modes, num_ensembles):
    """
    Latin Hypercube Design (LHS) works as follow:
    1. For each variable, divide the interval [0, 1] to number of ensembles,
       and call each interval a strip. Now, we randomly choose a number in each
       strip. If we use Median LHS, then each sample is chosen on the center
       point of each strip, so this is not really a random selection.
    2. Once for each variable we chose ensembles, we ARRANGE them on a
       hypercube so that in each row/column of the hypercube only one vector of
       samples exists.
    3. The distribution is not on U([0, 1]), which is uniform distribution. We
       use inverse commutation density function (iCDF) to map them in N(0, 1)
       with normal distribution. Now mean = 0, and std=1.

    Output:
    - random_vectors:
        (num_modes, num_ensembles). Each column is one sample of all variables.
        That is each column is one ensemble.

    Notes:
        - The Mean LHS is not really random. Each point is chosen on the center
          of each strip.
        - The MEAN LHS ensures that the mean is zero, and std=1. Since the
          distribution is symmetric in MEAN LHS, the skewness is exactly zero.

    Pros:

    1. Latin hypercube could have faster convergence. But this can be seem only
       at very large number of samples.

    Cons:

    1. Skewness is NOT zero, since samples are not symmetric around zero.

    """

    # Make sure the number of ensembles is more than variables
    # if num_ensembles < num_modes:
    #     print('Number of variables: %d'%num_modes)
    #     print('Number of Ensembles: %d'%num_ensembles)
    #     raise ValueError('In Latin Hypercube sampling, it is better to ' +
    #                      'have more number of ensembles than number of ' +
    #                      'variables.')

    # Mean (center) Latin Hypercube. lhs_uniform is of the
    # size (num_ensembles, num_modes)
    lhs_uniform = pyDOE.lhs(num_modes, samples=num_ensembles,
                            criterion='center')

    # Convert uniform distribution to normal distribution
    lhs_normal = scipy.stats.distributions.norm(loc=0.0, scale=1.0).ppf(
            lhs_uniform)

    # Make output matrix to the form (num_modes, num_ensembles)
    random_vector = lhs_normal.transpose()

    # Make sure mean and std are exactly zero and one
    # for mode_id in range(num_modes):
    #     random_vector[mode_id, :] = random_vector[mode_id, :] - \
    #             numpy.mean(random_vector[mode_id, :])
    #     random_vector[mode_id, :] = random_vector[mode_id, :] / \
    #             numpy.std(random_vector[mode_id, :])

    return random_vector


# =====================================
# Generate Symmetric Monte Carlo Design
# =====================================

def _generate_symmetric_monte_carlo_design(num_modes, num_ensembles):
    """
    Symmetric version of the _generate_monte_carlo_design() function.

    Note:
    About Shuffling the other_half_of_sample:
    For each sample, we create a other_half_of_sample which is opposite of the
    sample. Together they create a full sample that its mean is exactly zero.
    However, the stack of all of these samples that create the random_vectors
    matrix become ill-ranked since we are repeating columns. That is
    random_vectors = [samples, -samples]. There are two cases:

    1. If we want to generate a set of random vectors that any linear
       combination of them still have ZERO SKEWNESS, then all sample rows
       should have this structure, that is random_vectors = [sample, -sample]
       (without shuffling.) However, this matrix is low ranked, and if we want
       to make the random_vectors to have Identity covariance (that is to have
       no correlation, or E[Row_i, Row_j] = 0), then the number of columns
       should be more or equal to twice the number of rows. That is
       num_ensembles >= 2*num_modes.

    2. If we want to generate random_vectors that are exactly decorrelated, and
       if num_ensembles < 2*num_modes, we need to shuffle the
       other_half_of_sample, hence the line for shuffling should be
       uncommented.

    Pros:

    1. Makes skewness to be exactly zero, since samples are symmetric around
       zero.

    Cons:

    1. Slower convergence than Latin hypercube. However, at small number of
       samples, the difference between Monte Carlo Sampling (MCS) and Latin
       hypercube sampling (LHS) is not significant.

    Overall, if you want to use whitening, DO NOT USE THIS METHOD.
    """

    random_vectors = numpy.empty((num_modes, num_ensembles), dtype=float)

    half_num_ensembles = int(num_ensembles / 2.0)

    # Create a continuous normal distribution object with zero mean and unit
    # std.
    normal_distribution = scipy.stats.norm(0.0, 1.0)

    # Generate independent distributions for each variable
    for mode_id in range(num_modes):

        # Generate uniform distributing between [0.0, 0.5)]
        discrete_uniform_distribution = \
                numpy.random.rand(half_num_ensembles) / 2.0

        # Convert the uniform distribution to normal distribution in [0.0, inf)
        # with inverse CDF
        half_of_sample = normal_distribution.isf(discrete_uniform_distribution)

        # Other Half Of samples
        other_half_of_sample = -half_of_sample

        # We need to shuffle, otherwise the RandomVector=[samples, -samples]
        # will be a matrix with similar columns and reduces the rank. This will
        # not produce zero skewness samples after KL expansion. If you need
        # zero skewness, you should comment this line.
        # numpy.random.shuffle(other_half_of_sample)

        sample = numpy.r_[half_of_sample, other_half_of_sample]

        # Add a neutral sample to make number of ensembles odd (if it is
        # supposed to be odd number)
        if num_ensembles % 2 != 0:
            sample = numpy.r_[sample, 0.0]

        sample = sample / numpy.std(sample)
        random_vectors[mode_id, :] = sample[:]

    return random_vectors


# ===========================
# Generate Monte Carlo Design
# ===========================

def _generate_monte_carlo_design(num_modes, num_ensembles):
    """
    Monte Carlo design. This is purely random design.

    Output:
    - random_vectors:
        (num_modes, num_ensembles). Each column is one sample of all variables.
        That is each column is one ensemble.

    In MC, the samples of each variable is selected purely randomly. Also
    samples do not interact between each other variables.

    How we generate random variables:
        We generate random variables on N(0, 1). But since they might not have
        perfect mean and std, we shift and scale them to have exact mean=0 and
        std=1.

    Problem with this method:
        1. The convergence rate of such simulation is O(1/log(n)) where n is
           the number of ensembles.
        2. The distribution's skewness is not exactly zero. Also the kurtosis
           is way away from zero.

    A better option is Latin hypercube design.

    Cons:

    1. Skewness is not zero, since samples are not symmetric around zero.
    2. Slower convergence than Latin hypercube. However, at small number of
       samples, the difference between Monte Carlo Sampling (MCS) and Latin
       hypercube sampling (LHS) is not significant.
    """

    random_vectors = numpy.empty((num_modes, num_ensembles), dtype=float)
    for mode_id in range(num_modes):

        # Generate random samples with Gaussian distribution with mean 0 and
        # std 1.
        sample = numpy.random.randn(num_ensembles)

        # Generate random sample with exactly zero mean and std
        sample = sample - numpy.mean(sample)
        sample = sample / numpy.std(sample)
        random_vectors[mode_id, :] = sample

    return random_vectors


# ===============================
# Generate Valid Vector Ensembles
# ===============================

def generate_valid_vector_ensembles(
        valid_vector,
        valid_vector_error,
        valid_indices,
        num_ensembles,
        num_modes,
        acf_length_scale_lon,
        acf_length_scale_lat,
        quadratic_form,
        vel_component,
        plot=False,
        save=True):
    """
    For a given vector, generates similar vectors with a given vector error.

    Input:
    - valid_vector: (num_valid, ) size
    - valid_vector_error: (num_valid, ) size
    - valid_indices: (num_valid, 2) size

    Output:
    - valid_vector_ensembles: (num_valid, num_ensembles) size

    Each column of the valid_vector_ensembles is an ensemble of valid_vector.
    Each row of valid_vector_ensembles[i, :] has a normal distribution
    N(valid_vector[i], valid_vector_error[i]) that is with
    mean=valid_vector[i] and std=valid_vector_error[i]
    """

    # Correlation
    cor = compute_correlation_matrix(valid_indices, acf_length_scale_lon,
                                     acf_length_scale_lat, quadratic_form)

    # covariance
    sigma = numpy.diag(valid_vector_error)
    cov = numpy.dot(sigma, numpy.dot(cor, sigma))

    if plot:
        plot_cor_cov(cor, cov, vel_component, save)

    # KL Transform of covariance
    eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
    sorting_index = numpy.argsort(eigenvalues)
    sorting_index = sorting_index[::-1]
    eigenvalues = eigenvalues[sorting_index]
    eigenvectors = eigenvectors[:, sorting_index]

    # Check if there is any negative eigenvalues
    negative_indices = numpy.where(eigenvalues < 0.0)
    if negative_indices[0].size > 0:
        for i in range(negative_indices[0].size):
            if eigenvalues[negative_indices[0][i]] > -1e-5:
                eigenvalues[negative_indices[0][i]] = 0.0
            else:
                print('Negative eigenvalue: %f',
                      eigenvalues[negative_indices[0][i]])
                raise RuntimeError('Encountered negative eigenvalue in ' +
                                   'computing KL transform of positive ' +
                                   'definite covariance matrix.')

    # Number of modes for KL expansion
    if num_modes is None:
        # Default num modes
        num_modes = valid_vector.size   # Full number of nodes
        # num_modes = 100

    if num_modes > valid_vector.size:
        raise ValueError('Number of modes cannot be larger than the total' +
                         'number of valid points in the domain. ' +
                         'Num modes: %d. ' % num_modes +
                         'Num valid points: %d.' % valid_vector.size)

    # Generate Gaussian random process for each point (not for each ensemble)
    # with either Monte-Carlo Sampling (MCS) or Latin Hypercube Sampling (LHS)

    # Option 1: MCS, skewness nonzero. Slow convergence. Kurtosis remains zero
    # when whitened (which is good).
    # random_vectors = _generate_monte_carlo_design(num_modes, num_ensembles)

    # Option 2: Symmetric MCS. Skewness zero. Slow convergence. Zero kurtosis
    # becomes 1 when whitened (which is not good if care about kurtosis).
    random_vectors = _generate_symmetric_monte_carlo_design(
            num_modes, num_ensembles)

    # Option 3: LHS, skewness nonzero, better convergence. Kurtosis remains
    # zero when whitened.
    # random_vectors = _generate_mean_latin_hypercube_design(
    #         num_modes, num_ensembles)

    # Option 4: Symmetric LHS. Skewness zero. All eigenvalues of covariance
    # (except one) are zero. Hence, DO NOT WHITEN.
    # random_vectors = _generate_symmetric_mean_latin_hypercube_design(
    #         num_modes, num_ensembles)

    # Num ensembles might be modified if the symmetric mean Hypercube is used.
    num_ensembles_ = random_vectors.shape[1]

    # Decorrelate random vectors (if they still have a correlation)
    # RandomVariables has at least one dim=1 null space since the mean of
    # vectors are zero. Hence
    # to have a full rank, the condition should be num_ensembles > num_modes+1,
    # otherwise we  will have zero eigenvalues.
    if num_ensembles_ > num_modes + 1:

        random_vectors_cor = numpy.dot(
            random_vectors, random_vectors.transpose()) / num_ensembles_
        random_vectors_eig_val, random_vectors_eig_vect = \
            numpy.linalg.eigh(random_vectors_cor)

        # Ignore very small eigenvalues
        random_vectors_eig_val[
                numpy.abs(random_vectors_eig_val + 5e-9) < 5e-9] = 0.0

        if numpy.any(random_vectors_eig_val < -1e-14):
            print('Negative eigenvalues of covariance of random vectors: ')
            print(random_vectors_eig_val[random_vectors_eig_val < 0])
            raise RuntimeError('Random vectors covariance has negative ' +
                               'eigenvalues.')

        # Do not take inverse of zero eigenvalues. Note that the symmetric mean
        # Latin hypercube produces covariance which all eigenvalues are zero,
        # except one positive eigenvalue. Here we filter them out from taking
        # their inverse.
        non_zero_eig = numpy.argwhere(random_vectors_eig_val > 1e-8)

        # Inverse root of eigenvalues for whitening
        eig_whitening = numpy.zeros((random_vectors_eig_val.size, ))
        eig_whitening[non_zero_eig] = \
            1.0 / numpy.sqrt(random_vectors_eig_val[non_zero_eig])
        eig_whitening = numpy.diag(eig_whitening)

        # PCA whitening transformation
        # whitening_matrix  = eig_whitening @ random_vectors_eig_vect.T

        # ZCA whitening transformation
        whitening_matrix = random_vectors_eig_vect @ eig_whitening @ \
            random_vectors_eig_vect.T

        random_vectors = whitening_matrix @ random_vectors
    else:
        print('WARNING: cannot decorrelate random vectors when ' +
              'num_ensembles is less than num_modes. num_modes: ' +
              '%d, num_ensembles: %d' % (num_modes, num_ensembles_))

    # Generate each ensemble with correlations
    valid_vector_ensembles = numpy.empty((valid_vector.size, num_ensembles_),
                                         dtype=float)
    for ensemble_id in range(num_ensembles_):

        # KL expansion
        valid_vector_ensembles[:, ensemble_id] = valid_vector + \
                numpy.dot(eigenvectors[:, :num_modes],
                          numpy.sqrt(eigenvalues[:num_modes]) *
                          random_vectors[:num_modes, ensemble_id])

    # Plot ensembles
    if plot:
        plot_valid_vector_ensembles_stat(
                valid_vector, valid_vector_error, random_vectors,
                valid_vector_ensembles, vel_component, save)

    return valid_vector_ensembles, eigenvalues, eigenvectors
