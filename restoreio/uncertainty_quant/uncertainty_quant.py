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

__all__ = ['generate_image_ensembles', 'get_ensembles_stat',
           'plot_ensembles_stat']


# =============================
# Convert Image To Valid Vector
# =============================

def _convert_image_to_valid_vector(image, valid_indices):
    """
    - image:
      (n, m) 2D array. n is the lat size and m is lon size. Image is
      masked everywhere except where data is valid.

    - valid_indices:
      (N_valid, 2) array where first column is lats indices and second
      column is lon indices. Here N_valid is number of valid points.

    - valid_vector:
      (V_valid, ) vector. 
    """

    valid_vector = image[valid_indices[:, 0], valid_indices[:, 1]]
    return valid_vector


# =============================
# Convert Valid Vector To Image
# =============================

def _convert_valid_vector_to_image(valid_vector, valid_indices, image_shape):
    """
    - valid_vector:
      (N_valid) is the vector of all valid data values that is vectorized.

    - valid_indices:
      (N_valid, 2) array where first column is lats indices and second
      column is lon indices. Here N_valid is number of valid points.

    - Image:
      (n, m) 2D masked array, n is lat size, and m is lon size.
      Image is masked everywhere except where data is valid.
    """

    Image = numpy.ma.masked_all(image_shape, dtype=float)
    for i in range(valid_indices.shape[0]):
        Image[valid_indices[i, 0], valid_indices[i, 1]] = valid_vector[i]
    return Image


# ==============================
# Get Valid Indices For All Axes
# ==============================

def _get_valid_indices_for_all_axes(Array):
    """
    Generates a map between Ids and (TimeIndex, lat_index, lon_index)
    for only valid points. This is for the 3D Array.
    """

    # Along lon
    valid_indices_list_lon = []
    ids_lon = numpy.ones(Array.shape, dtype=int) * (-1)

    counter = 0
    for lat_index in range(Array.shape[0]):
        for lon_index in range(Array.shape[1]):
            if Array.mask[lat_index, lon_index] == False:
                valid_indices_list_lon.append((lat_index, lon_index))
                ids_lon[lat_index, lon_index] = counter
                counter += 1

    valid_indices_lon = numpy.array(valid_indices_list_lon)

    # Along lat
    valid_indices_list_lat = []
    ids_lat = numpy.ones(Array.shape, dtype=int) * (-1)

    counter = 0
    for lon_index in range(Array.shape[1]):
        for lat_index in range(Array.shape[0]):
            if Array.mask[lat_index, lon_index] == False:
                valid_indices_list_lat.append((lat_index, lon_index))
                ids_lat[lat_index, lon_index] = counter
                counter += 1

    valid_indices_lat = numpy.array(valid_indices_list_lat)

    return valid_indices_lon, ids_lon, valid_indices_lat, ids_lat

# =======================================
# Compute AutoCorrelation Of Valid Vector
# =======================================

def _compute_autocorrelation_of_valid_vector(valid_vector):
    """
    acf is computed from a vectorized data. The N-dimensional data is
    vectorized to be like a time series. Data is assumed periodic, so the
    indices are rotating around the size of the vector. The acf is then a
    simple shift in the 1D data, like
    
    acf(shift) = sum_{i=1}^N time_series(i)*time_series(i+shift)

    acf is normalized, so that acf[0] = 1, by acf =/ acf[0].
    """

    N_valid = valid_vector.size
    time_series = valid_vector - numpy.mean(valid_vector)

    # Limit the size of acf shift
    acf_size = int(N_valid / 100)
    if acf_size < 10: acf_size = 10
    if acf_size > 100: acf_size = 100

    acf = numpy.zeros((acf_size, ), dtype=float)

    for i in range(acf_size):
        for j in range(N_valid):
            acf[i] += time_series[j]*time_series[(j+i) % N_valid]

    # Normalize
    acf /= acf[0]

    return acf

# ===================================
# Estimate Autocorrelation RBF Kernel
# ===================================

def _estimate_autocorrelation_rbf_kernel(masked_image_data, valid_indices, Ids, window_lon, window_lat):
    """
    ---------
    Abstract:
    ---------

    The radial Basis Function (RBF) kernel is assumed in this function. We assume the 2D data has
    the following kernel:
        K(ii, jj) = exp(-0.5*sqrt(X.T * quadratic_form * X))
    where X is a vector:
        X = (i-ii, j-jj)
    That is, at the center point (i, j), the X is the distance vector with the shift (ii, jj).
    This function estimates the quadratic_form 2x2 symmetric matrix.

    -------
    Inputs:
    -------

        - masked_image_data: MxN masked data of the east/north velocity field.
        - valid_indices:    (N_valid, 2) array. Each rwo is of the form [lat_index, lon_index].
                           If there are NxM points in the grid, not all of these points have valid velocity
                           data defined. Suppose there are only N_valid points with valid velocities.
                           Each row of valid_indices is the latitude and longitudes of these points.
        - Ids:             (N, M) array of integers. If a point is non-valid, the value is -1, if
                           a point is valid, the value on the array is the Id (row number in valid_indices).
                           This array is used to show how we mapped valid_indices to the grid Ids.
        - window_lon:      Scalar. This is the half window of stencil in lon direction for sweeping the
                           kernel convolution. The rectangular area of the kernel is of size
                           (2*window_lon_1, 2*window_lat+1).
        - window_lat:      Scalar. This is the half window of stencil in lat direction for sweeping the
                           kernel convolution. The rectangular area of the kernel is of size
                           (2*window_lon_1, 2*window_lat+1).

    --------
    Outputs:
    --------

        - quadratic_form:   2x2 symmetric matrix.

    ---------------------------
    How we estimate the kernel:
    ---------------------------

    1. We subtract mean of data form data. Lets call it Data.

    2. For each valid point with Id1 (or lat and lon Indices), we look for all neighbor points
       within the central stencil of size window_lon, window_lat (This is a rectangle of size 2*window_lon+1, 2*window_lat+1),
       If any of these points are valid itself, say point Id2, we compute the correlation
            Correlation of Id1 and Id2 = Data[Id1] * Data[Id2].
       This value is stored in a 2D array of size of kernel, where the center of kernel corresponds to the point with Id1.
       We store this correlation value in the offset index of two points Id1 and Id2 in the Kernel array.

    2. Now we have N_valid 2D kernels for each valid points. We average them all to get a 2D KernelAverage array. We normalize
    this array w.r.t to the center value of this array. So the center of the KernelAverage is 1.0.
    If we plot this array, it should be descending.

    3. We fit an exponential RBG function to the 2D kernel:
        z = exp(-0.5*sqrt(X.T * quadratic_form * X))
       where X = (i-ii, j-jj) is a distance vector form the center of the kernel array.
       Suppose the kernel array is of size (P=2*window_lon+1, Q=2*window_lat_1)
       To do so we take Z = 4.0*(log(z))^2 = X.T * quadratic_form * X.
       Suppose (ii, jj) is the center indices of the kernel array.
       For each point (i, j) in the kernel matrix, we compute

       A = [(i-ii)**2, 2*(i-ii)*(j-jj), (j-jj)**2]  this is array of size (P*Q, 3) for each i, j on the Kernel.
       b = [Z] this is a vector of size (P*Q, ) for each i, j in the kernel.

       Now we find the least square AX=b.
       The quadratic form is

        quadratic_form = [ X[0], X[1] ]
                        [ X[1], X[2] ]

    -----
    Note:
    -----

        - Let Lambda_1 and Lambda_2 be the eigenvalues of Quadratic Form. Then L1 = 1/sqrt(Lambda_1) and L2=1/sqrt(Lambda_2) are
          the characteristic lengths of the RBF kernel. If the quadratic kernel is diagonal, this is essentially the ARM kernel.

        - The eigenvalues should be non-negative. But if they are negative, this is because we chose a large window_lon or window_lat,
          hence the 2D Kernel function is not strictly descending everywhere. To fix this choose  smaller window sizes.
    """

    # Subtract mean of masked Image Data
    Data = numpy.ma.copy(masked_image_data) - numpy.ma.mean(masked_image_data)

    # 3D array that the first index is for each valid point, and the 2 and 3 index creates a 2D matrix
    # that is the autocorrelation of that valid point (i, j) with the nearby point (i+ii, j+jj)
    KernelForAllValidPoints = numpy.ma.masked_all((valid_indices.shape[0], 2*window_lat+1, 2*window_lon+1), dtype=float)

    # Iterate over valid points
    for Id1 in range(valid_indices.shape[0]):
        lat_index_1, lon_index_1 = valid_indices[Id1, :]

        # Sweep the kernel rectangular area to find nearby points to the center point.
        for lat_offset in range(-window_lat, window_lat+1):
            for lon_offset in range(-window_lon, window_lon+1):

                lat_index_2 = lat_index_1 + lat_offset
                lon_index_2 = lon_index_1 + lon_offset

                if (lat_index_2 >= 0) and (lat_index_2 < masked_image_data.shape[0]):
                    if (lon_index_2 >= 0) and (lon_index_2 < masked_image_data.shape[1]):
                        Id2 = Ids[lat_index_2, lon_index_2]

                        if Id2 >= 0:
                            # The nearby point is a valid point. Compute the correlation of points Id1 and Id2 and store in the Kernel
                            KernelForAllValidPoints[Id1, lat_offset+window_lat, lon_offset+window_lon] = Data[lat_index_2, lon_index_2] * Data[lat_index_1, lon_index_1]

    # Average all 2D kernels over the valid points (over the first index)
    KernelAverage = numpy.ma.mean(KernelForAllValidPoints, axis=0)

    # Normalize the kernel w.r.t the center of the kernel. So the center is 1.0 and all other correlations are less that 1.0.
    KernelAverage = KernelAverage / KernelAverage[window_lat, window_lon]

    # Get the gradient of the kernel to find up to where the kernel is descending. We only use kernel in areas there it is descending.
    gradient_kernel_average = numpy.gradient(KernelAverage)

    # Find where on the grid the data is descending (in order to avoid ascending in acf)
    descending = numpy.zeros((2*window_lat+1, 2*window_lon+1), dtype=bool)
    for lat_offset in range(-window_lat, window_lat+1):
        for lon_offset in range(-window_lon, window_lon+1):
            radial = numpy.array([lat_offset, lon_offset])
            Norm = numpy.linalg.norm(radial)
            if Norm > 0:
                radial = radial / Norm
            elif Norm == 0:
                descending[lat_offset+window_lat, lon_offset+window_lon] = True
            Grad = numpy.array([gradient_kernel_average[0][lat_offset+window_lat, lon_offset+window_lon], gradient_kernel_average[1][lat_offset+window_lat, lon_offset+window_lon]])

            if numpy.dot(Grad, radial) < 0.0:
                descending[lat_offset+window_lat, lon_offset+window_lon] = True

    # Construct a Least square matrices
    A_List = []
    b_List = []

    for lat_offset in range(-window_lat, window_lat+1):
        for lon_offset in range(-window_lon, window_lon+1):
            Value = KernelAverage[lat_offset+window_lat, lon_offset+window_lon]
            if Value <= 0.05:
                continue
            elif descending[lat_offset+window_lat, lon_offset+window_lon] == False:
                continue

            a = numpy.zeros((3, ), dtype=float)
            a[0] = lon_offset**2
            a[1] = 2.0*lon_offset*lat_offset
            a[2] = lat_offset**2
            A_List.append(a)
            b_List.append(4.0*(numpy.log(Value))**2)

    # Check length
    if len(b_List) < 3:
        raise RuntimeError('Insufficient number of kernel points. Can not perform least square to estimate kernel.')

    # List to array
    A = numpy.array(A_List)
    b = numpy.array(b_List)

    # Least square
    AtA = numpy.dot(A.T, A)
    Atb = numpy.dot(A.T, b)
    X = numpy.linalg.solve(AtA, Atb)
    quadratic_form = numpy.array([[X[0], X[1]], [X[1], X[2]]])

    # ------------------------
    # Plot RBF Kernel Function
    # ------------------------

    def _plot_rbf_kernel_function():
        """
        Plots both the averaged kernel and its analytic exponential estimate.
        """

        # Print characteristic length scales
        E, V = numpy.linalg.eigh(quadratic_form)
        print('RBF Kernel characteristic length scales: %f, %f'%(numpy.sqrt(1.0/E[0]), numpy.sqrt(1.0/E[1])))

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Change font family
        plt.rcParams["font.family"] = "serif"

        fig1, ax1 = plt.subplots()
        ax1.set_rasterization_zorder(0)

        # Plot kernel with analytic exponential function
        xc = numpy.linspace(-window_lon, window_lon, 1000)
        yc = numpy.linspace(-window_lat, window_lat, 1000)
        xxc, yyc = numpy.meshgrid(xc, yc)
        z = numpy.exp(-0.5*numpy.sqrt(X[0]*xxc**2+2.0*X[1]*xxc*yyc+X[2]*yyc**2))
        Levels1 = numpy.linspace(0.0, 1.0, 6)
        cs = ax1.contour(xxc, yyc, z, levels=Levels1, cmap=plt.cm.Greys, vmin=0.0, vmax=1.0)
        ax1.clabel(cs, inline=1, fontsize=10, color='black')
        # ax1.contour(xxc, yyc, z, cmap=plt.cm.Greys, vmin=0.0, vmax=1.0)

        # Plot kernel function with statistical correlations that we found
        x = numpy.arange(-window_lon, window_lon+1)
        y = numpy.arange(-window_lat, window_lat+1)
        xx, yy = numpy.meshgrid(x, y)
        Levels2 = numpy.linspace(0, 1, 200)
        p = ax1.contourf(xx, yy, KernelAverage, levels=Levels2, cmap=plt.cm.Reds, vmin=0.0, vmax=1.0, zorder=-1)
        cb = fig1.colorbar(p, ticks=[0, 0.5, 1])
        cb.set_clim(0, 1)
        ax1.set_xticks([-window_lon, 0, window_lon])
        ax1.set_yticks([-window_lat, 0, window_lat])

        ax1.set_xlabel('lon offset')
        ax1.set_ylabel('lat offset')
        ax1.set_title('RBF Autocorrelation Kernel')

        # Plot 3D
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        ax2.plot_surface(xx, yy, KernelAverage, antialiased=False)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('RBF Autocorrelation Kernel')

        plt.show()

    # ---------------

    # _plot_rbf_kernel_function()

    return quadratic_form

# =====================================
# Estimate AutoCorrelation Length Scale
# =====================================

def _estimate_autocorrelaton_length_scale(acf):
    """
    Assuming a Markov-1 Stationary process (mean and std do not change over time),
    the autocorrelation function is acf = rho**(d), where d is spatial distance
    between two points.
    """

    # Find up to where the acf is positive
    window = 0
    while (acf[window] > 0.0) and (acf[window] > acf[window+1]):
        window += 1
        if window >= acf.size-1:
            break
    
    if window < 1:
        raise RuntimeError('window of positive acf is not enough to estimate parameter.')

    x = numpy.arange(1, window)
    y = numpy.log(acf[1:window])
    LengthScale = -numpy.mean(x/y)

    return LengthScale

# =======================================
# Auto Correlation ARD Exponential Kernel
# =======================================

def AutoCorrelationARDExponentialKernel(Id1, Id2, valid_indices, acf_length_scale_lon, acf_length_scale_lat):
    """
    Finds the correlation between two points with Id1 and Id2.
    """

    index_J1 = valid_indices[Id1, 0]
    index_J2 = valid_indices[Id2, 0]
    index_I1 = valid_indices[Id1, 1]
    index_I2 = valid_indices[Id2, 1]

    # Autocorrelation
    X = (index_I1 - index_I2) / acf_length_scale_lon
    Y = (index_J1 - index_J2) / acf_length_scale_lat
    acf_Id1_Id2 = numpy.exp(-numpy.sqrt(X**2+Y**2))

    return acf_Id1_Id2

# ===========================
# Auto Correlation RBF Kernel
# ===========================

def AutoCorrelationRBFKernel(Id1, Id2, valid_indices, quadratic_form):
    """
    RBF kernel with quadratic form.
    """

    index_J1 = valid_indices[Id1, 0]
    index_J2 = valid_indices[Id2, 0]
    index_I1 = valid_indices[Id1, 1]
    index_I2 = valid_indices[Id2, 1]
    distance_vector = numpy.array([index_I2-index_I1, index_J2-index_J1])
    quadrature = numpy.dot(distance_vector, numpy.dot(quadratic_form, distance_vector.T))
    acf_Id1_Id2 = numpy.exp(-0.5*numpy.sqrt(quadrature))

    return acf_Id1_Id2

# ==========================
# Compute Correlation Matrix
# ==========================

def _compute_correlation_matrix(valid_indices, acf_length_scale_lon, acf_length_scale_lat, quadratic_form):
    """
    valid_indices is array of size (N_valid, 2)
    Covariance matrix Cor is of size (N_valid, N_valid).
    """

    N_valid = valid_indices.shape[0]
    Cor = numpy.zeros((N_valid, N_valid), dtype=float) 

    for i in range(N_valid):
        for j in range(i, N_valid):
            # Cor[i, j] = AutoCorrelationARDExponentialKernel(i, j, valid_indices, acf_length_scale_lon, acf_length_scale_lat)
            Cor[i, j] = AutoCorrelationRBFKernel(i, j, valid_indices, quadratic_form)
            if i != j:
                Cor[j, i] = Cor[i, j]
    
    return Cor

# ===========================
# Generate Monte Carlo Design
# ===========================

def _generate_monte_carlo_design(num_modes, num_ensembles):
    """
    Monte Carlo design. This is purely random design.

    Output:
    - random_vectors: (num_modes, num_ensembles). Each column is one sample of all variables. That is
                     each column is one ensemble.

    In MC, the samples of each variable is selected purely randomly. Also samples do not interact between
    each other variables.

    How we generate random variables: 
        We generate random variables on N(0, 1). But since they might not have perfect mean and std, we shift and scale them 
        to have exact mean=0 and std=1.

    Problem with this method:
        1. The convergence rate of such simulation is O(1/log(n)) where n is the number of ensembles.
        2. The distribution's skewness is not exactly zero. Also the kurtosis is way away from zero.

    A better option is Latin hypercube design.
    """

    random_vectors = numpy.empty((num_modes, num_ensembles), dtype=float)
    for ModeId in range(num_modes):

        # Generate random samples with Gaussian distribution with mean 0 and std 1.
        Sample = numpy.random.randn(num_ensembles)

        # Generate random sample with exactly zero mean and std
        Sample = Sample - numpy.mean(Sample)
        Sample = Sample / numpy.std(Sample)
        random_vectors[ModeId, :] = Sample

    return random_vectors

# =====================================
# Generate Symmetric Monte Carlo Design
# =====================================

def _generate_symmetric_monte_carlo_design(num_modes, num_ensembles):
    """
    Symmetric version of the _generate_monte_carlo_design() function.

    Note:
    About Shuffling the OtherHalfOfSample:
    For each Sample, we create a OtherHalfOfSample which is opposite of the Sample.
    Together they create a full Sample that its mean is exactly zero.
    However, the stack of all of these samples that create the random_vectors matrix become
    ill-ranked since we are repeating columns. That is random_vectors = [Samples, -Samples].
    There are two cases:

    1. If we want to generate a set of random vectors that any linear combination of them still have ZERO SKEWNESS,
       then all Sample rows should have this structure, that is random_vectors = [Sample, -Sample] (withput shuffling.)
       However, this matrix is low ranked, and if we want to make the random_vectors to have Identity covariance (that is
       to have no correlation, or E[Row_i, Row_j] = 0), then the number of columns should be more or equal to twice the number
       of rows. That is num_ensembles >= 2*num_modes.

    2. If we want to generate random_vectors that are exactly decorrelated, and if num_ensembles < 2*num_modes,
       we need to shuffle the OtherHalfOfSample, hence the line for shuffling should be uncommented.
    """
    random_vectors = numpy.empty((num_modes, num_ensembles), dtype=float)

    Halfnum_ensembles = int(num_ensembles / 2.0)

    # Create a continuous normal distribution object with zero mean and unit std.
    NormalDistribution = scipy.stats.norm(0.0, 1.0)

    # Generate independent distributions for each variable
    for ModeId in range(num_modes):

        # Generate uniform distributing between [0.0, 0.5)]
        DiscreteUniformDistribution = numpy.random.rand(Halfnum_ensembles) / 2.0

        # Convert the uniform distribution to normal distribution in [0.0, inf) with inverse CDF
        HalfOfSample = NormalDistribution.isf(DiscreteUniformDistribution)

        # Other Half Of Samples
        OtherHalfOfSample = -HalfOfSample

        # We need to shuffle, otherwise the RandomVector=[Samples, -Samples] will be a matrix with similar columns and reduces the rank.
        # This will not produce zero skewness samples after KL expansion. If you need zero skewness, you should comment this line.
        # numpy.random.shuffle(OtherHalfOfSample)

        Sample = numpy.r_[HalfOfSample, OtherHalfOfSample]

        # Add a neutral sample to make number of ensembles odd (if it is supposed to be odd number)
        if num_ensembles % 2 != 0:
            Sample = numpy.r_[Sample, 0.0]

        Sample = Sample / numpy.std(Sample)
        random_vectors[ModeId, :] = Sample[:]

    return random_vectors

# ====================================
# Generate Mean Latin Hypercube Design
# ====================================

def _generate_mean_latin_hypercube_design(num_modes, num_ensembles):
    """
    Latin Hypercube Design (LHS) works as follow: 
    1. For each variable, divide the interval [0, 1] to number of ensembles, and call each interval a strip.
       Now, we randomly choose a number in each strip. If we use Median LHS, then each sample is chosen on the center
       point of each strip, so this is not really a random selection.
    2. Once for each variable we chose ensembles, we ARANGE them on a hypercube so that in each row/column of the hypercube
        only one vector of samples exists.
    3. The distribution is not on U([0, 1]), which is uniform distribution. We use inverse commutation density function (iCDF)
       to map them in N(0, 1) with normal distribution. Now mean = 0, and std=1.

    Output:
    - random_vectors: (num_modes, num_ensembles). Each column is one sample of all variables. That is
                     each column is one ensemble.

    Notes:
        - The Mean LHS is not really random. Each point is chosen on the center of each strip.
        - The MEAN LHS ensures that the mean is zero, and std=1. Since the distribution is symmetric in MEAN LHS, the 
          skewness is exactly zero.
    """

    # Make sure the number of ensembles is more than variables
    # if num_ensembles < num_modes:
    #     print('Number of variables: %d'%num_modes)
    #     print('Number of Ensembles: %d'%num_ensembles)
    #     raise ValueError('In Latin Hypercube sampling, it is better to have more number of ensembles than number of variables.')

    # Mean (center) Latin Hypercube. LHS_Uniform is of the size (num_ensembles, num_modes)
    LHS_Uniform = pyDOE.lhs(num_modes, samples=num_ensembles, criterion='center')

    # Convert uniform distribution to normal distribution
    LHS_Normal = scipy.stats.distributions.norm(loc=0.0, scale=1.0).ppf(LHS_Uniform)

    # Make output matrix to the form (num_modes, num_ensembles)
    RandomVector = LHS_Normal.transpose()

    # Make sure mean and std are exactly zero and one
    # for ModeId in range(num_modes):
    #     RandomVector[ModeId, :] = RandomVector[ModeId, :] - numpy.mean(RandomVector[ModeId, :])
    #     RandomVector[ModeId, :] = RandomVector[ModeId, :] / numpy.std(RandomVector[ModeId, :])

    return RandomVector

# ==============================================
# Generate Symmetric Mean Latin Hypercube Design
# ==============================================

def _generate_symmetric_mean_latin_hypercube_design(num_modes, num_ensembles):
    """
    Symmetric means it preserves Skewness during isometric rotations.
    """

    def _generate_samples_on_plane(num_ensembles):
        """
        Number of ensembles is modified to be the closest square number.
        """

        num_ensemblesSquareRoot = int(numpy.sqrt(num_ensembles))
        num_ensembles = num_ensemblesSquareRoot**2

        counter = 0
        SamplesOnPlane = numpy.empty((num_ensembles, 2), dtype=int)
        for i in range(num_ensemblesSquareRoot):
            for j in range(num_ensemblesSquareRoot):
                SamplesOnPlane[counter, 0] = num_ensemblesSquareRoot*i + j
                SamplesOnPlane[counter, 1] = num_ensemblesSquareRoot*(j+1) - (i+1)
                counter += 1

        SortingIndex = numpy.argsort(SamplesOnPlane[:, 0])
        SamplesOnPlane = SamplesOnPlane[SortingIndex, :]
        Permutation = SamplesOnPlane[SortingIndex, 1]

        return Permutation

    # ------------

    Permutation = _generate_samples_on_plane(num_ensembles)
    num_ensembles = Permutation.size

    SamplesOnHypercube = numpy.empty((num_ensembles, num_modes), dtype=int)
    SamplesOnHypercube[:, 0] = numpy.arange(num_ensembles)

    for ModeId in range(1, num_modes):
        SamplesOnHypercube[:, ModeId] = Permutation[SamplesOnHypercube[:, ModeId-1]]

    # Values
    SampleUniformValues = 1.0 / (num_ensembles * 2.0) + numpy.linspace(0.0, 1.0, num_ensembles, endpoint=False)
    SampleNormalValues = scipy.stats.distributions.norm(loc=0.0, scale=1.0).ppf(SampleUniformValues)

    # Values on Hypercube
    SampleNormalValuesOnHypercube = numpy.empty((num_ensembles, num_modes), dtype=float)
    for ModeId in range(num_modes):
        SampleNormalValuesOnHypercube[:, ModeId] = SampleNormalValues[SamplesOnHypercube[:, ModeId]]

    return SampleNormalValuesOnHypercube.transpose()

# ===============================
# Generate Valid Vector Ensembles
# ===============================

def _generate_valid_vector_ensembles(valid_vector, valid_vector_error, valid_indices, num_ensembles, num_modes, acf_length_scale_lon, acf_length_scale_lat, quadratic_form):
    """
    For a given vector, generates similar vectors with a given vector error.

    Input:
    - valid_vector: (N_valid, ) size
    - valid_vector_error: (N_valid, ) size
    - valid_indices: (N_valid, 2) size

    Output:
    - valid_vector_ensembles: (N_valid, num_ensembles) size

    Each column of the valid_vector_ensembles is an ensemble of valid_vector.
    Each row of valid_vector_ensembles[i, :] has a normal distribution N(valid_vector[i], valid_vector_error[i])
    that is with mean=valid_vector[i] and std=valid_vector_error[i]

    """

    # Correlation
    Cor = _compute_correlation_matrix(valid_indices, acf_length_scale_lon, acf_length_scale_lat, quadratic_form)

    # Covariance
    Sigma = numpy.diag(valid_vector_error)
    Cov = numpy.dot(Sigma, numpy.dot(Cor, Sigma))

    # ----------------
    # Plot Cor and Cov
    # ----------------

    def _plot_cor_and_cov():
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        plt.rc('font', family='serif')
        # cmap = plt.cm.YlOrRd_r
        cmap = plt.cm.YlGnBu
        Interp = 'none'
        fig = plt.figure(figsize=(10, 3))
        ax1 = fig.add_subplot(121)
        ax1.set_rasterization_zorder(0)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        mat1 = ax1.matshow(Cor, vmin=0, vmax=1, cmap=cmap, rasterized=True, zorder=-1, interpolation=Interp)
        cb1 = fig.colorbar(mat1, cax=cax1, ticks=numpy.array([0, 0.5, 1]))
        cb1.solids.set_rasterized(True)
        ax1.set_title('Correlation')

        ax2 = fig.add_subplot(122)
        ax2.set_rasterization_zorder(0)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        mat2 = ax2.matshow(Cov, vmin=0, cmap=cmap, rasterized=True, zorder=-1, interpolation=Interp)
        cb2 = fig.colorbar(mat2, cax=cax2)
        cb2.solids.set_rasterized(True)
        ax2.set_title('Covariance')
        plt.show()

    # ------------

    # _plot_cor_and_cov()

    # KL Transform of Covariance
    eigenvalues, eigenvectors = numpy.linalg.eigh(Cov)
    SortingIndex = numpy.argsort(eigenvalues)
    SortingIndex = SortingIndex[::-1]
    eigenvalues = eigenvalues[SortingIndex]
    eigenvectors = eigenvectors[:, SortingIndex]

    # Check if there is any negative eigenvalues
    NegativeIndices = numpy.where(eigenvalues < 0.0)
    if NegativeIndices[0].size > 0:
        for i in range(NegativeIndices[0].size):
            if eigenvalues[NegativeIndices[0][i]] > -1e-5:
                eigenvalues[NegativeIndices[0][i]] = 0.0
            else:
                print('Negative eigenvalue: %f', eigenvalues[NegativeIndices[0][i]])
                raise RuntimeError('Encountered negative eigenvalue in computing KL transform of positive definite covariance matrix.')

    # Number of modes for KL expansion
    if num_modes is None:
        # Default NumMondes
        # num_modes = valid_vector.size   # Full number of nodes
        num_modes = 100

    # Generate Gaussian random process for each point (not for each ensemble)
    # random_vectors = _generate_monte_carlo_design(num_modes, num_ensembles)
    random_vectors = _generate_symmetric_monte_carlo_design(num_modes, num_ensembles)
    # random_vectors = _generate_mean_latin_hypercube_design(num_modes, num_ensembles)
    # random_vectors = _generate_symmetric_mean_latin_hypercube_design(num_modes, num_ensembles)

    # Decorrelate random vectors (if they still have a correlation)
    # RandomVariables has atleast one dim=1 null space since the mean of vectors are zero. Hence
    # to have a full rank, the condition should be num_ensembles > num_modes + 1, otherwise we  will have zero eigenvalues.
    if num_ensembles > num_modes + 1:
        random_vectors_Cor = numpy.dot(random_vectors, random_vectors.transpose()) / num_ensembles
        random_vectors_EigVal, random_vectors_EigVect = numpy.linalg.eigh(random_vectors_Cor)
        # random_vectors = numpy.dot(numpy.diag(1.0/numpy.sqrt(random_vectors_EigVal)), numpy.dot(random_vectors_EigVect.transpose(), random_vectors))                                         # PCA whitening transformation
        random_vectors = numpy.dot(random_vectors_EigVect, numpy.dot(numpy.diag(1.0/numpy.sqrt(random_vectors_EigVal)), numpy.dot(random_vectors_EigVect.transpose(), random_vectors)))          # ZCA whitening transformation
    else:
        print('WARNING: cannot decorrelate RandomVariables when num_ensembles is less than num_modes. num_modes: %d, num_ensembles: %d'%(num_modes, num_ensembles))

    # Generate each ensemble with correlations
    valid_vector_ensembles = numpy.empty((valid_vector.size, num_ensembles), dtype=float)
    for ensemble_id in range(num_ensembles):

        # KL expansion
        valid_vector_ensembles[:, ensemble_id] = valid_vector + numpy.dot(eigenvectors[:, :num_modes], numpy.sqrt(eigenvalues[:num_modes])*random_vectors[:num_modes, ensemble_id])

    # Uncomment for plot
    # _plot_valid_vector_ensembles_statistics(valid_vector, valid_vector_error, random_vectors, valid_vector_ensembles)

    return valid_vector_ensembles, eigenvalues, eigenvectors

# ========================
# Generate Image Ensembles
# ========================

def generate_image_ensembles(lon, lat, masked_image_data, masked_image_dataError, valid_indices, num_ensembles, num_modes):
    """
    Note: The lon and lat is NOT needed for the computation of this function. However, if we want to plot
          the eigenvectors on the map, we need the lons and lats.

    Input:
    - masked_image_data: (n, m) Image array that are partially masked. This is the original velocity data (either u or v velocity)
    - masked_image_dataError: (n, m) Image array that is partially masked. This is the error of original velocity data (either u or v velocity)
    - valid_indices: (N_valid, 2) 1D array. First column are latitudes (i indices) and second column are longitude (j indices) of valid data on
                    velocity arrays and their errors.
    - num_ensembles: The number of output array of ensembles is actually num_ensembles+1, since the first ensemble that we output is the original
                    data itself in order to have the central ensemble in the data.

    Note:
        The first ensemble is the central ensemble, which is the ensemble without perturbation of variables and
        corresponds to the mean of each variable.
    """

    # Convert Image data to vector
    valid_vector = masked_image_data[valid_indices[:, 0], valid_indices[:, 1]]
    valid_vector_error = masked_image_dataError[valid_indices[:, 0], valid_indices[:, 1]]

    # Compute Autocorrelation of data for each axis
    valid_indices_lon, ids_lon, valid_indices_lat, ids_lat = _get_valid_indices_for_all_axes(masked_image_data)
    valid_vector_Lon = masked_image_data[valid_indices_lon[:, 0], valid_indices_lon[:, 1]]
    valid_vector_Lat = masked_image_data[valid_indices_lat[:, 0], valid_indices_lat[:, 1]]
    acf_lon = _compute_autocorrelation_of_valid_vector(valid_vector_Lon)
    acf_lat = _compute_autocorrelation_of_valid_vector(valid_vector_Lat)
    acf_length_scale_lon = _estimate_autocorrelaton_length_scale(acf_lon)
    acf_length_scale_lat = _estimate_autocorrelaton_length_scale(acf_lat)

    print('LengthScales: Lon: %f, Lat: %f'%(acf_length_scale_lon, acf_length_scale_lat))

    # Plot acf
    # _plot_auto_correlation(acf_lon, acf_lat, acf_length_scale_lon, acf_length_scale_lat)

    # window of kernel
    window_lon = 5
    window_lat = 5
    quadratic_form = _estimate_autocorrelation_rbf_kernel(masked_image_data, valid_indices_lon, ids_lon, window_lon, window_lat)

    # Generate ensembles for vector (Note: eigenvalues and eigenvectors are only needed for plotting them)
    valid_vector_ensembles, eigenvalues, eigenvectors = _generate_valid_vector_ensembles(valid_vector, valid_vector_error, valid_indices, num_ensembles, num_modes, acf_length_scale_lon, acf_length_scale_lat, quadratic_form)
    num_ensembles = valid_vector_ensembles.shape[1]

    # Convert back vector to image
    masked_image_data_ensembles = numpy.ma.masked_all((num_ensembles+1, )+masked_image_data.shape, dtype=float)

    # Add the original data to the first ensemble as the central ensemble
    masked_image_data_ensembles[0, :, :] = masked_image_data

    # Add ensembles that are produced by KL expansion with perturbation of variables
    for ensemble_id in range(num_ensembles):
        masked_image_data_ensembles[ensemble_id+1, :, :] = _convert_valid_vector_to_image(valid_vector_ensembles[:, ensemble_id], valid_indices, masked_image_data.shape)

    # Plot eigenvalues and eigenvectors (Uncomment to plot)
    # _plot_kl_transform(lon, lat, eigenvalues, eigenvectors, acf_lon, acf_lat, valid_indices, masked_image_data.shape)

    return masked_image_data_ensembles


# ========================
# Get Ensembles Statistics
# ========================

def get_ensembles_stat(
            LandIndices,
            valid_indices,
            missing_indices_in_ocean_inside_hull,
            missing_indices_in_ocean_outside_hull,
            vel_one_time,
            error_vel_one_time,
            vel_all_ensembles_inpainted,
            fill_value):
    """
    Gets the mean and std of all inpainted ensembles in regions where inpainted.

    Inputs:
        - vel_one_time:
          The original velocity that is not inpainted, but ony one time-frame of it.
          This is used for its shape and mask, but not its data.
        - Velocity_AllEnsembes_Inpainted:
          This is the array that we need its data. Ensembles are iterated in the first index, i.e 
          vel_all_ensembles_inpainted[ensemble_id, lat_index, lon_index]
          The first index vel_all_ensembles_inpainted[0, :] is the central ensemble, 
          which is the actual inpainted velocity data in that specific timeframe without perturbation.
    """

    # Create a mask for the masked array
    mask = numpy.zeros(vel_one_time.shape, dtype=bool)

    # mask missing points in ocean outside hull
    for i in range(missing_indices_in_ocean_outside_hull.shape[0]):
        mask[missing_indices_in_ocean_outside_hull[i, 0], missing_indices_in_ocean_outside_hull[i, 1]] = True

    # mask missing or even valid points on land
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):
            mask[LandIndices[i, 0], LandIndices[i, 1]] = True

    # mask points on land even if they have valid values
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):

            # Velocities
            vel_one_time[LandIndices[i, 0], LandIndices[i, 1]] = numpy.ma.masked

            # Velocities Errors
            error_vel_one_time[LandIndices[i, 0], LandIndices[i, 1]] = numpy.ma.masked

    # Initialize Outputs
    vel_one_time_inpainted_stats = \
    {
        'central_ensemble': vel_all_ensembles_inpainted[0, :],
        'Mean': numpy.ma.masked_array(vel_one_time, mask=mask, fill_value=fill_value),
        'AbsoluteError': numpy.ma.masked_array(error_vel_one_time, mask=mask, fill_value=fill_value),
        'STD': numpy.ma.masked_array(error_vel_one_time, mask=mask, fill_value=fill_value),
        'RMSD': numpy.ma.masked_all(error_vel_one_time.shape, dtype=float),
        'NRMSD': numpy.ma.masked_all(error_vel_one_time.shape, dtype=float),
        'ExNMSD': numpy.ma.masked_all(error_vel_one_time.shape, dtype=float),
        'Skewness': numpy.ma.masked_all(vel_one_time.shape, dtype=float),
        'ExKurtosis': numpy.ma.masked_all(vel_one_time.shape, dtype=float),
        'Entropy': numpy.ma.masked_all(vel_one_time.shape, dtype=float),
        'RelativeEntropy': numpy.ma.masked_all(vel_one_time.shape, dtype=float)
    }

    # Fill outputs with statistics only at missing_indices_in_ocean_inside_hull or all_missing_indices_in_ocean
    # Note: We exclude the first ensemble since it is the central ensemble and is not coming from the Gaussian distribution.
    # Hence, this preserves the mean, std exactly as it was described for the random Gaussian distribution.
    # Indices = missing_indices_in_ocean_inside_hull
    Indices = numpy.vstack((valid_indices, missing_indices_in_ocean_inside_hull))
    for Id in range(Indices.shape[0]):

        # Point Id to Point index
        i, j = Indices[Id, :]

        # All ensembles of the point (i, j)
        data_ensembles = vel_all_ensembles_inpainted[1:, i, j]

        # Central ensemble
        central_data = vel_all_ensembles_inpainted[0, i, j]

        # Mean of Velocity
        vel_one_time_inpainted_stats['Mean'][i, j] = numpy.mean(data_ensembles)

        # Absolute Error
        vel_one_time_inpainted_stats['AbsoluteError'][i, j] = error_vel_one_time[i, j]

        # STD of Velocity (Error)
        vel_one_time_inpainted_stats['STD'][i, j] = numpy.std(data_ensembles)

        # Root Mean Square Deviation
        vel_one_time_inpainted_stats['RMSD'][i, j] = numpy.sqrt(numpy.mean((data_ensembles[:] - central_data)**2))

        # Normalized Root Mean Square Deviation
        # vel_one_time_inpainted_stats['NRMSD'][i, j] = numpy.ma.abs(vel_one_time_inpainted_stats['RMSD'][i, j] / (numpy.fabs(central_data)+1e-10))
        vel_one_time_inpainted_stats['NRMSD'][i, j] = numpy.ma.abs(vel_one_time_inpainted_stats['RMSD'][i, j] / vel_one_time_inpainted_stats['STD'][i, j])

        # Excess Normalized Mean Square Deviation (Ex NMSD)
        vel_one_time_inpainted_stats['ExNMSD'][i, j] = numpy.mean(((data_ensembles[:] - central_data) / vel_one_time_inpainted_stats['STD'][i, j])**2) - 1.0
        # vel_one_time_inpainted_stats['ExNMSD'][i, j] = numpy.sqrt(numpy.mean(((data_ensembles[:] - central_data) / vel_one_time_inpainted_stats['STD'][i, j])**2)) - 1.0

        # Skewness of Velocity (Error)
        # vel_one_time_inpainted_stats['Skewness'][i, j] = scipy.stats.skew(data_ensembles)
        # vel_one_time_inpainted_stats['Skewness'][i, j] = numpy.mean(((data_ensembles[:] - vel_one_time_inpainted_stats['Mean'][i, j])/vel_one_time_inpainted_stats['STD'][i, j])**3)
        vel_one_time_inpainted_stats['Skewness'][i, j] = numpy.mean(((data_ensembles[:] - central_data)/vel_one_time_inpainted_stats['STD'][i, j])**3)

        # Excess Kurtosis of Velocity (Error) according to Fisher definition (3.0 is subtracted)
        # vel_one_time_inpainted_stats['ExKurtosis'][i, j] = scipy.stats.kurtosis(data_ensembles, fisher=True)
        # vel_one_time_inpainted_stats['ExKurtosis'][i, j] = numpy.mean(((data_ensembles[:] - vel_one_time_inpainted_stats['Mean'][i, j])/vel_one_time_inpainted_stats['STD'][i, j])**4) - 3.0
        vel_one_time_inpainted_stats['ExKurtosis'][i, j] = numpy.mean(((data_ensembles[:] - central_data)/vel_one_time_inpainted_stats['STD'][i, j])**4) - 3.0

        # Entropy
        # NumBins = 21
        # Counts, Bins = numpy.histogram(data_ensembles, bins=NumBins)
        # PDF = Counts / numpy.sum(Counts, dtype=float)
        # vel_one_time_inpainted_stats['Entropy'][i, j] = scipy.stats.entropy(PDF)
        vel_one_time_inpainted_stats['Entropy'][i, j] = numpy.log(numpy.std(data_ensembles) * numpy.sqrt(2.0 * numpy.pi * numpy.exp(1)))  # Only for normal distribution

        # Relative Entropy
        # Normal = scipy.stats.norm(vel_one_time[i, j], error_vel_one_time[i, j])
        # Normal = scipy.stats.norm(numpy.mean(data_ensembles), numpy.std(data_ensembles))
        # Normal = scipy.stats.norm(central_data, numpy.std(data_ensembles))
        # Discrete_Normal_PDF = numpy.diff(Normal.cdf(Bins))
        # vel_one_time_inpainted_stats['RelativeEntropy'][i, j] = scipy.stats.entropy(PDF, Discrete_Normal_PDF)

        vel_one_time_inpainted_stats['RelativeEntropy'][i, j] = 0.5 * ((central_data - numpy.mean(data_ensembles)) / numpy.std(data_ensembles))**2  # Only for two normal dist with the same std

        # mask zeros
        if numpy.fabs(vel_one_time_inpainted_stats['RMSD'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['RMSD'][i, j] = numpy.ma.masked
        if numpy.fabs(vel_one_time_inpainted_stats['NRMSD'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['NRMSD'][i, j] = numpy.ma.masked
        if numpy.fabs(vel_one_time_inpainted_stats['ExNMSD'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['ExNMSD'][i, j] = numpy.ma.masked
        if numpy.fabs(vel_one_time_inpainted_stats['Skewness'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['Skewness'][i, j] = numpy.ma.masked
        if numpy.fabs(vel_one_time_inpainted_stats['RelativeEntropy'][i, j]) < 1e-8:
            vel_one_time_inpainted_stats['RelativeEntropy'][i, j] = numpy.ma.masked

    return vel_one_time_inpainted_stats


# ======================================
# Plot Valid Vector Ensembles Statistics
# ======================================

def _plot_valid_vector_ensembles_statistics(
        valid_vector,
        valid_vector_error,
        random_vectors,
        valid_vector_ensembles):
    """
    Compare the mean, std, skewness, kurtosis of ensembles with the generated
    random vectors.
    """

    import scipy.stats
    import matplotlib.pyplot as plt
    m1 = numpy.mean(valid_vector_ensembles, axis=1) - valid_vector
    m2 = numpy.std(valid_vector_ensembles, axis=1) - valid_vector_error
    r2 = numpy.std(random_vectors, axis=1) - 1.0
    m3 = scipy.stats.skew(valid_vector_ensembles, axis=1)
    r3 = scipy.stats.skew(random_vectors, axis=1)
    m4 = scipy.stats.kurtosis(valid_vector_ensembles, axis=1)
    r4 = scipy.stats.kurtosis(random_vectors, axis=1)

    ax1 = plt.subplot(2, 2, 1)
    plt.plot(m1, color='black',
             label='Difference of Mean of ensembles with central ensemble')
    plt.title('Mean Difference')
    plt.xlabel('Point')
    plt.legend()

    ax2 = plt.subplot(2, 2, 2)
    plt.plot(m2, color='black', label='Diff std ensembles with actual error')
    plt.plot(r2, color='red', label='std of generate random vectors')
    plt.title('Standard Deviation Difference')
    plt.xlabel('Point')
    plt.legend()

    ax3 = plt.subplot(2, 2, 3)
    plt.plot(m3, color='black', label='Skewness of ensembles')
    plt.plot(r3, color='red', label='Skewness of generated random vectors')
    plt.xlabel('Point')
    plt.title('Skewness')
    plt.legend()

    ax4 = plt.subplot(2, 2, 4)
    plt.plot(m4, color='black', label='Kurtosis of ensembles')
    plt.plot(r4, color='red', label='Kurtosis of generated random vectors')
    plt.xlabel('Points')
    plt.title('Excess Kurtosis')
    plt.legend()
    plt.show()


# =====================
# Plot Auto Correlation
# =====================

def _plot_auto_correlation(
        acf_lon,
        acf_lat,
        acf_length_scale_lon,
        acf_length_scale_lat):
    """
    Plots ACF.
    """

    import matplotlib.pyplot as plt

    # Change font family
    plt.rcParams["font.family"] = "serif"

    plot_size = 8
    x = numpy.arange(plot_size)
    y1 = numpy.exp(-x/acf_length_scale_lon)
    y2 = numpy.exp(-x/acf_length_scale_lat)

    # Plot
    # plt.plot(acf_lon, '-o', color='blue', label='Eastward autocorrelation')
    # plt.plot(acf_lat, '-o', color='green', label='Northward autocorrelation')
    # plt.plot(x, y1, '--s', color='blue',
    #         label='Eastward exponential kernel fit')
    # plt.plot(x, y2, '--s', color='green',
    #          label='Northward exponential kernel it')
    # plt.plot(numpy.array([x[0], x[-1]]),
    #          numpy.array([numpy.exp(-1), numpy.exp(-1)]), '--')

    plt.semilogy(acf_lon, 'o', color='blue', label='Eastward autocorrelation')
    plt.semilogy(acf_lat, 'o', color='green',
                 label='Northward autocorrelation')
    plt.semilogy(x, y1, '--', color='blue',
                 label='Eastward exponential kernel fit')
    plt.semilogy(x, y2, '--', color='green',
                 label='Northward exponential kernel fit')
    plt.semilogy(numpy.array([x[0], x[plot_size-1]]),
                 numpy.array([numpy.exp(-1), numpy.exp(-1)]), '--')
    plt.semilogy(numpy.array([x[0], x[plot_size-1]]),
                 numpy.array([numpy.exp(-3), numpy.exp(-3)]), '--')

    plt.title('Autocorrelation function')
    plt.xlabel('Shift')
    plt.ylabel('ACF')
    plt.xlim(0, plot_size-1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()


# =================
# Plot KL Transform
# =================

def _plot_kl_transform(
        lon,
        lat,
        eigenvalues,
        eigenvectors,
        acf_lon,
        acf_lat,
        valid_indices,
        image_shape):
    """
    Plots eigenvalues and eigenvectors of the KL transform.
    """

    # Imports
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Change font family
    plt.rcParams["font.family"] = "serif"

    # ----------------
    # Plot eigenvalues
    # ----------------

    def _plot_eigenvalues():
        """
        Plots log log scale of eigenvalues
        """

        # eigenvalues
        fig, ax1 = plt.subplots()
        # ax1.loglog(eigenvalues, color='green', label='eigenvalues')
        ax1.semilogy(eigenvalues, color='green', label='eigenvalues')
        ax1.set_xlabel('Modes number')
        ax1.set_ylabel('eigenvalues', color='green')
        ax1.grid(True)
        ax1.set_xlim([1, eigenvalues.size])
        ax1.set_ylim([1e-5, 2e-1])
        ax1.tick_params('y', colors='green')

        # Commutative eigenvalues
        eigenvaluesCumSum = numpy.cumsum(eigenvalues)
        ax2 = ax1.twinx()
        # ax2.semilogx(eigenvaluesCumSum/eigenvaluesCumSum[-1], color='blue', label='Normalized Cumulative sum')
        ax2.plot(eigenvaluesCumSum/eigenvaluesCumSum[-1], color='blue', label='Normalized Cumulative sum')
        ax2.set_ylabel('Normalized Cumulative Sum', color='blue')
        ax2.set_xlim([1, eigenvalues.size])
        ax2.tick_params('y', colors='blue')
        h1, l1 = ax1.get_legend_handles_labels() # legend for both ax and its twin ax
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='lower center')

        plt.title('Decay of eigenvalues for KL transform')

    # -----------------
    # Plot eigenvectors
    # -----------------

    def _plot_eigenvectors():
        """
        Plot eigenvectors on the map.
        """
 
        # Mesh grid
        lons_grid, lats_grid = numpy.meshgrid(lon, lat)

        # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
        # percent = 0.05   # For Monterey Dataset
        percent = 0.0
        # percent = 0.1     # For Martha Dataset
        lon_offset = percent * numpy.abs(lon[-1] - lon[0])
        lat_offset = percent * numpy.abs(lat[-1] - lat[0])

        min_lon = numpy.min(lon)
        min_lon_with_offset = min_lon - lon_offset
        mid_lon = numpy.mean(lon)
        max_lon = numpy.max(lon)
        max_lon_with_offset = max_lon + lon_offset
        min_lat = numpy.min(lat)
        min_lat_with_offset = min_lat - lat_offset
        mid_lat = numpy.mean(lat)
        max_lat = numpy.max(lat)
        max_lat_with_offset = max_lat + lat_offset

        # --------
        # Draw map
        # --------

        def _draw_map(axis):

            # Basemap (set resolution to 'i' for faster rasterization and 'f' for full resolution but very slow.)
            map = Basemap( \
                    ax = axis, \
                    projection = 'aeqd', \
                    llcrnrlon=min_lon_with_offset, \
                    llcrnrlat=min_lat_with_offset, \
                    urcrnrlon=max_lon_with_offset, \
                    urcrnrlat=max_lat_with_offset, \
                    area_thresh = 0.1, \
                    lon_0 = mid_lon, \
                    lat_0 = mid_lat, \
                    resolution='f')

            # Map features
            map.drawcoastlines()
            # map.drawstates()
            # map.drawcountries()
            # map.drawcounties()

            # Note: We disabled this, sicne it draws ocean with low resolution. Instead, we created a blue facecolor for the axes.
            # map.drawlsmask(land_color='Linen', ocean_color='#C7DCEF', lakes=True, zorder=-2)

            # map.fillcontinents(color='red', lake_color='white', zorder=0)
            map.fillcontinents(color='moccasin')

            # map.bluemarble()
            # map.shadedrelief()
            # map.etopo()

            # lat and lon lines
            # lon_lines = numpy.linspace(numpy.min(lon), numpy.max(lon), 2)
            # lat_lines = numpy.linspace(numpy.min(lat), numpy.max(lat), 2)
            # map.drawparallels(lat_lines, labels=[1, 0, 0, 0], fontsize=10)
            # map.drawmeridians(lon_lines, labels=[0, 0, 0, 1], fontsize=10)

            return map

        # -----------------
        # Plot on Each axis
        # -----------------

        def _plot_on_each_axis(axis, scalar_field, Title):
            """
            This plots in each of left or right axes.
            """

            axis.set_aspect('equal')
            axis.set_rasterization_zorder(0)
            axis.set_facecolor('#C7DCEF')
            Map = _draw_map(axis)

            # Meshgrids
            lons_grid_on_map, lats_grid_on_map= Map(lons_grid, lats_grid)

            # Pcolormesh
            ContourLevels = 200
            Draw = Map.pcolormesh(lons_grid_on_map, lats_grid_on_map, scalar_field, cmap=cm.jet, rasterized=True, zorder=-1)
            # Draw = Map.pcolormesh(lons_grid_on_map, lats_grid_on_map, scalar_field, cmap=cm.jet)
            # Draw = Map.contourf(lons_grid_on_map, lats_grid_on_map, scalar_field, ContourLevels, cmap=cm.jet, rasterized=True, zorder=-1, corner_mask=False)

            # Create axes for colorbar that is the same size as the plot axes
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Colorbar
            cb = plt.colorbar(Draw, cax=cax, ticks=numpy.array([numpy.ma.min(scalar_field), numpy.ma.max(scalar_field)]))
            cb.solids.set_rasterized(True)
            cb.ax.tick_params(labelsize=7)

            axis.set_title(Title, fontdict={'fontsize':7})

        # --------------

        num_rows = 4
        num_columns = 4
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, 10))
        fig.suptitle('Mercer eigenvectors')

        counter = 0
        for axis in axes.flat:
            print('Plotting eigenvector %d'%counter)
            eigenvector_image = _convert_valid_vector_to_image(eigenvectors[:, counter], valid_indices, image_shape)
            _plot_on_each_axis(axis, eigenvector_image, 'Mode %d'%(counter+1))
            counter += 1

    # -------------

    _plot_eigenvalues()
    # _plot_eigenvectors()

    plt.show()

# =========================
# Plot Ensembles Statistics
# =========================

def plot_ensembles_stat(
        lon, \
        lat, \
        valid_indices, \
        missing_indices_in_ocean_inside_hull, \
        U_one_time, \
        V_one_time, \
        error_U_one_time, \
        error_V_one_time, \
        U_all_ensembles_inpainted, \
        V_all_ensembles_inpainted, \
        U_all_ensembles_inpainted_stats, \
        V_all_ensembles_inpainted_stats):
    """
    Plots of ensembles statistics.
    """

    # Imports
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    import matplotlib.colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Change font family
    plt.rc('font', family='serif')

    # Uncomment the next 3 lines for rendering latex
    # plt.rc('text', usetex=True)
    # plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')
    # matplotlib.verbose.level = 'debug-annoying'

    # ----------------
    # Plot convergence
    # ----------------

    def _plot_convergence():

        # Note: The first ensemble is not generated by random process, since it is the central ensemble. So we exclude it.
        num_ensembles = U_all_ensembles_inpainted.shape[0] - 1
        means_U = numpy.zeros((num_ensembles, missing_indices_in_ocean_inside_hull.shape[0]), dtype=float)
        means_V = numpy.zeros((num_ensembles, missing_indices_in_ocean_inside_hull.shape[0]), dtype=float)

        # Mean of ensembles from the second ensemble to the i-th where i varies to the end of array.
        # We do not take to account of the first ensemble since it is the central ensemble and was not generated by Gaussian random process.
        # Also means are obtained from only the points that were inpainted, not from the valid points.
        for Ensemble in range(num_ensembles):
            for PointId in range(missing_indices_in_ocean_inside_hull.shape[0]):
                index_I, index_J = missing_indices_in_ocean_inside_hull[PointId, :]
                means_U[Ensemble, PointId]=numpy.ma.mean(U_all_ensembles_inpainted[1:Ensemble+2, index_I, index_J])
                means_V[Ensemble, PointId]=numpy.ma.mean(V_all_ensembles_inpainted[1:Ensemble+2, index_I, index_J])

        # Difference of each consecutive mean
        diff_means_U = numpy.max(numpy.abs(numpy.diff(means_U, axis=0)), axis=1)
        diff_means_V = numpy.max(numpy.abs(numpy.diff(means_V, axis=0)), axis=1)

        x = numpy.arange(num_ensembles)
        log_x = numpy.log(x[1:])

        log_diff_means_U = numpy.log(diff_means_U)
        log_diff_means_U_Fit = numpy.polyval(numpy.polyfit(log_x, log_diff_means_U, 1), log_x)
        diff_means_U_Fit = numpy.exp(log_diff_means_U_Fit)

        log_diff_means_V = numpy.log(diff_means_V)
        log_diff_means_V_Fit = numpy.polyval(numpy.polyfit(log_x, log_diff_means_V, 1), log_x)
        diff_means_V_Fit = numpy.exp(log_diff_means_V_Fit)

        # Plot
        fig, ax = plt.subplots()
        ax.loglog(diff_means_U, color='lightsteelblue', zorder=0, label='East velocity data')
        ax.loglog(diff_means_U_Fit, color='darkblue', zorder=1, label='Line fit')
        ax.loglog(diff_means_V, color='palegreen', zorder=0, label='North velocity data')
        ax.loglog(diff_means_V_Fit, color='green', zorder=1, label='Line fit')
        ax.set_xlim([x[1], x[-1]])
        ax.grid(True)
        ax.legend(loc='lower left')
        ax.set_xlabel('Ensumbles')
        ax.set_ylabel('max Mean difference')
        plt.title('Convergence of mean of ensembles population')

    # ------------------
    # Plot convergence 2
    # ------------------

    def _plot_convergence2():

        Indices = missing_indices_in_ocean_inside_hull

        # Note: The first ensemble is not generated by random process, since it is the central ensemble. So we exclude it.
        num_ensembles = U_all_ensembles_inpainted.shape[0] - 1

        means_U = numpy.ma.masked_all((num_ensembles, Indices.shape[0]), dtype=float)
        means_V = numpy.ma.masked_all((num_ensembles, Indices.shape[0]), dtype=float)

        U_Data = U_all_ensembles_inpainted[1:, :, :]
        V_Data = V_all_ensembles_inpainted[1:, :, :]

        # Mean of ensembles from the second ensemble to the i-th where i varies to the end of array.
        # We do not take to account of the first ensemble since it is the central ensemble and was not generated by Gaussian random process.
        # Also means are obtained from only the points that were inpainted, not from the valid points.
        for Ensemble in range(num_ensembles):
            print('Ensemble: %d'%Ensemble)
            for PointId in range(Indices.shape[0]):
                index_I, index_J = Indices[PointId, :]
                means_U[Ensemble, PointId]=numpy.ma.mean(U_Data[:Ensemble+1, index_I, index_J])
                means_V[Ensemble, PointId]=numpy.ma.mean(V_Data[:Ensemble+1, index_I, index_J])

        # mean_diff_mean_U = numpy.mean(numpy.abs(means_U - means_U[-1, :]), axis=1)
        # mean_diff_mean_V = numpy.mean(numpy.abs(means_V - means_V[-1, :]), axis=1)

        U_Errors = numpy.empty((Indices.shape[0]), dtype=float)
        V_Errors = numpy.empty((Indices.shape[0]), dtype=float)
        for PointId in range(Indices.shape[0]):
            index_I, index_J = Indices[PointId, :]
            U_Errors[PointId] = U_all_ensembles_inpainted_stats['STD'][0, index_I, index_J]
            V_Errors[PointId] = V_all_ensembles_inpainted_stats['STD'][0, index_I, index_J]
        mean_diff_mean_U = numpy.mean(numpy.abs(means_U - means_U[-1, :])/U_Errors, axis=1)
        mean_diff_mean_V = numpy.mean(numpy.abs(means_V - means_V[-1, :])/V_Errors, axis=1)

        std_diff_mean_U = numpy.std(numpy.abs(means_U - means_U[-1, :])/U_Errors, axis=1)
        std_diff_mean_V = numpy.std(numpy.abs(means_V - means_V[-1, :])/V_Errors, axis=1)


        # Z score for 84% confidence interval
        Z_Score = 1.0

        # Plot
        fig, ax = plt.subplots()

        yu = mean_diff_mean_U
        yv = mean_diff_mean_V

        yu_up = mean_diff_mean_U + Z_Score * std_diff_mean_U
        yu_dn = mean_diff_mean_U - Z_Score * std_diff_mean_U
        yu_dn[yu_dn <= 0.0] = 1e-4
        
        yv_up = mean_diff_mean_V + Z_Score * std_diff_mean_V
        yv_dn = mean_diff_mean_V - Z_Score * std_diff_mean_V
        yv_dn[yv_dn <= 0.0] = 1e-4

        N = yu.size
        x = numpy.arange(1, N+1)

        # -------------
        # Fit Power Law
        # -------------

        def _fit_power_law(x, y):
            """
            Fits data to y = a x^b.
            By taking logarithm, this is a linear regression to find a and b.
            """
            log_x = numpy.log10(x)
            log_y = numpy.log10(y)
            PolyFit = numpy.polyfit(log_x, log_y, 1)
            log_y_fit = numpy.polyval(PolyFit, log_x)
            y_fit = 10**(log_y_fit)
            print("Polyfit slope: %f"%PolyFit[0])
            return y_fit

        # --------

        ax.plot(x, yu, color='blue', label=(r'East velocity data ($\psi = U$)'))
        ax.plot(x, _fit_power_law(x, yu), '--', color='blue', label='East velocity data (fit)')
        ax.fill_between(x, yu_dn, yu_up, color='lightskyblue', label=('84\% (std) bound'))

        ax.plot(x, yv, color='green', label=r'North velocity data ($\psi=V$)')
        ax.plot(x, _fit_power_law(x, yv), '--', color='green', label='North velocity data (fit)')
        ax.fill_between(x, yv_dn, yv_up, color='palegreen', label='84\% (std) bound')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([1e-3, 1])
        ax.legend(loc='lower left')
        ax.set_xlabel(r'Number of ensembles $s=1, \cdots, S$')
        ax.set_ylabel(r'$\frac{|\bar{\psi}_s(\mathbf{x}) - \bar{\psi}_S(\mathbf{x})|}{\sigma(\mathbf{x})}$')
        plt.title('Convergence of mean of ensembles population')

        plt.show()

    # ------------

    # _plot_convergence()
    # _plot_convergence2()

    # -----------------------
    # Plot Spatial Statistics
    # -----------------------
    
    # Mesh grid
    lons_grid, lats_grid = numpy.meshgrid(lon, lat)

    # # All Missing points coordinates
    # all_missing_lon = lons_grid[AllMissingIndices[:, 0], AllMissingIndices[:, 1]]
    # all_missing_lat = lats_grid[AllMissingIndices[:, 0], AllMissingIndices[:, 1]]
    # all_missing_points_coord = numpy.vstack((all_missing_lon, all_missing_lat)).T

    # # Missing points coordinates inside hull
    # missing_lon_inside_hull = lons_grid[MissingIndicesInsideHull[:, 0], MissingIndicesInsideHull[:, 1]]
    # missing_lat_inside_hull = lats_grid[MissingIndicesInsideHull[:, 0], MissingIndicesInsideHull[:, 1]]
    # missing_points_coord_inside_hull = numpy.vstack((missing_lon_inside_hull, missing_lat_inside_hull)).T

    # # Missing points coordinates outside hull
    # missing_lon_outside_hull = lons_grid[MissingIndicesOutsideHull[:, 0], MissingIndicesOutsideHull[:, 1]]
    # missing_lat_outside_hull = lats_grid[MissingIndicesOutsideHull[:, 0], MissingIndicesOutsideHull[:, 1]]
    # missing_points_coord_outside_hull = numpy.vstack((missing_lon_outside_hull, missing_lat_outside_hull)).T

    # # Valid points coordinates
    # valid_lon = lons_grid[valid_indices[:, 0], valid_indices[:, 1]]
    # valid_lat = lats_grid[valid_indices[:, 0], valid_indices[:, 1]]
    # valid_points_coord = numpy.c_[valid_lon, valid_lat]

    # # Land Point Coordinates
    # if numpy.any(numpy.isnan(LandIndices)) == False:
    #     land_lon = lons_grid[LandIndices[:, 0], LandIndices[:, 1]]
    #     land_lat = lats_grid[LandIndices[:, 0], LandIndices[:, 1]]
    #     land_point_coord = numpy.c_[land_lon, land_lat]
    # else:
    #     land_point_coord = numpy.nan

    # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
    percent = 0.05   # For Monterey Dtaaset
    # percent = 0.1     # For Martha Dataet
    lon_offset = percent * numpy.abs(lon[-1] - lon[0])
    lat_offset = percent * numpy.abs(lat[-1] - lat[0])

    min_lon = numpy.min(lon)
    min_lon_with_offset = min_lon - lon_offset
    mid_lon = numpy.mean(lon)
    max_lon = numpy.max(lon)
    max_lon_with_offset = max_lon + lon_offset
    min_lat = numpy.min(lat)
    min_lat_with_offset = min_lat - lat_offset
    mid_lat = numpy.mean(lat)
    max_lat = numpy.max(lat)
    max_lat_with_offset = max_lat + lat_offset

    # --------
    # Draw map
    # --------

    def _draw_map(axis):

        # Basemap (set resolution to 'i' for faster rasterization and 'f' for full resolution but very slow.)
        map = Basemap( \
                ax = axis, \
                projection = 'aeqd', \
                llcrnrlon=min_lon_with_offset, \
                llcrnrlat=min_lat_with_offset, \
                urcrnrlon=max_lon_with_offset, \
                urcrnrlat=max_lat_with_offset, \
                area_thresh = 0.1, \
                lon_0 = mid_lon, \
                lat_0 = mid_lat, \
                resolution='f')

        # Map features
        map.drawcoastlines()
        # map.drawstates()
        # map.drawcountries()
        # map.drawcounties()
        # map.drawlsmask(land_color='Linen', ocean_color='#C7DCEF', lakes=True, zorder=-2)
        # map.fillcontinents(color='red', lake_color='white', zorder=0)
        map.fillcontinents(color='moccasin')

        # map.bluemarble()
        # map.shadedrelief()
        # map.etopo()

        # lat and lon lines
        lon_lines = numpy.linspace(numpy.min(lon), numpy.max(lon), 2)
        lat_lines = numpy.linspace(numpy.min(lat), numpy.max(lat), 2)
        map.drawparallels(lat_lines, labels=[1, 0, 0, 0], fontsize=10)
        map.drawmeridians(lon_lines, labels=[0, 0, 0, 1], fontsize=10)

        return map

    # ----------------
    # Shifted Colormap
    # ----------------

    def _shifted_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero

        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        '''
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        # regular index to compute the colors
        reg_index = numpy.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = numpy.hstack([
            numpy.linspace(0.0, midpoint, 128, endpoint=False),
            numpy.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)

        return newcmap

    # -----------------
    # Plot Scalar Field
    # -----------------

    def _plot_scalar_fields(scalar_field_1, scalar_field_2, Title, colormap, shift_colormap_status, log_norm):
        """
        This creates a figure with two axes.
        The quantity scalar_field_1 is related to the east velocity and will be plotted on the left axis.
        The quantity scalar_field_2 is related to the north velocity and will be plotted on the right axis.

        If ShiftColorMap is True, we set the zero to the center of the colormap range. This is useful if we
        use divergent colormaps like cm.bwr.
        """

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.5, 4))
        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')
        map_1 = _draw_map(axes[0])
        map_2 = _draw_map(axes[1])

        # Resterization. Anything with zorder less than 0 will be rasterized.
        axes[0].set_rasterization_zorder(0)
        axes[1].set_rasterization_zorder(0)

        # Draw Mapscale
        # Index = int(lat.size / 4)
        # x0, y0 = map_11(lon[0], lat[0])
        # x1, y1 = map_11(lon[Index], lat[0])
        # Distance = (x1 - x0) / 1000 # Length of scale in Km
        # Distance = 40 # For Monterey Dataset
        Distance = 5 # For Martha Dataset
        map_1.drawmapscale(min_lon + (max_lon - min_lon) * 0.88, min_lat, mid_lon, mid_lat, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')
        map_2.drawmapscale(min_lon + (max_lon - min_lon) * 0.88, min_lat, mid_lon, mid_lat, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')

        # Meshgrids
        lons_grid_on_map, lats_grid_on_map= map_1(lons_grid, lats_grid)

        # Default arguments for colormap
        if colormap is None:
            colormap = cm.jet
        if shift_colormap_status is None:
            shift_colormap_status = False

        # -----------------
        # Plot on Each axis
        # -----------------

        def _plot_on_each_axis(axis, Map, scalar_field, Title, colormap, log_norm):
            """
            This plots in each of left or right axes.
            """

            # Shift colormap
            if shift_colormap_status == True:
                min = numpy.ma.min(scalar_field)
                max = numpy.ma.max(scalar_field)
                if (min < 0) and (max > 0):
                    mid_point = -min/(max-min)
                    colormap = _shifted_colormap(colormap, start=0.0, midpoint=mid_point, stop=1.0)

            # Pcolormesh
            if log_norm == True:
                # Plot in log scale
                min = numpy.max([numpy.min(scalar_field), 1e-6])
                Draw = Map.pcolormesh(lons_grid_on_map, lats_grid_on_map, scalar_field, cmap=colormap, rasterized=True, zorder=-1, norm=matplotlib.colors.log_norm(vmin=min))
                # Draw = Map.contourf(lons_grid_on_map, lats_grid_on_map, scalar_field, 200, cmap=colormap, corner_mask=False, rasterized=True, zorder=-1)
            else:
                # Do not plot in log scale
                Draw = Map.pcolormesh(lons_grid_on_map, lats_grid_on_map, scalar_field, cmap=colormap, rasterized=True, zorder=-1)

            # Create axes for colorbar that is the same size as the plot axes
            divider = make_axes_locatable(axis)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            cax = divider.append_axes("right", size="5%", pad=0.07)

            # Colorbar
            cb = plt.colorbar(Draw, cax=cax)
            cb.solids.set_rasterized(True)
            cb.set_label('m/s')

            # axis labels
            axis.set_title(Title)
            axis.set_xlabel('lon (degrees)')
            axis.set_ylabel('lat (degrees)')

            # Background blue for ocean
            axis.set_facecolor('#C7DCEF')

        # ---------------

        fig.suptitle(Title)
        _plot_on_each_axis(axes[0], map_1, scalar_field_1, 'East velocity Data', colormap, log_norm)
        _plot_on_each_axis(axes[1], map_2, scalar_field_2, 'North velocity Data', colormap, log_norm)

    # --------------

    # Original (Uninpainted) Data
    ## _plot_scalar_fields(U_one_time, V_one_time, 'Velocities', cm.jet, shift_colormap_status=False, log_norm=False)
    ## _plot_scalar_fields(error_U_one_time, error_V_one_time, 'Velocity Errors', cm.Reds, shift_colormap_status=False, log_norm=False)

    # Central Ensemble
    central_ensemble_east_vel = U_all_ensembles_inpainted_stats['central_ensemble'][0, :]
    central_ensemble_north_vel = V_all_ensembles_inpainted_stats['central_ensemble'][0, :]
    ## _plot_scalar_fields(central_ensemble_east_vel, central_ensemble_north_vel, 'Central ensemble', cm.jet, shift_colormap_status=False, log_norm=False)

    # Mean Difference
    mean_diff_east_vel = numpy.ma.abs((U_all_ensembles_inpainted_stats['Mean'][0, :]-U_all_ensembles_inpainted[0, :])/U_all_ensembles_inpainted_stats['STD'][0, :])
    mean_diff_north_vel = numpy.ma.abs((V_all_ensembles_inpainted_stats['Mean'][0, :]-V_all_ensembles_inpainted[0, :])/V_all_ensembles_inpainted_stats['STD'][0, :])
    mean_diff_east_vel[numpy.ma.where(mean_diff_east_vel < 1e-8)] = numpy.ma.masked
    mean_diff_north_vel[numpy.ma.where(mean_diff_north_vel < 1e-8)] = numpy.ma.masked
    # mean_diff_east_vel = numpy.ma.abs((U_all_ensembles_inpainted_stats['Mean'][0, :]-U_all_ensembles_inpainted_stats['central_ensemble'][0, :])/U_all_ensembles_inpainted_stats['central_ensemble'][0, :])
    # mean_diff_north_vel = numpy.ma.abs((V_all_ensembles_inpainted_stats['Mean'][0, :]-V_all_ensembles_inpainted_stats['central_ensemble'][0, :])/V_all_ensembles_inpainted_stats['central_ensemble'][0, :])
    # mean_diff_east_vel = numpy.ma.abs((U_all_ensembles_inpainted_stats['Mean'][0, :]-U_all_ensembles_inpainted_stats['central_ensemble'][0, :]))
    # mean_diff_north_vel = numpy.ma.abs((V_all_ensembles_inpainted_stats['Mean'][0, :]-V_all_ensembles_inpainted_stats['central_ensemble'][0, :]))
    ##_plot_scalar_fields(mean_diff_east_vel, mean_diff_north_vel, 'Normalized first moment deviation w.r.t central ensemble', cm.Reds, shift_colormap_status=False, log_norm=True)

    # Mean
    std_east_vel = U_all_ensembles_inpainted_stats['Mean'][0, :]
    std_north_vel = V_all_ensembles_inpainted_stats['Mean'][0, :]
    ##_plot_scalar_fields(std_east_vel, std_north_vel, 'Mean of ensembles', cm.jet, shift_colormap_status=False, log_norm=False)

    # STD
    # std_east_vel = numpy.log(U_all_ensembles_inpainted_stats['STD'][0, :])
    # std_north_vel = numpy.log(V_all_ensembles_inpainted_stats['STD'][0, :])
    std_east_vel = U_all_ensembles_inpainted_stats['STD'][0, :]
    std_north_vel = V_all_ensembles_inpainted_stats['STD'][0, :]
    ##_plot_scalar_fields(std_east_vel, std_north_vel, 'Standard deviation of ensembles', cm.Reds, shift_colormap_status=False, log_norm=True)

    # RMSD
    # rmsd_east_vel = numpy.log(U_all_ensembles_inpainted_stats['RMSD'][0, :])
    # rmsd_north_vel = numpy.log(V_all_ensembles_inpainted_stats['RMSD'][0, :])
    rmsd_east_vel = U_all_ensembles_inpainted_stats['RMSD'][0, :]
    rmsd_north_vel = V_all_ensembles_inpainted_stats['RMSD'][0, :]
    ## _plot_scalar_fields(rmsd_east_vel, rmsd_north_vel, 'RMSD of ensembles w.r.t central ensemble', cm.Reds, shift_colormap_status=False, log_norm=True)

    # NRMSD
    # nrmsd_east_vel = numpy.log(U_all_ensembles_inpainted_stats['NRMSD'][0, :])
    # nrmsd_north_vel = numpy.log(V_all_ensembles_inpainted_stats['NRMSD'][0, :])
    nrmsd_east_vel = U_all_ensembles_inpainted_stats['NRMSD'][0, :]
    nrmsd_north_vel = V_all_ensembles_inpainted_stats['NRMSD'][0, :]
    ## _plot_scalar_fields(nrmsd_east_vel, nrmsd_north_vel, 'NRMSD of ensembles w.r.t central ensemble', cm.Reds, shift_colormap_status=False, log_norm=True)

    # Excess Normalized MSD
    # ex_nmsd_east_vel = numpy.log(U_all_ensembles_inpainted_stats['ExNMSD'][0, :])
    # ex_nmsd_north_vel = numpy.log(V_all_ensembles_inpainted_stats['ExNMSD'][0, :])
    ex_nmsd_east_vel = U_all_ensembles_inpainted_stats['ExNMSD'][0, :]
    ex_nmsd_north_vel = V_all_ensembles_inpainted_stats['ExNMSD'][0, :]
    ##_plot_scalar_fields(ex_nmsd_east_vel, ex_nmsd_north_vel, 'Excess Normalized second moment deviation w.r.t central ensemble', cm.Reds, shift_colormap_status=False, log_norm=True)

    # Skewness
    skewness_east_vel = U_all_ensembles_inpainted_stats['Skewness'][0, :]
    skewness_north_vel = V_all_ensembles_inpainted_stats['Skewness'][0, :]
    # Trim = 1.2
    # skewness_east_vel[numpy.ma.where(skewness_east_vel > Trim)] = Trim
    # skewness_east_vel[numpy.ma.where(skewness_east_vel < -Trim)] = -Trim
    # skewness_north_vel[numpy.ma.where(skewness_north_vel > Trim)] = Trim
    # skewness_north_vel[numpy.ma.where(skewness_north_vel < -Trim)] = -Trim
    ##_plot_scalar_fields(skewness_east_vel, skewness_north_vel, 'Normalizsed third moment deviation (Skewness) w.r.t central ensemble', cm.bwr, shift_colormap_status=True, log_norm=False)

    # Excess Kurtosis
    ex_kurtosis_east_vel = U_all_ensembles_inpainted_stats['ExKurtosis'][0, :]
    ex_kurtosis_north_vel = V_all_ensembles_inpainted_stats['ExKurtosis'][0, :]
    ##_plot_scalar_fields(ex_kurtosis_east_vel, ex_kurtosis_north_vel, 'Normalzied fourth moment deviation (Excess Kurtosis) w.r.t central ensemble', cm.bwr, shift_colormap_status=True, log_norm=False)

    # Entropy
    entropy_east_vel = U_all_ensembles_inpainted_stats['Entropy'][0, :]
    entropy_north_vel = V_all_ensembles_inpainted_stats['Entropy'][0, :]
    ## _plot_scalar_fields(entropy_east_vel, entropy_north_vel, 'Entropy of ensembles', cm.RdBu_r, shift_colormap_status=True, log_norm=False)
    # _plot_scalar_fields(entropy_east_vel, entropy_north_vel, 'Entropy of ensembles', cm.coolwarm, True)
    # _plot_scalar_fields(entropy_east_vel, entropy_north_vel, 'Entropy of ensembles', cm.RdYlGn_r, True)

    # Relative Entropy (KL Divergence) with respect to the normal distribution
    relative_entripyeast_vel = U_all_ensembles_inpainted_stats['RelativeEntropy'][0, :]
    relative_entripynorth_vel = V_all_ensembles_inpainted_stats['RelativeEntropy'][0, :]
    # Trim = 0.1
    # relative_entripyeast_vel[numpy.ma.where(relative_entripyeast_vel > Trim)] = Trim
    # relative_entripynorth_vel[numpy.ma.where(relative_entripynorth_vel > Trim)] = Trim
    ## _plot_scalar_fields(relative_entripyeast_vel, relative_entripynorth_vel, 'Kullback-Leibler Divergence of ensembles', cm.Reds, shift_colormap_status=True, log_norm=False)
    # _plot_scalar_fields(relative_entripyeast_vel, relative_entripynorth_vel, 'Kullback-Leibler Divergence of ensembles', cm.YlOrRd, True)

    # -------------------------------
    # JS Distance Of Two Distribution
    # -------------------------------

    def _js_distance_of_two_distributions(filename_1, filename_2):
        """
        Reads two files, and computes the JS metric distance of their east/north velocities.
        The JS metric distance is the square root of JS distance. Log base 2 is used,
        hence the output is in range [0, 1].
        """

        def _kl_distance(mean_1, mean_2, sigma_1, sigma_2):
            """
            KL distance of two normal distributions.
            """
            kld = numpy.log(sigma_2/sigma_1) + 0.5 * (sigma_1/sigma_2)**2 + ((mean_1-mean_2)**2)/(2.0*sigma_2**2) - 0.5
            return kld

        def _symmetric_kl_distance(mean_1, mean_2, sigma_1, sigma_2):
            skld = 0.5 * (_kl_distance(mean_1, mean_2, sigma_1, sigma_2) + _kl_distance(mean_2, mean_1, sigma_2, sigma_1))
            return skld

        def _js_distance(field_mean_1, field_mean_2, Field_sigma_1, Field_sigma_2):
            field_js_metric = numpy.ma.masked_all(field_mean_1.shape, dtype=float)
            for i in range(field_mean_1.shape[0]):
                for j in range(field_mean_1.shape[1]):

                    if field_mean_1.mask[i, j] == False:
                        mean_1 = field_mean_1[i, j]
                        mean_2 = field_mean_2[i, j]
                        sigma_1 = numpy.abs(Field_sigma_1[i, j])
                        sigma_2 = numpy.abs(Field_sigma_2[i, j])

                        x = numpy.linspace(numpy.min([mean_1-6*sigma_1, mean_2-6*sigma_2]), numpy.max([mean_1+6*sigma_1, mean_2+6*sigma_2]), 10000)
                        norm_1 = scipy.stats.norm.pdf(x, loc=mean_1, scale=sigma_1)
                        norm_2 = scipy.stats.norm.pdf(x, loc=mean_2, scale=sigma_2)
                        norm_12 = 0.5*(norm_1+norm_2)
                        jsd = 0.5 * (scipy.stats.entropy(norm_1, norm_12, base=2) + scipy.stats.entropy(norm_2, norm_12, base=2))
                        if jsd < 0.0:
                            if jsd > -1e-8:
                                jsd = 0.0
                            else:
                                print('WARNING: Negative JS distance: %f'%jsd)
                        field_js_metric[i, j] = numpy.sqrt(jsd)
            return field_js_metric

        import netCDF4
        nc_f = netCDF4.Dataset(filename_1)
        nc_t = netCDF4.Dataset(filename_2)

        east_mean_f = nc_f.variables['east_vel'][0, :]
        east_mean_t = nc_t.variables['east_vel'][0, :]
        east_sigma_f = nc_f.variables['east_err'][0, :]
        east_sigma_t = nc_t.variables['east_err'][0, :]
        east_jsd = _js_distance(east_mean_t, east_mean_f, east_sigma_t, east_sigma_f)

        north_mean_f = nc_f.variables['north_vel'][0, :]
        north_mean_t = nc_t.variables['north_vel'][0, :]
        north_sigma_f = nc_f.variables['north_err'][0, :]
        north_sigma_t = nc_t.variables['north_err'][0, :]
        north_jsd = _js_distance(north_mean_t, north_mean_f, north_sigma_t, north_sigma_f)

        return east_jsd, north_jsd

    # --------------------------
    # Ratio of Truncation Energy
    # --------------------------

    def _ratio_of_truncation_energy(filename_1, filename_2):
        """
        Ratio of StandardDeviation^2 for truncated and full KL expansion.
        """

        import netCDF4
        nc_f = netCDF4.Dataset(filename_1)
        nc_t = netCDF4.Dataset(filename_2)

        east_std_f = nc_f.variables['east_err'][0, :]
        east_std_t = nc_t.variables['east_err'][0, :]
        east_energy_ratio = numpy.ma.masked_all(east_std_f.shape, dtype=float)
        east_energy_ratio[:] = 1 - (east_std_t / east_std_f)**2
        
        north_std_f = nc_f.variables['north_err'][0, :]
        north_std_t = nc_t.variables['north_err'][0, :]
        north_energy_ratio = numpy.ma.masked_all(east_std_f.shape, dtype=float)
        north_energy_ratio[:] = 1 - (north_std_t / north_std_f)**2

        # mask
        for Id in range(missing_indices_in_ocean_inside_hull.shape[0]):
            lat_index = missing_indices_in_ocean_inside_hull[Id, 0]
            lon_index = missing_indices_in_ocean_inside_hull[Id, 1]
            east_energy_ratio[lat_index, lon_index] = numpy.ma.masked
            north_energy_ratio[lat_index, lon_index] = numpy.ma.masked

        return east_energy_ratio, north_energy_ratio 

    # ---------------------------------------

    # Plotting additional entropies between two distributions
    filename_1 = '/home/sia/work/ImageProcessing/HFR-Uncertainty/files/MontereyBay_2km_Output_Full_KL_Expansion.nc'
    filename_2 = '/home/sia/work/ImageProcessing/HFR-Uncertainty/files/MontereyBay_2km_Output_Truncated_100Modes_KL_Expansion.nc'
    east_jsd, north_jsd = _js_distance_of_two_distributions(filename_1, filename_2)
    _plot_scalar_fields(east_jsd, north_jsd, 'Jensen-Shannon divergence between full and truncated KL expansion', cm.Reds, shift_colormap_status=False, log_norm=False)

    # Plot ratio of energy for truncation
    east_energy_ratio, north_energy_ratio = _ratio_of_truncation_energy(filename_1, filename_2)
    _plot_scalar_fields(east_energy_ratio, north_energy_ratio, 'Ratio of truncation error energy over total energy of KL expansion', cm.Reds, shift_colormap_status=False, log_norm=False)

    plt.show()
