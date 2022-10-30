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

# =============================
# Convert Image To Valid Vector
# =============================

def ConvertImageToValidVector(Image, ValidIndices):
    """
    - Image:        (n, m) 2D array. n is the Latitude size and m is Longitude size.
                    Image is masked everywhere except where data is valid.
    - ValidIndices: (N_Valid, 2) array where first column is Latitudes indices and
                    second column is Longitude indices.
                    Here N_valid is number of valid points.
    - ValidVector:  (V_Valid, ) vector. 
    """
    ValidVector = Image[ValidIndices[:, 0], ValidIndices[:, 1]]
    return ValidVector

# =============================
# Convert Valid Vector To Image
# =============================

def ConvertValidVectorToImage(ValidVector, ValidIndices, ImageShape):
    """
    - ValidVector:  (N_Valid) is the vector of all valid data values that is vectorized.
    - ValidIndices: (N_Valid, 2) array where first column is Latitudes indices and
                    second column is Longitude indices.
                    Here N_valid is number of valid points.
    - Image:        (n, m) 2D masked array, n is Latitude size, and m is Longitude size.
                    Image is masked everywhere except where data is valid.
    """
    Image = numpy.ma.masked_all(ImageShape, dtype=float)
    for i in range(ValidIndices.shape[0]):
        Image[ValidIndices[i, 0], ValidIndices[i, 1]] = ValidVector[i]
    return Image

# ==============================
# Get Valid Indices For All Axes
# ==============================

def GetValidIndicesForAllAxes(Array):
    """
    Generates a map between Ids and (TimeIndex, LatitudeIndex, LongitudeIndex) for only valid points.
    This is for the 3D Array.
    """

    # Along Longitude
    ValidIndicesList_Lon = []
    Ids_Lon = numpy.ones(Array.shape, dtype=int)*-1

    Counter = 0
    for LatitudeIndex in range(Array.shape[0]):
        for LongitudeIndex in range(Array.shape[1]):
            if Array.mask[LatitudeIndex, LongitudeIndex] == False:
                ValidIndicesList_Lon.append((LatitudeIndex, LongitudeIndex))
                Ids_Lon[LatitudeIndex, LongitudeIndex] = Counter
                Counter += 1

    ValidIndices_Lon = numpy.array(ValidIndicesList_Lon)

    # Along Latitude
    ValidIndicesList_Lat = []
    Ids_Lat = numpy.ones(Array.shape, dtype=int)*-1

    Counter = 0
    for LongitudeIndex in range(Array.shape[1]):
        for LatitudeIndex in range(Array.shape[0]):
            if Array.mask[LatitudeIndex, LongitudeIndex] == False:
                ValidIndicesList_Lat.append((LatitudeIndex, LongitudeIndex))
                Ids_Lat[LatitudeIndex, LongitudeIndex] = Counter
                Counter += 1

    ValidIndices_Lat = numpy.array(ValidIndicesList_Lat)

    return ValidIndices_Lon, Ids_Lon, ValidIndices_Lat, Ids_Lat

# =======================================
# Compute AutoCorrelation Of Valid Vector
# =======================================

def ComputeAutoCorrelationOfValidVector(ValidVector):
    """
    ACF is computed from a vectorized data. The N-dimensional data is vectorized to be like a time series.
    Data is assumed periodic, so the indices are rotating around the size of the vector.
    The ACF is then a simple shift in the 1D data, like
    
    ACF(shift) = sum_{i=1}^N TimeSeries(i)*TimeSeries(i+shift)

    ACF is normalized, so that ACF[0] = 1, by ACF =/ ACF[0].
    """

    N_Valid = ValidVector.size
    TimeSeries = ValidVector - numpy.mean(ValidVector)

    # Limit the size of ACF shift
    ACF_Size = int(N_Valid / 100)
    if ACF_Size < 10: ACF_Size = 10
    if ACF_Size > 100: ACF_Size = 100

    ACF = numpy.zeros((ACF_Size, ), dtype=float)

    for i in range(ACF_Size):
        for j in range(N_Valid):
            ACF[i] += TimeSeries[j]*TimeSeries[(j+i) % N_Valid]

    # Normalize
    ACF /= ACF[0]

    return ACF

# ===================================
# Estimate Autocorrelation RBF Kernel
# ===================================

def EstimateAutocorrelationRBFKernel(MaskedImageData, ValidIndices, Ids, Window_Lon, Window_Lat):
    """
    ---------
    Abstract:
    ---------

    The Radial Basis Function (RBF) kernel is assumed in this function. We assume the 2D data has
    the following kernel:
        K(ii, jj) = exp(-0.5*sqrt(X.T * QuadraticForm * X))
    where X is a vector:
        X = (i-ii, j-jj)
    That is, at the center point (i, j), the X is the distance vector with the shift (ii, jj).
    This function estimates the QuadraticForm 2x2 symmetric matrix.

    -------
    Inputs:
    -------

        - MaskedImageData: MxN masked data of the east/north velocity field.
        - ValidIndices:    (N_Valid, 2) array. Each rwo is of the form [LatitudeIndex, LongitudeIndex].
                           If there are NxM points in the grid, not all of these points have valid velocity
                           data defined. Suppose there are only N_Valid points with valid velocities.
                           Each row of ValidIndices is the latitude and longitudes of these points.
        - Ids:             (N, M) array of integers. If a point is non-valid, the value is -1, if
                           a point is valid, the value on the array is the Id (row number in ValidIndices).
                           This array is used to show how we mapped ValidIndices to the grid Ids.
        - Window_Lon:      Scalar. This is the half window of stencil in Longitude direction for sweeping the
                           kernel convolution. The rectangular area of the kernel is of size
                           (2*Window_Lon_1, 2*Window_Lat+1).
        - Window_Lat:      Scalar. This is the half window of stencil in Latitude direction for sweeping the
                           kernel convolution. The rectangular area of the kernel is of size
                           (2*Window_Lon_1, 2*Window_Lat+1).

    --------
    Outputs:
    --------

        - QuadraticForm:   2x2 symmetric matrix.

    ---------------------------
    How we estimate the kernel:
    ---------------------------

    1. We subtract mean of data form data. Lets call it Data.

    2. For each valid point with Id1 (or Latitude and Longitude Indices), we look for all neighbor points
       within the central stencil of size Window_Lon, Window_Lat (This is a rectangle of size 2*Window_Lon+1, 2*Window_Lat+1),
       If any of these points are valid itself, say point Id2, we compute the correlation
            Correlation of Id1 and Id2 = Data[Id1] * Data[Id2].
       This value is stored in a 2D array of size of kernel, where the center of kernel corresponds to the point with Id1.
       We store this correlation value in the offset index of two points Id1 and Id2 in the Kernel array.

    2. Now we have N_Valid 2D kernels for each valid points. We average them all to get a 2D KernelAverage array. We normalize
    this array w.r.t to the center value of this array. So the center of the KernelAverage is 1.0.
    If we plot this array, it should be descending.

    3. We fit an exponential RBG function to the 2D kernel:
        z = exp(-0.5*sqrt(X.T * QuadraticForm * X))
       where X = (i-ii, j-jj) is a distance vector form the center of the kernel array.
       Suppose the kernel array is of size (P=2*Window_Lon+1, Q=2*Window_Lat_1)
       To do so we take Z = 4.0*(log(z))^2 = X.T * QuadraticForm * X.
       Suppose (ii, jj) is the center indices of the kernel array.
       For each point (i, j) in the kernel matrix, we compute

       A = [(i-ii)**2, 2*(i-ii)*(j-jj), (j-jj)**2]  this is array of size (P*Q, 3) for each i, j on the Kernel.
       b = [Z] this is a vector of size (P*Q, ) for each i, j in the kernel.

       Now we find the least square AX=b.
       The quadratic form is

        QuadraticForm = [ X[0], X[1] ]
                        [ X[1], X[2] ]

    -----
    Note:
    -----

        - Let Lambda_1 and Lambda_2 be the eigenvalues of QuadraticForm. Then L1 = 1/sqrt(Lambda_1) and L2=1/sqrt(Lambda_2) are
          the characteristic lengths of the RBF kernel. If the quadratic kernel is diagonal, this is essentially the ARM kernel.

        - The eigenvalues should be non-negative. But if they are negative, this is becase we chose a large Wiwndow_Lon or Window_Lat,
          hence the 2D Kernel function is not strictly descending everywhere. To fix this choose  smaller window sizes.
    """

    # Subtract mean of Masked Image Data
    Data = numpy.ma.copy(MaskedImageData) - numpy.ma.mean(MaskedImageData)

    # 3D array that the first index is for each valid point, and the 2 and 3 index creates a 2D matrix
    # that is the autocorrelation of that valid point (i, j) with the nearby point (i+ii, j+jj)
    KernelForAllValidPoints = numpy.ma.masked_all((ValidIndices.shape[0], 2*Window_Lat+1, 2*Window_Lon+1), dtype=float)

    # Itertate over valid points
    for Id1 in range(ValidIndices.shape[0]):
        LatitudeIndex_1, LongitudeIndex_1 = ValidIndices[Id1, :]

        # Sweep the kernel rectangular area to find nearby points to the center point.
        for LatitudeOffset in range(-Window_Lat, Window_Lat+1):
            for LongitudeOffset in range(-Window_Lon, Window_Lon+1):

                LatitudeIndex_2 = LatitudeIndex_1 + LatitudeOffset
                LongitudeIndex_2 = LongitudeIndex_1 + LongitudeOffset

                if (LatitudeIndex_2 >= 0) and (LatitudeIndex_2 < MaskedImageData.shape[0]):
                    if (LongitudeIndex_2 >= 0) and (LongitudeIndex_2 < MaskedImageData.shape[1]):
                        Id2 = Ids[LatitudeIndex_2, LongitudeIndex_2]

                        if Id2 >= 0:
                            # The nearby point is a valid point. Compute the correlation of points Id1 and Id2 and store in the Kernel
                            KernelForAllValidPoints[Id1, LatitudeOffset+Window_Lat, LongitudeOffset+Window_Lon] = Data[LatitudeIndex_2, LongitudeIndex_2] * Data[LatitudeIndex_1, LongitudeIndex_1]

    # Average all 2D kernels over the valid points (over the first index)
    KernelAverage = numpy.ma.mean(KernelForAllValidPoints, axis=0)

    # Normalize the kernel w.r.t the center of the kernel. So the center is 1.0 and all other correlations are less that 1.0.
    KernelAverage = KernelAverage / KernelAverage[Window_Lat, Window_Lon]

    # Get the gradient of the kernel to find up to where the kernel is descending. We only use kernel in areas there it is descending.
    GradientKernelAverage = numpy.gradient(KernelAverage)

    # Find where on the grid the data is descending (in order to avoid ascendings in ACF)
    Descending = numpy.zeros((2*Window_Lat+1, 2*Window_Lon+1), dtype=bool)
    for LatitudeOffset in range(-Window_Lat, Window_Lat+1):
        for LongitudeOffset in range(-Window_Lon, Window_Lon+1):
            Radial = numpy.array([LatitudeOffset, LongitudeOffset])
            Norm = numpy.linalg.norm(Radial)
            if Norm > 0:
                Radial = Radial / Norm
            elif Norm == 0:
                Descending[LatitudeOffset+Window_Lat, LongitudeOffset+Window_Lon] = True
            Grad = numpy.array([GradientKernelAverage[0][LatitudeOffset+Window_Lat, LongitudeOffset+Window_Lon], GradientKernelAverage[1][LatitudeOffset+Window_Lat, LongitudeOffset+Window_Lon]])

            if numpy.dot(Grad, Radial) < 0.0:
                Descending[LatitudeOffset+Window_Lat, LongitudeOffset+Window_Lon] = True

    # Construct a Least square matrices
    A_List = []
    b_List = []

    for LatitudeOffset in range(-Window_Lat, Window_Lat+1):
        for LongitudeOffset in range(-Window_Lon, Window_Lon+1):
            Value = KernelAverage[LatitudeOffset+Window_Lat, LongitudeOffset+Window_Lon]
            if Value <= 0.05:
                continue
            elif Descending[LatitudeOffset+Window_Lat, LongitudeOffset+Window_Lon] == False:
                continue

            a = numpy.zeros((3, ), dtype=float)
            a[0] = LongitudeOffset**2
            a[1] = 2.0*LongitudeOffset*LatitudeOffset
            a[2] = LatitudeOffset**2
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
    QuadraticForm = numpy.array([[X[0], X[1]], [X[1], X[2]]])

    # ------------------------
    # Plot RBF Kernel Function
    # ------------------------

    def PlotRBFKernelFunction():
        """
        Plots both the averaged kernel and its analytic exponental estimate.
        """

        # Print characteristic length scales
        E, V = numpy.linalg.eigh(QuadraticForm)
        print('RBF Kernel characteristic length scales: %f, %f'%(numpy.sqrt(1.0/E[0]), numpy.sqrt(1.0/E[1])))

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Change font family
        plt.rcParams["font.family"] = "serif"

        fig1, ax1 = plt.subplots()
        ax1.set_rasterization_zorder(0)

        # Plot kernel with analytic exponential function
        xc = numpy.linspace(-Window_Lon, Window_Lon, 1000)
        yc = numpy.linspace(-Window_Lat, Window_Lat, 1000)
        xxc, yyc = numpy.meshgrid(xc, yc)
        z = numpy.exp(-0.5*numpy.sqrt(X[0]*xxc**2+2.0*X[1]*xxc*yyc+X[2]*yyc**2))
        Levels1 = numpy.linspace(0.0, 1.0, 6)
        cs = ax1.contour(xxc, yyc, z, levels=Levels1, cmap=plt.cm.Greys, vmin=0.0, vmax=1.0)
        ax1.clabel(cs, inline=1, fontsize=10, color='black')
        # ax1.contour(xxc, yyc, z, cmap=plt.cm.Greys, vmin=0.0, vmax=1.0)

        # Plot kernel function with statistical correlations that we found
        x = numpy.arange(-Window_Lon, Window_Lon+1)
        y = numpy.arange(-Window_Lat, Window_Lat+1)
        xx, yy = numpy.meshgrid(x, y)
        Levels2 = numpy.linspace(0, 1, 200)
        p = ax1.contourf(xx, yy, KernelAverage, levels=Levels2, cmap=plt.cm.Reds, vmin=0.0, vmax=1.0, zorder=-1)
        cb = fig1.colorbar(p, ticks=[0, 0.5, 1])
        cb.set_clim(0, 1)
        ax1.set_xticks([-Window_Lon, 0, Window_Lon])
        ax1.set_yticks([-Window_Lat, 0, Window_Lat])

        ax1.set_xlabel('Longitude offset')
        ax1.set_ylabel('Latitude offset')
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

    # PlotRBFKernelFunction()

    return QuadraticForm

# =====================================
# Estimate AutoCorrelation Length Scale
# =====================================

def EstimateAutoCorrelationLengthScale(ACF):
    """
    Assuming a Markov-1 Stationary process (mena and std do not change over time),
    the autocorrelation function is acf = rho**(d), where d is spatial distance
    between two points.
    """

    # Find up to where the ACF is positive
    Window = 0
    while (ACF[Window] > 0.0) and (ACF[Window] > ACF[Window+1]):
        Window += 1
        if Window >= ACF.size-1:
            break
    
    if Window < 1:
        raise RuntimeError('Window of positive ACF is not enough to eatimate paramter.')

    x = numpy.arange(1, Window)
    y = numpy.log(ACF[1:Window])
    LengthScale = -numpy.mean(x/y)

    return LengthScale

# =======================================
# Auto Correlation ARD Exponential Kernel
# =======================================

def AutoCorrelationARDExponentialKernel(Id1, Id2, ValidIndices, ACF_LengthScale_Lon, ACF_LengthScale_Lat):
    """
    Finds the correlation between two points with Id1 and Id2.
    """

    Index_J1 = ValidIndices[Id1, 0]
    Index_J2 = ValidIndices[Id2, 0]
    Index_I1 = ValidIndices[Id1, 1]
    Index_I2 = ValidIndices[Id2, 1]

    # Autocorrelation
    X = (Index_I1 - Index_I2) / ACF_LengthScale_Lon
    Y = (Index_J1 - Index_J2) / ACF_LengthScale_Lat
    ACF_Id1_Id2 = numpy.exp(-numpy.sqrt(X**2+Y**2))

    return ACF_Id1_Id2

# ===========================
# Auto Correlation RBF Kernel
# ===========================

def AutoCorrelationRBFKernel(Id1, Id2, ValidIndices, QuadraticForm):
    """
    RBF kernel with quadratic form.
    """

    Index_J1 = ValidIndices[Id1, 0]
    Index_J2 = ValidIndices[Id2, 0]
    Index_I1 = ValidIndices[Id1, 1]
    Index_I2 = ValidIndices[Id2, 1]
    DistanceVector = numpy.array([Index_I2-Index_I1, Index_J2-Index_J1])
    Quadrature = numpy.dot(DistanceVector, numpy.dot(QuadraticForm, DistanceVector.T))
    ACF_Id1_Id2 = numpy.exp(-0.5*numpy.sqrt(Quadrature))

    return ACF_Id1_Id2

# ==========================
# Compute Correlation Matrix
# ==========================

def ComputeCorrelationMatrix(ValidIndices, ACF_LengthScale_Lon, ACF_LengthScale_Lat, QuadraticForm):
    """
    ValidIndices is array of size (N_Valid, 2)
    Covariance matrix Cor is of size (N_Valid, N_Valid).
    """

    N_Valid = ValidIndices.shape[0]
    Cor = numpy.zeros((N_Valid, N_Valid), dtype=float) 

    for i in range(N_Valid):
        for j in range(i, N_Valid):
            # Cor[i, j] = AutoCorrelationARDExponentialKernel(i, j, ValidIndices, ACF_LengthScale_Lon, ACF_LengthScale_Lat)
            Cor[i, j] = AutoCorrelationRBFKernel(i, j, ValidIndices, QuadraticForm)
            if i != j:
                Cor[j, i] = Cor[i, j]
    
    return Cor

# ===========================
# Generate Monte Carlo Desing
# ===========================

def GenerateMonteCarloDesign(NumModes, NumEnsembles):
    """
    Monte Carlo design. This is purely random design.

    Output:
    - RandomVectors: (NumModes, NumEnsembles). Each column is one sample of all variables. That is
                     each column is one ensemble.

    In MC, the samples of each variable is selected purely randomly. Also samples do not interact between
    each other variables.

    How we generate random variables: 
        We generate random variables on N(0, 1). But since they might not have perfect mean and std, we shift and scale them 
        to have exact mean=0 and std=1.

    Problem with this method:
        1. The convergence rate of such simulation is O(1/log(n)) where n is the number of ensembles.
        2. The distribution's skewness is not exactly zero. Also the kurtosis is way away from zero.

    A better option is latin hypercube design.
    """

    RandomVectors = numpy.empty((NumModes, NumEnsembles), dtype=float)
    for ModeId in range(NumModes):

        # Generate random samples with Gaussianb distribution with mean 0 and std 1.
        Sample = numpy.random.randn(NumEnsembles)

        # Generate random sample with excatly zero mean and std
        Sample = Sample - numpy.mean(Sample)
        Sample = Sample / numpy.std(Sample)
        RandomVectors[ModeId, :] = Sample

    return RandomVectors

# =====================================
# Generate Symmetric Monte Carlo Design
# =====================================

def GenerateSymmetricMonteCarloDesign(NumModes, NumEnsembles):
    """
    Symmetric version of the GenerateMonteCarloDesign() function.

    Note:
    About Shuffling the OtherHalfOfSample:
    For each Sample, we create a OtherHalfOfSample which is opposite of the Sample.
    Together they create a full Sample that its mean is exactly zero.
    However, the stack of all of these samples that create the RandomVectors matrix become
    ill-ranked since we are repeating columns. That is RandomVectors = [Samples, -Samples].
    There are two cases:

    1. If we want to generate a set of random vectors that any linear combination of them still have ZERO SKEWNESS,
       then all Sample rows should have this structure, that is RandomVectors = [Sample, -Sample] (withput shuffling.)
       However, this matrix is low ranked, and if we want to make the RandomVectors to have Identity covariance (that is
       to have no correlation, or E[Row_i, Row_j] = 0), then the number of columns should be more or equal to twice the number
       of rows. That is NumEnsembles >= 2*NumModes.

    2. If we want to generate RandomVectors that are exactly decorrelated, and if NumEnsebles < 2*NumModes,
       we need to shuffle the OtherHalfOfSample, hence the line for shuffling should be uncommented.
    """
    RandomVectors = numpy.empty((NumModes, NumEnsembles), dtype=float)

    HalfNumEnsembles = int(NumEnsembles / 2.0)

    # Create a continuous normal distribution object with zero mean and unit std.
    NormalDistribution = scipy.stats.norm(0.0, 1.0)

    # Generate independent distributions for each variable
    for ModeId in range(NumModes):

        # Generate uniform distributun between [0.0, 0.5)]
        DiscreteUniformDistribution = numpy.random.rand(HalfNumEnsembles) / 2.0

        # Convert the uniform distribution to normal distribution in [0.0, inf) with inverse CDF
        HalfOfSample = NormalDistribution.isf(DiscreteUniformDistribution)

        # Other Half Of Samples
        OtherHalfOfSample = -HalfOfSample

        # We need to shuffle, otherwise the RandomVector=[Samples, -Samples] will be a matrix with similar columns and reduces the rank.
        # This will not produce zero skewness samples after KL expansion. If you need zero skewness, you should comment this line.
        # numpy.random.shuffle(OtherHalfOfSample)

        Sample = numpy.r_[HalfOfSample, OtherHalfOfSample]

        # Add a neutral sample to make number of ensembles odd (if it is supposed to be odd number)
        if NumEnsembles % 2 != 0:
            Sample = numpy.r_[Sample, 0.0]

        Sample = Sample / numpy.std(Sample)
        RandomVectors[ModeId, :] = Sample[:]

    return RandomVectors

# ====================================
# Generate Mean Latin Hypercube Design
# ====================================

def GenerateMeanLatinHypercubeDesign(NumModes, NumEnsembles):
    """
    Latin Hypercube Design (LHS) works as follow: 
    1. For each variable, divide the interval [0, 1] to number of ensembles, and call each interval a strip.
       Now, we randomly choose a number in each strip. If we use Medean LHS, then each sample is choosen on the center
       point of each strip, so this is not really a random selection.
    2. Once for each variable we chose ensembles, we ARANGE them on a hypercube so that in each row/column of the hypercube
        only one vector of samples exists.
    3. The distribution is not on U([0, 1]), which is uniform distrubution. We use inverse commulation density function (iCDF)
       to map them in N(0, 1) with normal distrubution. Now mean = 0, and std=1.

    Output:
    - RandomVectors: (NumModes, NumEnsembles). Each column is one sample of all variables. That is
                     each column is one ensemble.

    Notes:
        - The Mean LHS is not really random. Each point is choosen on the center of each strip.
        - The MEAN LHS ensures that the mean is zero, and std=1. Since the distribution is symmetric in MEAN LHS, the 
          skewness is exactly zero.
    """

    # Make sure the number of ensembels is more than variables
    # if NumEnsembles < NumModes:
    #     print('Number of variables: %d'%NumModes)
    #     print('Number of Ensembles: %d'%NumEnsembles)
    #     raise ValueError('In Latin Hypercube sampling, it is better to have more number of ensembels than number of variables.')

    # Mean (center) Latin Hypercube. LHS_Uniform is of the size (NumEnsembles, NumModes)
    LHS_Uniform = pyDOE.lhs(NumModes, samples=NumEnsembles, criterion='center')

    # Convert uniform distribution to normal distribution
    LHS_Normal = scipy.stats.distributions.norm(loc=0.0, scale=1.0).ppf(LHS_Uniform)

    # Make output matrix to the form (NumModes, NumEnsembles)
    RandomVector = LHS_Normal.transpose()

    # Make sure mean and std are exactly zero and one
    # for ModeId in range(NumModes):
    #     RandomVector[ModeId, :] = RandomVector[ModeId, :] - numpy.mean(RandomVector[ModeId, :])
    #     RandomVector[ModeId, :] = RandomVector[ModeId, :] / numpy.std(RandomVector[ModeId, :])

    return RandomVector

# ==============================================
# Generate Symmetric Mean Latin Hypercube Design
# ==============================================

def GenerateSymmetricMeanLatinHypercubeDesign(NumModes, NumEnsembles):
    """
    Symmetric means it preserves Skewness during isetric rotations.
    """

    def GenerateSamplesOnPlane(NumEnsembles):
        """
        Number of ensembles is modified to be the closest square number.
        """

        NumEnsemblesSquareRoot = int(numpy.sqrt(NumEnsembles))
        NumEnsembles = NumEnsemblesSquareRoot**2

        Counter = 0
        SamplesOnPlane = numpy.empty((NumEnsembles, 2), dtype=int)
        for i in range(NumEnsemblesSquareRoot):
            for j in range(NumEnsemblesSquareRoot):
                SamplesOnPlane[Counter, 0] = NumEnsemblesSquareRoot*i + j
                SamplesOnPlane[Counter, 1] = NumEnsemblesSquareRoot*(j+1) - (i+1)
                Counter += 1

        SortingIndex = numpy.argsort(SamplesOnPlane[:, 0])
        SamplesOnPlane = SamplesOnPlane[SortingIndex, :]
        Permutation = SamplesOnPlane[SortingIndex, 1]

        return Permutation

    # ------------

    Permutation = GenerateSamplesOnPlane(NumEnsembles)
    NumEnsembles = Permutation.size

    SamplesOnHypercube = numpy.empty((NumEnsembles, NumModes), dtype=int)
    SamplesOnHypercube[:, 0] = numpy.arange(NumEnsembles)

    for ModeId in range(1, NumModes):
        SamplesOnHypercube[:, ModeId] = Permutation[SamplesOnHypercube[:, ModeId-1]]

    # Values
    SampleUniformValues = 1.0 / (NumEnsembles * 2.0) + numpy.linspace(0.0, 1.0, NumEnsembles, endpoint=False)
    SampleNormalValues = scipy.stats.distributions.norm(loc=0.0, scale=1.0).ppf(SampleUniformValues)

    # Values on Hypercube
    SampleNormalValuesOnHypercube = numpy.empty((NumEnsembles, NumModes), dtype=float)
    for ModeId in range(NumModes):
        SampleNormalValuesOnHypercube[:, ModeId] = SampleNormalValues[SamplesOnHypercube[:, ModeId]]

    return SampleNormalValuesOnHypercube.transpose()

# ===============================
# Generate Valid Vector Ensembles
# ===============================

def GenerateValidVectorEnsembles(ValidVector, ValidVectorError, ValidIndices, NumEnsembles, NumModes, ACF_LengthScale_Lon, ACF_LengthScale_Lat, QuadraticForm):
    """
    For a given vector, generates similar vectors with a given vector error.

    Input:
    - ValidVector: (N_Valid, ) size
    - ValidVectorError: (N_Valid, ) size
    - ValidIndices: (N_Valid, 2) size

    Output:
    - ValidVectorEnsembles: (N_Valid, NumEnsembles) size

    Each column of the ValidVectorEnsembles is an ensemble of ValidVector.
    Each row of ValidVectorEnsembles[i, :] has a normal distribution N(ValidVector[i], ValidVectorError[i])
    that is with mean=ValidVector[i] and std=ValidVectorError[i]

    """

    # Correlation
    Cor = ComputeCorrelationMatrix(ValidIndices, ACF_LengthScale_Lon, ACF_LengthScale_Lat, QuadraticForm)

    # Covariance
    Sigma = numpy.diag(ValidVectorError)
    Cov = numpy.dot(Sigma, numpy.dot(Cor, Sigma))

    # ----------------
    # Plot Cor and Cov
    # ----------------

    def PlotCorAndCov():
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

    # PlotCorAndCov()

    # KL Transform of Covariance
    Eigenvalues, Eigenvectors = numpy.linalg.eigh(Cov)
    SortingIndex = numpy.argsort(Eigenvalues)
    SortingIndex = SortingIndex[::-1]
    Eigenvalues = Eigenvalues[SortingIndex]
    Eigenvectors = Eigenvectors[:, SortingIndex]

    # Check if there is any negative eigenvalues
    NegativeIndices = numpy.where(Eigenvalues < 0.0)
    if NegativeIndices[0].size > 0:
        for i in range(NegativeIndices[0].size):
            if Eigenvalues[NegativeIndices[0][i]] > -1e-5:
                Eigenvalues[NegativeIndices[0][i]] = 0.0
            else:
                print('Negative eigenvalue: %f', Eigenvalues[NegativeIndices[0][i]])
                raise RuntimeError('Encountered negative eigenvalue in computing KL transform of positive definite covariance matrix.')

    # Number of modes for KL expansion
    if NumModes is None:
        # Default NumMondes
        # NumModes = ValidVector.size   # Full number of nodes
        NumModes = 100

    # Generate Gaussian random process for each point (not for each ensemble)
    # RandomVectors = GenerateMonteCarloDesign(NumModes, NumEnsembles)
    RandomVectors = GenerateSymmetricMonteCarloDesign(NumModes, NumEnsembles)
    # RandomVectors = GenerateMeanLatinHypercubeDesign(NumModes, NumEnsembles)
    # RandomVectors = GenerateSymmetricMeanLatinHypercubeDesign(NumModes, NumEnsembles)

    # Decorrelate random vectors (if they still have a correlation)
    # RandomVariables has atleast one dim=1 null space since the mean of vectors are zero. Hence
    # to have a full rank, the condition should be NumEnsembles > NumModes + 1, otherwise we  will have zero eigenvalues.
    if NumEnsembles > NumModes + 1:
        RandomVectors_Cor = numpy.dot(RandomVectors, RandomVectors.transpose()) / NumEnsembles
        RandomVectors_EigVal, RandomVectors_EigVect = numpy.linalg.eigh(RandomVectors_Cor)
        # RandomVectors = numpy.dot(numpy.diag(1.0/numpy.sqrt(RandomVectors_EigVal)), numpy.dot(RandomVectors_EigVect.transpose(), RandomVectors))                                         # PCA whitening transformation
        RandomVectors = numpy.dot(RandomVectors_EigVect, numpy.dot(numpy.diag(1.0/numpy.sqrt(RandomVectors_EigVal)), numpy.dot(RandomVectors_EigVect.transpose(), RandomVectors)))          # ZCA whitening transformation
    else:
        print('WARNING: cannot decorrelate RandomVariables when NumEnsembles is less than NumModes. NumModes: %d, NumEnsembles: %d'%(NumModes, NumEnsembles))

    # Generate each ensemble with correlations
    ValidVectorEnsembles = numpy.empty((ValidVector.size, NumEnsembles), dtype=float)
    for EnsembleId in range(NumEnsembles):

        # KL expansion
        ValidVectorEnsembles[:, EnsembleId] = ValidVector + numpy.dot(Eigenvectors[:, :NumModes], numpy.sqrt(Eigenvalues[:NumModes])*RandomVectors[:NumModes, EnsembleId])

    # Uncomment for plot
    # PlotValidVectorEnsemblesStatistics(ValidVector, ValidVectorError, RandomVectors, ValidVectorEnsembles)

    return ValidVectorEnsembles, Eigenvalues, Eigenvectors

# ========================
# Generate Image Ensembles
# ========================

def GenerateImageEnsembles(Longitude, Latitude, MaskedImageData, MaskedImageDataError, ValidIndices, NumEnsembles, NumModes):
    """
    Note: The Longitude and Latitude is NOT needed for the computatrion of this function. However, if we want to plot
          the eigenvectors on the map, we need the Longitudes and Latitudes.

    Input:
    - MaskedImageData: (n, m) Image array that are partially masked. This is the original velocity data (either u or v velocity)
    - MaskedImageDataError: (n, m) Image array that is partially masked. This is the error of original velocity data (either u or v velocity)
    - ValidIndices: (N_Valid, 2) 1D array. First column are latitudes (i indices) and second column are longitude (j indices) of valid data on
                    velocity arrays and their errors.
    - NumEnsembles: The number of output array of ensembles is actually NumEnsembles+1, since the first ensemble that we output is the original
                    data itself in order to have the central ensemble in the data.

    Note:
        The first ensemble is the central ensemble, which is the ensemble without perturbation of variables and
        corresponds to the mean of each variable.
    """

    # Convert Image data to vector
    ValidVector = MaskedImageData[ValidIndices[:, 0], ValidIndices[:, 1]]
    ValidVectorError = MaskedImageDataError[ValidIndices[:, 0], ValidIndices[:, 1]]

    # Compute Autocorrelation of data for each axis
    ValidIndices_Lon, Ids_Lon, ValidIndices_Lat, Ids_Lat = GetValidIndicesForAllAxes(MaskedImageData)
    ValidVector_Lon = MaskedImageData[ValidIndices_Lon[:, 0], ValidIndices_Lon[:, 1]]
    ValidVector_Lat = MaskedImageData[ValidIndices_Lat[:, 0], ValidIndices_Lat[:, 1]]
    ACF_Lon = ComputeAutoCorrelationOfValidVector(ValidVector_Lon)
    ACF_Lat = ComputeAutoCorrelationOfValidVector(ValidVector_Lat)
    ACF_LengthScale_Lon = EstimateAutoCorrelationLengthScale(ACF_Lon)
    ACF_LengthScale_Lat = EstimateAutoCorrelationLengthScale(ACF_Lat)

    print('LengthScales: Lon: %f, Lat: %f'%(ACF_LengthScale_Lon, ACF_LengthScale_Lat))

    # Plot ACF
    # PlotAutoCorrelation(ACF_Lon, ACF_Lat, ACF_LengthScale_Lon, ACF_LengthScale_Lat)

    # Window of kernel
    Window_Lon = 5
    Window_Lat = 5
    QuadraticForm = EstimateAutocorrelationRBFKernel(MaskedImageData, ValidIndices_Lon, Ids_Lon, Window_Lon, Window_Lat)

    # Generate ensembles for vector (Note: Eigenvalues and Eigenvectors are only needed for plotting them)
    ValidVectorEnsembles, Eigenvalues, Eigenvectors = GenerateValidVectorEnsembles(ValidVector, ValidVectorError, ValidIndices, NumEnsembles, NumModes, ACF_LengthScale_Lon, ACF_LengthScale_Lat, QuadraticForm)
    NumEnsembles = ValidVectorEnsembles.shape[1]

    # Convert back vector to image
    MaskedImageDataEnsembles = numpy.ma.masked_all((NumEnsembles+1, )+MaskedImageData.shape, dtype=float)

    # Add the original data to the first ensemble as the central ensemble
    MaskedImageDataEnsembles[0, :, :] = MaskedImageData

    # Add ensembles that are produced by KL expantion with perturbation of variables
    for EnsembleId in range(NumEnsembles):
        MaskedImageDataEnsembles[EnsembleId+1, :, :] = ConvertValidVectorToImage(ValidVectorEnsembles[:, EnsembleId], ValidIndices, MaskedImageData.shape)

    # Plot eigenvalues and eigenvectors (Uncommnet to plot)
    # PlotKLTransform(Longitude, Latitude, Eigenvalues, Eigenvectors, ACF_Lon, ACF_Lat, ValidIndices, MaskedImageData.shape)

    return MaskedImageDataEnsembles

# ========================
# Get Ensembles Statistics
# ========================

def GetEnsemblesStatistics( \
            LandIndices, \
            ValidIndices, \
            MissingIndicesInOceanInsideHull, \
            MissingIndicesInOceanOutsideHull, \
            Velocity_OneTime, \
            Error_Velocity_OneTime, \
            Velocity_AllEnsembles_Inpainted, \
            FillValue):
    """
    Gets the mean and std of all inpainted ensembles in regions where inpainted.

    Inputs:
        - Velocity_OneTime:
          The original velocity that is not inpainted, but ony one time-frame of it.
          This is used for its shape and mask, but not its data.
        - Velocity_AllEnsembes_Inpainted:
          This is the array that we need its data. Ensembles are itrated in the first index, i.e 
          Velocity_AllEnsembles_Inpainted[EnsembleId, LatitudeIndex, LongitudeIndex]
          The first index Velocity_AllEnsembles_Inpainted[0, :] is the central ensemble, 
          which is the actual inpainted velocity data in that specific timeframe without perturbation.
    """

    # Create a mask for the masked array
    Mask = numpy.zeros(Velocity_OneTime.shape, dtype=bool)

    # Mask missing points in ocean outside hull
    for i in range(MissingIndicesInOceanOutsideHull.shape[0]):
        Mask[MissingIndicesInOceanOutsideHull[i, 0], MissingIndicesInOceanOutsideHull[i, 1]] = True

    # Mask missing or even valid points on land
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):
            Mask[LandIndices[i, 0], LandIndices[i, 1]] = True

    # Mask points on land even if they have valid values
    if numpy.any(numpy.isnan(LandIndices)) == False:
        for i in range(LandIndices.shape[0]):

            # Velocities
            Velocity_OneTime[LandIndices[i, 0], LandIndices[i, 1]] = numpy.ma.masked

            # Velocities Errors
            Error_Velocity_OneTime[LandIndices[i, 0], LandIndices[i, 1]] = numpy.ma.masked

    # Initialize Outputs
    Velocity_OneTime_Inpainted_Stats = \
    {
        'CentralEnsemble': Velocity_AllEnsembles_Inpainted[0, :],
        'Mean': numpy.ma.masked_array(Velocity_OneTime, mask=Mask, fill_value=FillValue),
        'AbsoluteError': numpy.ma.masked_array(Error_Velocity_OneTime, mask=Mask, fill_value=FillValue),
        'STD': numpy.ma.masked_array(Error_Velocity_OneTime, mask=Mask, fill_value=FillValue),
        'RMSD': numpy.ma.masked_all(Error_Velocity_OneTime.shape, dtype=float),
        'NRMSD': numpy.ma.masked_all(Error_Velocity_OneTime.shape, dtype=float),
        'ExNMSD': numpy.ma.masked_all(Error_Velocity_OneTime.shape, dtype=float),
        'Skewness': numpy.ma.masked_all(Velocity_OneTime.shape, dtype=float),
        'ExKurtosis': numpy.ma.masked_all(Velocity_OneTime.shape, dtype=float),
        'Entropy': numpy.ma.masked_all(Velocity_OneTime.shape, dtype=float),
        'RelativeEntropy': numpy.ma.masked_all(Velocity_OneTime.shape, dtype=float)
    }

    # Fill outputs with statistics only at MissingIndicesInOceanInsideHull or AllMissingIndicesInOcean
    # Note: We exclude the first ensemble since it is the central ensemble and is not comming from the Gaussian distrubution.
    # Hence, this preseves the mean, std exactly as it was described for the random Gaussan distribution.
    # Indices = MissingIndicesInOceanInsideHull
    Indices = numpy.vstack((ValidIndices, MissingIndicesInOceanInsideHull))
    for Id in range(Indices.shape[0]):

        # Point Id to Point index
        i, j = Indices[Id, :]

        # All ensmebles of the point (i, j)
        DataEnsembles = Velocity_AllEnsembles_Inpainted[1:, i, j]

        # Central ensemble
        CentralData = Velocity_AllEnsembles_Inpainted[0, i, j]

        # Mean of Velocity
        Velocity_OneTime_Inpainted_Stats['Mean'][i, j] = numpy.mean(DataEnsembles)

        # Absolute Error
        Velocity_OneTime_Inpainted_Stats['AbsoluteError'][i, j] = Error_Velocity_OneTime[i, j]

        # STD of Velocity (Error)
        Velocity_OneTime_Inpainted_Stats['STD'][i, j] = numpy.std(DataEnsembles)

        # Root Mean Square Deviation
        Velocity_OneTime_Inpainted_Stats['RMSD'][i, j] = numpy.sqrt(numpy.mean((DataEnsembles[:] - CentralData)**2))

        # Normalized Root Mean Square Deviation
        # Velocity_OneTime_Inpainted_Stats['NRMSD'][i, j] = numpy.ma.abs(Velocity_OneTime_Inpainted_Stats['RMSD'][i, j] / (numpy.fabs(CentralData)+1e-10))
        Velocity_OneTime_Inpainted_Stats['NRMSD'][i, j] = numpy.ma.abs(Velocity_OneTime_Inpainted_Stats['RMSD'][i, j] / Velocity_OneTime_Inpainted_Stats['STD'][i, j])

        # Excess Normalized Mean Square Deviation (Ex NMSD)
        Velocity_OneTime_Inpainted_Stats['ExNMSD'][i, j] = numpy.mean(((DataEnsembles[:] - CentralData) / Velocity_OneTime_Inpainted_Stats['STD'][i, j])**2) - 1.0
        # Velocity_OneTime_Inpainted_Stats['ExNMSD'][i, j] = numpy.sqrt(numpy.mean(((DataEnsembles[:] - CentralData) / Velocity_OneTime_Inpainted_Stats['STD'][i, j])**2)) - 1.0

        # Skewness of Velocity (Error)
        # Velocity_OneTime_Inpainted_Stats['Skewness'][i, j] = scipy.stats.skew(DataEnsembles)
        # Velocity_OneTime_Inpainted_Stats['Skewness'][i, j] = numpy.mean(((DataEnsembles[:] - Velocity_OneTime_Inpainted_Stats['Mean'][i, j])/Velocity_OneTime_Inpainted_Stats['STD'][i, j])**3)
        Velocity_OneTime_Inpainted_Stats['Skewness'][i, j] = numpy.mean(((DataEnsembles[:] - CentralData)/Velocity_OneTime_Inpainted_Stats['STD'][i, j])**3)

        # Excess Kurtosis of Velocity (Error) according to Fisher definition (3.0 is subtracted)
        # Velocity_OneTime_Inpainted_Stats['ExKurtosis'][i, j] = scipy.stats.kurtosis(DataEnsembles, fisher=True)
        # Velocity_OneTime_Inpainted_Stats['ExKurtosis'][i, j] = numpy.mean(((DataEnsembles[:] - Velocity_OneTime_Inpainted_Stats['Mean'][i, j])/Velocity_OneTime_Inpainted_Stats['STD'][i, j])**4) - 3.0
        Velocity_OneTime_Inpainted_Stats['ExKurtosis'][i, j] = numpy.mean(((DataEnsembles[:] - CentralData)/Velocity_OneTime_Inpainted_Stats['STD'][i, j])**4) - 3.0

        # Entropy
        # NumBins = 21
        # Counts, Bins = numpy.histogram(DataEnsembles, bins=NumBins)
        # PDF = Counts / numpy.sum(Counts, dtype=float)
        # Velocity_OneTime_Inpainted_Stats['Entropy'][i, j] = scipy.stats.entropy(PDF)
        Velocity_OneTime_Inpainted_Stats['Entropy'][i, j] = numpy.log(numpy.std(DataEnsembles) * numpy.sqrt(2.0 * numpy.pi * numpy.exp(1)))  # Only for normal distribution

        # Relative Entropy
        # Normal = scipy.stats.norm(Velocity_OneTime[i, j], Error_Velocity_OneTime[i, j])
        # Normal = scipy.stats.norm(numpy.mean(DataEnsembles), numpy.std(DataEnsembles))
        # Normal = scipy.stats.norm(CentralData, numpy.std(DataEnsembles))
        # Discrete_Normal_PDF = numpy.diff(Normal.cdf(Bins))
        # Velocity_OneTime_Inpainted_Stats['RelativeEntropy'][i, j] = scipy.stats.entropy(PDF, Discrete_Normal_PDF)

        Velocity_OneTime_Inpainted_Stats['RelativeEntropy'][i, j] = 0.5 * ((CentralData - numpy.mean(DataEnsembles)) / numpy.std(DataEnsembles))**2  # Only for two normal dist with the same std

        # Mask zeros
        if numpy.fabs(Velocity_OneTime_Inpainted_Stats['RMSD'][i, j]) < 1e-8:
            Velocity_OneTime_Inpainted_Stats['RMSD'][i, j] = numpy.ma.masked
        if numpy.fabs(Velocity_OneTime_Inpainted_Stats['NRMSD'][i, j]) < 1e-8:
            Velocity_OneTime_Inpainted_Stats['NRMSD'][i, j] = numpy.ma.masked
        if numpy.fabs(Velocity_OneTime_Inpainted_Stats['ExNMSD'][i, j]) < 1e-8:
            Velocity_OneTime_Inpainted_Stats['ExNMSD'][i, j] = numpy.ma.masked
        if numpy.fabs(Velocity_OneTime_Inpainted_Stats['Skewness'][i, j]) < 1e-8:
            Velocity_OneTime_Inpainted_Stats['Skewness'][i, j] = numpy.ma.masked
        if numpy.fabs(Velocity_OneTime_Inpainted_Stats['RelativeEntropy'][i, j]) < 1e-8:
            Velocity_OneTime_Inpainted_Stats['RelativeEntropy'][i, j] = numpy.ma.masked

    return Velocity_OneTime_Inpainted_Stats

# ======================================
# Plot Valid Vector Ensembles Statistics
# ======================================

def PlotValidVectorEnsemblesStatistics(ValidVector, ValidVectorError, RandomVectors, ValidVectorEnsembles):
    """
    Compare the mean, std, skewness, kurtosis of ensembles with the generated random vectors.
    """
    import scipy.stats
    import matplotlib.pyplot as plt
    m1 = numpy.mean(ValidVectorEnsembles, axis=1) - ValidVector
    m2 = numpy.std(ValidVectorEnsembles, axis=1) - ValidVectorError
    r2 = numpy.std(RandomVectors, axis=1) - 1.0
    m3 = scipy.stats.skew(ValidVectorEnsembles, axis=1)
    r3 = scipy.stats.skew(RandomVectors, axis=1)
    m4 = scipy.stats.kurtosis(ValidVectorEnsembles, axis=1)
    r4 = scipy.stats.kurtosis(RandomVectors, axis=1)

    ax1 = plt.subplot(2, 2, 1)
    plt.plot(m1, color='black', label='Difference of Mean of ensembles with central ensemble')
    plt.title('Mean Difference')
    plt.xlabel('Point')
    plt.legend()

    ax2 = plt.subplot(2, 2, 2)
    plt.plot(m2, color='black', label='Diff std ensembles with actual error')
    plt.plot(r2, color='red', label='std of generatd random vectors')
    plt.title('Standard Deviation Difference')
    plt.xlabel('Point')
    plt.legend()

    ax3 = plt.subplot(2, 2, 3)
    plt.plot(m3, color='black', label='Skewness of ensembles')
    plt.plot(r3, color='red', label='Skewness of generated ranfom vectors')
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

def PlotAutoCorrelation(ACF_Lon, ACF_Lat, ACF_LengthScale_Lon, ACF_LengthScale_Lat):
    """
    Plots ACF
    """

    import matplotlib.pyplot as plt

    # Change font family
    plt.rcParams["font.family"] = "serif"

    PlotSize = 8
    x = numpy.arange(PlotSize)
    y1 = numpy.exp(-x/ACF_LengthScale_Lon)
    y2 = numpy.exp(-x/ACF_LengthScale_Lat)

    # Plot
    # plt.plot(ACF_Lon, '-o', color='blue', label='Eastward autocorrelation')
    # plt.plot(ACF_Lat, '-o', color='green', label='Northward autocorrelation')
    # plt.plot(x, y1, '--s', color='blue', label='Eastward exponential kernel fit')
    # plt.plot(x, y2, '--s', color='green', label='Northward exponential kernel it')
    # plt.plot(numpy.array([x[0], x[-1]]), numpy.array([numpy.exp(-1), numpy.exp(-1)]), '--')

    plt.semilogy(ACF_Lon, 'o', color='blue', label='Eastward autocorrelation')
    plt.semilogy(ACF_Lat, 'o', color='green', label='Northward autocorrelation')
    plt.semilogy(x, y1, '--', color='blue', label='Eastward exponential kernel fit')
    plt.semilogy(x, y2, '--', color='green', label='Northward exponential kernel fit')
    plt.semilogy(numpy.array([x[0], x[PlotSize-1]]), numpy.array([numpy.exp(-1), numpy.exp(-1)]), '--')
    plt.semilogy(numpy.array([x[0], x[PlotSize-1]]), numpy.array([numpy.exp(-3), numpy.exp(-3)]), '--')

    plt.title('Autocorrelation function')
    plt.xlabel('Shift')
    plt.ylabel('ACF')
    plt.xlim(0, PlotSize-1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

# =================
# Plot KL Transform
# =================

def PlotKLTransform(Longitude, Latitude, Eigenvalues, Eigenvectors, ACF_Lon, ACF_Lat, ValidIndices, ImageShape):
    """
    Plots Eigenvalues and eigenvectors of the KL transform.
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
    # Plot Eigenvalues
    # ----------------

    def PlotEigenvalues():
        """
        Plots log log scale of eigenvalues
        """

        # Eigenvalues
        fig, ax1 = plt.subplots()
        # ax1.loglog(Eigenvalues, color='green', label='Eigenvalues')
        ax1.semilogy(Eigenvalues, color='green', label='Eigenvalues')
        ax1.set_xlabel('Modes number')
        ax1.set_ylabel('Eigenvalues', color='green')
        ax1.grid(True)
        ax1.set_xlim([1, Eigenvalues.size])
        ax1.set_ylim([1e-5, 2e-1])
        ax1.tick_params('y', colors='green')

        # Commulative eigenvalues
        EigenvaluesCumSum = numpy.cumsum(Eigenvalues)
        ax2 = ax1.twinx()
        # ax2.semilogx(EigenvaluesCumSum/EigenvaluesCumSum[-1], color='blue', label='Normalized Cumulative sum')
        ax2.plot(EigenvaluesCumSum/EigenvaluesCumSum[-1], color='blue', label='Normalized Cumulative sum')
        ax2.set_ylabel('Normalized Cummulative Sum', color='blue')
        ax2.set_xlim([1, Eigenvalues.size])
        ax2.tick_params('y', colors='blue')
        h1, l1 = ax1.get_legend_handles_labels() # legend for both ax and its twin ax
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='lower center')

        plt.title('Decay of eigenvalues for KL transform')

    # -----------------
    # Plot Eigenvectors
    # -----------------

    def PlotEigenvectors():
        """
        Plot eigenvectors on the map.
        """
 
        # Mesh grid
        LongitudesGrid, LatitudesGrid = numpy.meshgrid(Longitude, Latitude)

        # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
        # Percent = 0.05   # For Monterey Dataset
        Percent = 0.0
        # Percent = 0.1     # For Martha Dataset
        LongitudeOffset = Percent * numpy.abs(Longitude[-1] - Longitude[0])
        LatitudeOffset = Percent * numpy.abs(Latitude[-1] - Latitude[0])

        MinLongitude = numpy.min(Longitude)
        MinLongitudeWithOffset = MinLongitude - LongitudeOffset
        MidLongitude = numpy.mean(Longitude)
        MaxLongitude = numpy.max(Longitude)
        MaxLongitudeWithOffset = MaxLongitude + LongitudeOffset
        MinLatitude = numpy.min(Latitude)
        MinLatitudeWithOffset = MinLatitude - LatitudeOffset
        MidLatitude = numpy.mean(Latitude)
        MaxLatitude = numpy.max(Latitude)
        MaxLatitudeWithOffset = MaxLatitude + LatitudeOffset

        # --------
        # Draw map
        # --------

        def DrawMap(axis):

            # Basemap (set resolution to 'i' for faster rasterization and 'f' for full resolution but very slow.)
            map = Basemap( \
                    ax = axis, \
                    projection = 'aeqd', \
                    llcrnrlon=MinLongitudeWithOffset, \
                    llcrnrlat=MinLatitudeWithOffset, \
                    urcrnrlon=MaxLongitudeWithOffset, \
                    urcrnrlat=MaxLatitudeWithOffset, \
                    area_thresh = 0.1, \
                    lon_0 = MidLongitude, \
                    lat_0 = MidLatitude, \
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

            # Latitude and Longitude lines
            # LongitudeLines = numpy.linspace(numpy.min(Longitude), numpy.max(Longitude), 2)
            # LatitudeLines = numpy.linspace(numpy.min(Latitude), numpy.max(Latitude), 2)
            # map.drawparallels(LatitudeLines, labels=[1, 0, 0, 0], fontsize=10)
            # map.drawmeridians(LongitudeLines, labels=[0, 0, 0, 1], fontsize=10)

            return map

        # -----------------
        # Plot on Each Axis
        # -----------------

        def PlotOnEachAxis(Axis, ScalarField, Title):
            """
            This plots in each of left or right axes.
            """

            Axis.set_aspect('equal')
            Axis.set_rasterization_zorder(0)
            Axis.set_facecolor('#C7DCEF')
            Map = DrawMap(Axis)

            # Meshgrids
            LongitudesGridOnMap, LatitudesGridOnMap= Map(LongitudesGrid, LatitudesGrid)

            # Pcolormesh
            ContourLevels = 200
            Draw = Map.pcolormesh(LongitudesGridOnMap, LatitudesGridOnMap, ScalarField, cmap=cm.jet, rasterized=True, zorder=-1)
            # Draw = Map.pcolormesh(LongitudesGridOnMap, LatitudesGridOnMap, ScalarField, cmap=cm.jet)
            # Draw = Map.contourf(LongitudesGridOnMap, LatitudesGridOnMap, ScalarField, ContourLevels, cmap=cm.jet, rasterized=True, zorder=-1, corner_mask=False)

            # Create axes for colorbar that is the same size as the plot axes
            divider = make_axes_locatable(Axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Colorbar
            cb = plt.colorbar(Draw, cax=cax, ticks=numpy.array([numpy.ma.min(ScalarField), numpy.ma.max(ScalarField)]))
            cb.solids.set_rasterized(True)
            cb.ax.tick_params(labelsize=7)

            Axis.set_title(Title, fontdict={'fontsize':7})

        # --------------

        NumRows = 4
        NumColumns = 4
        fig, axes = plt.subplots(nrows=NumRows, ncols=NumColumns, figsize=(15, 10))
        fig.suptitle('Mercer Eigenvectors')

        Counter = 0
        for Axis in axes.flat:
            print('Plotting eigenvector %d'%Counter)
            EigenvectorImage = ConvertValidVectorToImage(Eigenvectors[:, Counter], ValidIndices, ImageShape)
            PlotOnEachAxis(Axis, EigenvectorImage, 'Mode %d'%(Counter+1))
            Counter += 1

    # -------------

    PlotEigenvalues()
    # PlotEigenvectors()

    plt.show()

# =========================
# Plot Ensembles Statistics
# =========================

def PlotEnsemblesStatistics(
        Longitude, \
        Latitude, \
        ValidIndices, \
        MissingIndicesInOceanInsideHull, \
        U_OneTime, \
        V_OneTime, \
        Error_U_OneTime, \
        Error_V_OneTime, \
        U_AllEnsembles_Inpainted, \
        V_AllEnsembles_Inpainted, \
        U_AllEnsembles_Inpainted_Stats, \
        V_AllEnsembles_Inpainted_Stats):
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

    def PlotConvergence():

        # Note: The first ensemble is not generated by random process, since it is the central ensemble. So we exclude it.
        NumEnsembles = U_AllEnsembles_Inpainted.shape[0] - 1
        Means_U = numpy.zeros((NumEnsembles, MissingIndicesInOceanInsideHull.shape[0]), dtype=float)
        Means_V = numpy.zeros((NumEnsembles, MissingIndicesInOceanInsideHull.shape[0]), dtype=float)

        # Mean of ensembles from the second ensemble to the i-th where i varies to the end of array.
        # We do not take to account of the first ensemble since it is the central ensemble and was not generated by Gaussian random process.
        # Also means are obtained from only the points that were inpainted, not from the valid points.
        for Ensemble in range(NumEnsembles):
            for PointId in range(MissingIndicesInOceanInsideHull.shape[0]):
                IndexI, IndexJ = MissingIndicesInOceanInsideHull[PointId, :]
                Means_U[Ensemble, PointId]=numpy.ma.mean(U_AllEnsembles_Inpainted[1:Ensemble+2, IndexI, IndexJ])
                Means_V[Ensemble, PointId]=numpy.ma.mean(V_AllEnsembles_Inpainted[1:Ensemble+2, IndexI, IndexJ])

        # Difference of each consecutive mean
        Diff_Means_U = numpy.max(numpy.abs(numpy.diff(Means_U, axis=0)), axis=1)
        Diff_Means_V = numpy.max(numpy.abs(numpy.diff(Means_V, axis=0)), axis=1)

        x = numpy.arange(NumEnsembles)
        log_x = numpy.log(x[1:])

        log_Diff_Means_U = numpy.log(Diff_Means_U)
        log_Diff_Means_U_Fit = numpy.polyval(numpy.polyfit(log_x, log_Diff_Means_U, 1), log_x)
        Diff_Means_U_Fit = numpy.exp(log_Diff_Means_U_Fit)

        log_Diff_Means_V = numpy.log(Diff_Means_V)
        log_Diff_Means_V_Fit = numpy.polyval(numpy.polyfit(log_x, log_Diff_Means_V, 1), log_x)
        Diff_Means_V_Fit = numpy.exp(log_Diff_Means_V_Fit)

        # Plot
        fig, ax = plt.subplots()
        ax.loglog(Diff_Means_U, color='lightsteelblue', zorder=0, label='East velocity data')
        ax.loglog(Diff_Means_U_Fit, color='darkblue', zorder=1, label='Line fit')
        ax.loglog(Diff_Means_V, color='palegreen', zorder=0, label='North velocity data')
        ax.loglog(Diff_Means_V_Fit, color='green', zorder=1, label='Line fit')
        ax.set_xlim([x[1], x[-1]])
        ax.grid(True)
        ax.legend(loc='lower left')
        ax.set_xlabel('Ensumbles')
        ax.set_ylabel('Max Mean difference')
        plt.title('Convergence of mean of ensembles population')

    # ------------------
    # Plot convergence 2
    # ------------------

    def PlotConvergence2():

        Indices = MissingIndicesInOceanInsideHull

        # Note: The first ensemble is not generated by random process, since it is the central ensemble. So we exclude it.
        NumEnsembles = U_AllEnsembles_Inpainted.shape[0] - 1

        Means_U = numpy.ma.masked_all((NumEnsembles, Indices.shape[0]), dtype=float)
        Means_V = numpy.ma.masked_all((NumEnsembles, Indices.shape[0]), dtype=float)

        U_Data = U_AllEnsembles_Inpainted[1:, :, :]
        V_Data = V_AllEnsembles_Inpainted[1:, :, :]

        # Mean of ensembles from the second ensemble to the i-th where i varies to the end of array.
        # We do not take to account of the first ensemble since it is the central ensemble and was not generated by Gaussian random process.
        # Also means are obtained from only the points that were inpainted, not from the valid points.
        for Ensemble in range(NumEnsembles):
            print('Ensemble: %d'%Ensemble)
            for PointId in range(Indices.shape[0]):
                IndexI, IndexJ = Indices[PointId, :]
                Means_U[Ensemble, PointId]=numpy.ma.mean(U_Data[:Ensemble+1, IndexI, IndexJ])
                Means_V[Ensemble, PointId]=numpy.ma.mean(V_Data[:Ensemble+1, IndexI, IndexJ])

        # MeanDiff_Mean_U = numpy.mean(numpy.abs(Means_U - Means_U[-1, :]), axis=1)
        # MeanDiff_Mean_V = numpy.mean(numpy.abs(Means_V - Means_V[-1, :]), axis=1)

        U_Errors = numpy.empty((Indices.shape[0]), dtype=float)
        V_Errors = numpy.empty((Indices.shape[0]), dtype=float)
        for PointId in range(Indices.shape[0]):
            IndexI, IndexJ = Indices[PointId, :]
            U_Errors[PointId] = U_AllEnsembles_Inpainted_Stats['STD'][0, IndexI, IndexJ]
            V_Errors[PointId] = V_AllEnsembles_Inpainted_Stats['STD'][0, IndexI, IndexJ]
        MeanDiff_Mean_U = numpy.mean(numpy.abs(Means_U - Means_U[-1, :])/U_Errors, axis=1)
        MeanDiff_Mean_V = numpy.mean(numpy.abs(Means_V - Means_V[-1, :])/V_Errors, axis=1)

        StdDiff_Mean_U = numpy.std(numpy.abs(Means_U - Means_U[-1, :])/U_Errors, axis=1)
        StdDiff_Mean_V = numpy.std(numpy.abs(Means_V - Means_V[-1, :])/V_Errors, axis=1)


        # Z score for 84% condidence interval
        Z_Score = 1.0

        # Plot
        fig, ax = plt.subplots()

        yu = MeanDiff_Mean_U
        yv = MeanDiff_Mean_V

        yu_up = MeanDiff_Mean_U + Z_Score * StdDiff_Mean_U
        yu_dn = MeanDiff_Mean_U - Z_Score * StdDiff_Mean_U
        yu_dn[yu_dn <= 0.0] = 1e-4
        
        yv_up = MeanDiff_Mean_V + Z_Score * StdDiff_Mean_V
        yv_dn = MeanDiff_Mean_V - Z_Score * StdDiff_Mean_V
        yv_dn[yv_dn <= 0.0] = 1e-4

        N = yu.size
        x = numpy.arange(1, N+1)

        # -------------
        # Fit Power Law
        # -------------

        def FitPowerLaw(x, y):
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
        ax.plot(x, FitPowerLaw(x, yu), '--', color='blue', label='East velocity data (fit)')
        ax.fill_between(x, yu_dn, yu_up, color='lightskyblue', label=('84\% (std) bound'))

        ax.plot(x, yv, color='green', label=r'North velocity data ($\psi=V$)')
        ax.plot(x, FitPowerLaw(x, yv), '--', color='green', label='North velocity data (fit)')
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

    # PlotConvergence()
    # PlotConvergence2()

    # -----------------------
    # Plot Spatial Statistics
    # -----------------------
    
    # Mesh grid
    LongitudesGrid, LatitudesGrid = numpy.meshgrid(Longitude, Latitude)

    # # All Missing points coordinates
    # AllMissingLongitudes = LongitudesGrid[AllMissingIndices[:, 0], AllMissingIndices[:, 1]]
    # AllMissingLatitudes = LatitudesGrid[AllMissingIndices[:, 0], AllMissingIndices[:, 1]]
    # AllMissingPointsCoordinates = numpy.vstack((AllMissingLongitudes, AllMissingLatitudes)).T

    # # Missing points coordinates inside hull
    # MissingLongitudesInsideHull = LongitudesGrid[MissingIndicesInsideHull[:, 0], MissingIndicesInsideHull[:, 1]]
    # MissingLatitudesInsideHull = LatitudesGrid[MissingIndicesInsideHull[:, 0], MissingIndicesInsideHull[:, 1]]
    # MissingPointsCoordinatesInsideHull = numpy.vstack((MissingLongitudesInsideHull, MissingLatitudesInsideHull)).T

    # # Missing points coordinates outside hull
    # MissingLongitudesOutsideHull = LongitudesGrid[MissingIndicesOutsideHull[:, 0], MissingIndicesOutsideHull[:, 1]]
    # MissingLatitudesOutsideHull = LatitudesGrid[MissingIndicesOutsideHull[:, 0], MissingIndicesOutsideHull[:, 1]]
    # MissingPointsCoordinatesOutsideHull = numpy.vstack((MissingLongitudesOutsideHull, MissingLatitudesOutsideHull)).T

    # # Valid points coordinates
    # ValidLongitudes = LongitudesGrid[ValidIndices[:, 0], ValidIndices[:, 1]]
    # ValidLatitudes = LatitudesGrid[ValidIndices[:, 0], ValidIndices[:, 1]]
    # ValidPointsCoordinates = numpy.c_[ValidLongitudes, ValidLatitudes]

    # # Land Point Coordinates
    # if numpy.any(numpy.isnan(LandIndices)) == False:
    #     LandLongitudes = LongitudesGrid[LandIndices[:, 0], LandIndices[:, 1]]
    #     LandLatitudes = LatitudesGrid[LandIndices[:, 0], LandIndices[:, 1]]
    #     LandPointCoordinates = numpy.c_[LandLongitudes, LandLatitudes]
    # else:
    #     LandPointCoordinates = numpy.nan

    # Corner points (Use 0.05 for MontereyBay and 0.1 for Martha dataset)
    Percent = 0.05   # For Monterey Dtaaset
    # Percent = 0.1     # For Martha Dataet
    LongitudeOffset = Percent * numpy.abs(Longitude[-1] - Longitude[0])
    LatitudeOffset = Percent * numpy.abs(Latitude[-1] - Latitude[0])

    MinLongitude = numpy.min(Longitude)
    MinLongitudeWithOffset = MinLongitude - LongitudeOffset
    MidLongitude = numpy.mean(Longitude)
    MaxLongitude = numpy.max(Longitude)
    MaxLongitudeWithOffset = MaxLongitude + LongitudeOffset
    MinLatitude = numpy.min(Latitude)
    MinLatitudeWithOffset = MinLatitude - LatitudeOffset
    MidLatitude = numpy.mean(Latitude)
    MaxLatitude = numpy.max(Latitude)
    MaxLatitudeWithOffset = MaxLatitude + LatitudeOffset

    # --------
    # Draw map
    # --------

    def DrawMap(axis):

        # Basemap (set resolution to 'i' for faster rasterization and 'f' for full resolution but very slow.)
        map = Basemap( \
                ax = axis, \
                projection = 'aeqd', \
                llcrnrlon=MinLongitudeWithOffset, \
                llcrnrlat=MinLatitudeWithOffset, \
                urcrnrlon=MaxLongitudeWithOffset, \
                urcrnrlat=MaxLatitudeWithOffset, \
                area_thresh = 0.1, \
                lon_0 = MidLongitude, \
                lat_0 = MidLatitude, \
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

        # Latitude and Longitude lines
        LongitudeLines = numpy.linspace(numpy.min(Longitude), numpy.max(Longitude), 2)
        LatitudeLines = numpy.linspace(numpy.min(Latitude), numpy.max(Latitude), 2)
        map.drawparallels(LatitudeLines, labels=[1, 0, 0, 0], fontsize=10)
        map.drawmeridians(LongitudeLines, labels=[0, 0, 0, 1], fontsize=10)

        return map

    # ----------------
    # Shifted Colormap
    # ----------------

    def ShiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
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

    def PlotScalarFields(ScalarField1, ScalarField2, Title, ColorMap, ShiftColorMapStatus, LogNorm):
        """
        This creats a figure with two axes.
        The quantity ScalarField1 is related to the east velocity and will be plotted on the left axis.
        The quantity ScalarField2 is related to the north velocity and will be plotted on the right axis.

        If ShiftColorMap is True, we set the zero to the center of the colormap range. This is usefull if we
        use divergent colormaps like cm.bwr.
        """

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.5, 4))
        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')
        map_1 = DrawMap(axes[0])
        map_2 = DrawMap(axes[1])

        # Resterization. Anything with zorder less than 0 will be rasterized.
        axes[0].set_rasterization_zorder(0)
        axes[1].set_rasterization_zorder(0)

        # Draw Mapscale
        # Index = int(Latitude.size / 4)
        # x0, y0 = map_11(Longitude[0], Latitude[0])
        # x1, y1 = map_11(Longitude[Index], Latitude[0])
        # Distance = (x1 - x0) / 1000 # Length of scale in Km
        # Distance = 40 # For Monterey Dataset
        Distance = 5 # For Martha Dataset
        map_1.drawmapscale(MinLongitude + (MaxLongitude - MinLongitude) * 0.88, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')
        map_2.drawmapscale(MinLongitude + (MaxLongitude - MinLongitude) * 0.88, MinLatitude, MidLongitude, MidLatitude, Distance, barstyle='simple', units='km', labelstyle='simple', fontsize= '7')

        # Meshgrids
        LongitudesGridOnMap, LatitudesGridOnMap= map_1(LongitudesGrid, LatitudesGrid)

        # Default arguments for colormap
        if ColorMap is None:
            ColorMap = cm.jet
        if ShiftColorMapStatus is None:
            ShiftColorMapStatus = False

        # -----------------
        # Plot on Each Axis
        # -----------------

        def PlotOnEachAxis(Axis, Map, ScalarField, Title, ColorMap, LogNorm):
            """
            This plots in each of left or right axes.
            """

            # Shift colormap
            if ShiftColorMapStatus == True:
                Min = numpy.ma.min(ScalarField)
                Max = numpy.ma.max(ScalarField)
                if (Min < 0) and (Max > 0):
                    MidPoint = -Min/(Max-Min)
                    ColorMap = ShiftedColorMap(ColorMap, start=0.0, midpoint=MidPoint, stop=1.0)

            # Pcolormesh
            if LogNorm == True:
                # Plot in log scale
                Min = numpy.max([numpy.min(ScalarField), 1e-6])
                Draw = Map.pcolormesh(LongitudesGridOnMap, LatitudesGridOnMap, ScalarField, cmap=ColorMap, rasterized=True, zorder=-1, norm=matplotlib.colors.LogNorm(vmin=Min))
                # Draw = Map.contourf(LongitudesGridOnMap, LatitudesGridOnMap, ScalarField, 200, cmap=ColorMap, corner_mask=False, rasterized=True, zorder=-1)
            else:
                # Do not plot in log scale
                Draw = Map.pcolormesh(LongitudesGridOnMap, LatitudesGridOnMap, ScalarField, cmap=ColorMap, rasterized=True, zorder=-1)

            # Create axes for colorbar that is the same size as the plot axes
            divider = make_axes_locatable(Axis)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            cax = divider.append_axes("right", size="5%", pad=0.07)

            # Colorbar
            cb = plt.colorbar(Draw, cax=cax)
            cb.solids.set_rasterized(True)
            cb.set_label('m/s')

            # Axis labels
            Axis.set_title(Title)
            Axis.set_xlabel('Longitude (degrees)')
            Axis.set_ylabel('Latitude (degrees)')

            # Background blue for ocean
            Axis.set_facecolor('#C7DCEF')

        # ---------------

        fig.suptitle(Title)
        PlotOnEachAxis(axes[0], map_1, ScalarField1, 'East velocity Data', ColorMap, LogNorm)
        PlotOnEachAxis(axes[1], map_2, ScalarField2, 'North velocity Data', ColorMap, LogNorm)

    # --------------

    # Original (Uninpainted) Data
    ## PlotScalarFields(U_OneTime, V_OneTime, 'Velocities', cm.jet, ShiftColorMapStatus=False, LogNorm=False)
    ## PlotScalarFields(Error_U_OneTime, Error_V_OneTime, 'Velocity Errors', cm.Reds, ShiftColorMapStatus=False, LogNorm=False)

    # Central Ensemble
    CentralEnsemble_EastVel = U_AllEnsembles_Inpainted_Stats['CentralEnsemble'][0, :]
    CentralEnsemble_NorthVel = V_AllEnsembles_Inpainted_Stats['CentralEnsemble'][0, :]
    ## PlotScalarFields(CentralEnsemble_EastVel, CentralEnsemble_NorthVel, 'Central ensemble', cm.jet, ShiftColorMapStatus=False, LogNorm=False)

    # Mean Difference
    MeanDifference_EastVel = numpy.ma.abs((U_AllEnsembles_Inpainted_Stats['Mean'][0, :]-U_AllEnsembles_Inpainted[0, :])/U_AllEnsembles_Inpainted_Stats['STD'][0, :])
    MeanDifference_NorthVel = numpy.ma.abs((V_AllEnsembles_Inpainted_Stats['Mean'][0, :]-V_AllEnsembles_Inpainted[0, :])/V_AllEnsembles_Inpainted_Stats['STD'][0, :])
    MeanDifference_EastVel[numpy.ma.where(MeanDifference_EastVel < 1e-8)] = numpy.ma.masked
    MeanDifference_NorthVel[numpy.ma.where(MeanDifference_NorthVel < 1e-8)] = numpy.ma.masked
    # MeanDifference_EastVel = numpy.ma.abs((U_AllEnsembles_Inpainted_Stats['Mean'][0, :]-U_AllEnsembles_Inpainted_Stats['CentralEnsemble'][0, :])/U_AllEnsembles_Inpainted_Stats['CentralEnsemble'][0, :])
    # MeanDifference_NorthVel = numpy.ma.abs((V_AllEnsembles_Inpainted_Stats['Mean'][0, :]-V_AllEnsembles_Inpainted_Stats['CentralEnsemble'][0, :])/V_AllEnsembles_Inpainted_Stats['CentralEnsemble'][0, :])
    # MeanDifference_EastVel = numpy.ma.abs((U_AllEnsembles_Inpainted_Stats['Mean'][0, :]-U_AllEnsembles_Inpainted_Stats['CentralEnsemble'][0, :]))
    # MeanDifference_NorthVel = numpy.ma.abs((V_AllEnsembles_Inpainted_Stats['Mean'][0, :]-V_AllEnsembles_Inpainted_Stats['CentralEnsemble'][0, :]))
    ##PlotScalarFields(MeanDifference_EastVel, MeanDifference_NorthVel, 'Normalized first moment deviation w.r.t central ensemble', cm.Reds, ShiftColorMapStatus=False, LogNorm=True)

    # Mean
    STD_EastVel = U_AllEnsembles_Inpainted_Stats['Mean'][0, :]
    STD_NorthVel = V_AllEnsembles_Inpainted_Stats['Mean'][0, :]
    ##PlotScalarFields(STD_EastVel, STD_NorthVel, 'Mean of ensembles', cm.jet, ShiftColorMapStatus=False, LogNorm=False)

    # STD
    # STD_EastVel = numpy.log(U_AllEnsembles_Inpainted_Stats['STD'][0, :])
    # STD_NorthVel = numpy.log(V_AllEnsembles_Inpainted_Stats['STD'][0, :])
    STD_EastVel = U_AllEnsembles_Inpainted_Stats['STD'][0, :]
    STD_NorthVel = V_AllEnsembles_Inpainted_Stats['STD'][0, :]
    ##PlotScalarFields(STD_EastVel, STD_NorthVel, 'Standard deviation of ensembles', cm.Reds, ShiftColorMapStatus=False, LogNorm=True)

    # RMSD
    # RMSD_EastVel = numpy.log(U_AllEnsembles_Inpainted_Stats['RMSD'][0, :])
    # RMSD_NorthVel = numpy.log(V_AllEnsembles_Inpainted_Stats['RMSD'][0, :])
    RMSD_EastVel = U_AllEnsembles_Inpainted_Stats['RMSD'][0, :]
    RMSD_NorthVel = V_AllEnsembles_Inpainted_Stats['RMSD'][0, :]
    ## PlotScalarFields(RMSD_EastVel, RMSD_NorthVel, 'RMSD of ensembles w.r.t central ensemble', cm.Reds, ShiftColorMapStatus=False, LogNorm=True)

    # NRMSD
    # NRMSD_EastVel = numpy.log(U_AllEnsembles_Inpainted_Stats['NRMSD'][0, :])
    # NRMSD_NorthVel = numpy.log(V_AllEnsembles_Inpainted_Stats['NRMSD'][0, :])
    NRMSD_EastVel = U_AllEnsembles_Inpainted_Stats['NRMSD'][0, :]
    NRMSD_NorthVel = V_AllEnsembles_Inpainted_Stats['NRMSD'][0, :]
    ## PlotScalarFields(NRMSD_EastVel, NRMSD_NorthVel, 'NRMSD of ensembles w.r.t central ensemble', cm.Reds, ShiftColorMapStatus=False, LogNorm=True)

    # Excess Normalized MSD
    # ExNMSD_EastVel = numpy.log(U_AllEnsembles_Inpainted_Stats['ExNMSD'][0, :])
    # ExNMSD_NorthVel = numpy.log(V_AllEnsembles_Inpainted_Stats['ExNMSD'][0, :])
    ExNMSD_EastVel = U_AllEnsembles_Inpainted_Stats['ExNMSD'][0, :]
    ExNMSD_NorthVel = V_AllEnsembles_Inpainted_Stats['ExNMSD'][0, :]
    ##PlotScalarFields(ExNMSD_EastVel, ExNMSD_NorthVel, 'Excess Normalized second moment deviation w.r.t central ensemble', cm.Reds, ShiftColorMapStatus=False, LogNorm=True)

    # Skewness
    Skewness_EastVel = U_AllEnsembles_Inpainted_Stats['Skewness'][0, :]
    Skewness_NorthVel = V_AllEnsembles_Inpainted_Stats['Skewness'][0, :]
    # Trim = 1.2
    # Skewness_EastVel[numpy.ma.where(Skewness_EastVel > Trim)] = Trim
    # Skewness_EastVel[numpy.ma.where(Skewness_EastVel < -Trim)] = -Trim
    # Skewness_NorthVel[numpy.ma.where(Skewness_NorthVel > Trim)] = Trim
    # Skewness_NorthVel[numpy.ma.where(Skewness_NorthVel < -Trim)] = -Trim
    ##PlotScalarFields(Skewness_EastVel, Skewness_NorthVel, 'Normalizsed third moment deviation (Skewness) w.r.t central ensemble', cm.bwr, ShiftColorMapStatus=True, LogNorm=False)

    # Excess Kurtosis
    ExKurtosis_EastVel = U_AllEnsembles_Inpainted_Stats['ExKurtosis'][0, :]
    ExKurtosis_NorthVel = V_AllEnsembles_Inpainted_Stats['ExKurtosis'][0, :]
    ##PlotScalarFields(ExKurtosis_EastVel, ExKurtosis_NorthVel, 'Normalzied fourth moment deviation (Excess Kurtosis) w.r.t central ensemble', cm.bwr, ShiftColorMapStatus=True, LogNorm=False)

    # Entropy
    Entropy_EastVel = U_AllEnsembles_Inpainted_Stats['Entropy'][0, :]
    Entropy_NorthVel = V_AllEnsembles_Inpainted_Stats['Entropy'][0, :]
    ## PlotScalarFields(Entropy_EastVel, Entropy_NorthVel, 'Entropy of ensembles', cm.RdBu_r, ShiftColorMapStatus=True, LogNorm=False)
    # PlotScalarFields(Entropy_EastVel, Entropy_NorthVel, 'Entropy of ensembles', cm.coolwarm, True)
    # PlotScalarFields(Entropy_EastVel, Entropy_NorthVel, 'Entropy of ensembles', cm.RdYlGn_r, True)

    # Relative Entropy (KL Divergence) with respect to the normal distribution
    RelativeEntropy_EastVel = U_AllEnsembles_Inpainted_Stats['RelativeEntropy'][0, :]
    RelativeEntropy_NorthVel = V_AllEnsembles_Inpainted_Stats['RelativeEntropy'][0, :]
    # Trim = 0.1
    # RelativeEntropy_EastVel[numpy.ma.where(RelativeEntropy_EastVel > Trim)] = Trim
    # RelativeEntropy_NorthVel[numpy.ma.where(RelativeEntropy_NorthVel > Trim)] = Trim
    ## PlotScalarFields(RelativeEntropy_EastVel, RelativeEntropy_NorthVel, 'Kullback-Leibler Divergence of ensembles', cm.Reds, ShiftColorMapStatus=True, LogNorm=False)
    # PlotScalarFields(RelativeEntropy_EastVel, RelativeEntropy_NorthVel, 'Kullback-Leibler Divergence of ensembles', cm.YlOrRd, True)

    # -------------------------------
    # JS Distance Of Two Distribution
    # -------------------------------

    def JSMetricDistanceOfTwoDistributions(Filename1, Filename2):
        """
        Reads two files, and computes the JS metric distance of their east/north velocities.
        The JS metric distance is the square root of JS distance. Log base 2 is used,
        hence the output is in range [0, 1].
        """

        def KLDistance(Mean1, Mean2, Sigma1, Sigma2):
            """
            KL distance of two normal distributions.
            """
            KLD = numpy.log(Sigma2/Sigma1) + 0.5 * (Sigma1/Sigma2)**2 + ((Mean1-Mean2)**2)/(2.0*Sigma2**2) - 0.5
            return KLD

        def SymmetricKLDistance(Mean1, Mean2, Sigma1, Sigma2):
            SKLD = 0.5 * (KLDistance(Mean1, Mean2, Sigma1, Sigma2) + KLDistance(Mean2, Mean1, Sigma2, Sigma1))
            return SKLD

        def JSDistance(Field_Mean1, Field_Mean2, Field_Sigma1, Field_Sigma2):
            Field_JS_Metric = numpy.ma.masked_all(Field_Mean1.shape, dtype=float)
            for i in range(Field_Mean1.shape[0]):
                for j in range(Field_Mean1.shape[1]):

                    if Field_Mean1.mask[i, j] == False:
                        Mean1 = Field_Mean1[i, j]
                        Mean2 = Field_Mean2[i, j]
                        Sigma1 = numpy.abs(Field_Sigma1[i, j])
                        Sigma2 = numpy.abs(Field_Sigma2[i, j])

                        x = numpy.linspace(numpy.min([Mean1-6*Sigma1, Mean2-6*Sigma2]), numpy.max([Mean1+6*Sigma1, Mean2+6*Sigma2]), 10000)
                        Norm1 = scipy.stats.norm.pdf(x, loc=Mean1, scale=Sigma1)
                        Norm2 = scipy.stats.norm.pdf(x, loc=Mean2, scale=Sigma2)
                        Norm12 = 0.5*(Norm1+Norm2)
                        JSD = 0.5 * (scipy.stats.entropy(Norm1, Norm12, base=2) + scipy.stats.entropy(Norm2, Norm12, base=2))
                        if JSD < 0.0:
                            if JSD > -1e-8:
                                JSD = 0.0
                            else:
                                print('WARNING: Negative JS distance: %f'%JSD)
                        Field_JS_Metric[i, j] = numpy.sqrt(JSD)
            return Field_JS_Metric

        import netCDF4
        nc_f = netCDF4.Dataset(Filename1)
        nc_t = netCDF4.Dataset(Filename2)

        East_mean_f = nc_f.variables['East_vel'][0, :]
        East_mean_t = nc_t.variables['East_vel'][0, :]
        East_sigma_f = nc_f.variables['East_err'][0, :]
        East_sigma_t = nc_t.variables['East_err'][0, :]
        East_JSD = JSDistance(East_mean_t, East_mean_f, East_sigma_t, East_sigma_f)

        North_mean_f = nc_f.variables['North_vel'][0, :]
        North_mean_t = nc_t.variables['North_vel'][0, :]
        North_sigma_f = nc_f.variables['North_err'][0, :]
        North_sigma_t = nc_t.variables['North_err'][0, :]
        North_JSD = JSDistance(North_mean_t, North_mean_f, North_sigma_t, North_sigma_f)

        return East_JSD, North_JSD

    # --------------------------
    # Ratio of Truncation Energy
    # --------------------------

    def RatioOfTruncationEnergy(Filename1, Filename2):
        """
        Ratio of StandardDeviation^2 for truncated and full KL expansion.
        """

        import netCDF4
        nc_f = netCDF4.Dataset(Filename1)
        nc_t = netCDF4.Dataset(Filename2)

        East_std_f = nc_f.variables['East_err'][0, :]
        East_std_t = nc_t.variables['East_err'][0, :]
        East_EnergyRatio = numpy.ma.masked_all(East_std_f.shape, dtype=float)
        East_EnergyRatio[:] = 1 - (East_std_t / East_std_f)**2
        
        North_std_f = nc_f.variables['North_err'][0, :]
        North_std_t = nc_t.variables['North_err'][0, :]
        North_EnergyRatio = numpy.ma.masked_all(East_std_f.shape, dtype=float)
        North_EnergyRatio[:] = 1 - (North_std_t / North_std_f)**2

        # Mask
        for Id in range(MissingIndicesInOceanInsideHull.shape[0]):
            LatitudeIndex = MissingIndicesInOceanInsideHull[Id, 0]
            LongitudeIndex = MissingIndicesInOceanInsideHull[Id, 1]
            East_EnergyRatio[LatitudeIndex, LongitudeIndex] = numpy.ma.masked
            North_EnergyRatio[LatitudeIndex, LongitudeIndex] = numpy.ma.masked

        return East_EnergyRatio, North_EnergyRatio 

    # ---------------------------------------

    # Plotting additional entropies between two distributions
    Filename1 = '/home/sia/work/ImageProcessing/HFR-Uncertainty/files/MontereyBay_2km_Output_Full_KL_Expansion.nc'
    Filename2 = '/home/sia/work/ImageProcessing/HFR-Uncertainty/files/MontereyBay_2km_Output_Truncated_100Modes_KL_Expansion.nc'
    East_JSD, North_JSD = JSMetricDistanceOfTwoDistributions(Filename1, Filename2)
    PlotScalarFields(East_JSD, North_JSD, 'Jensen-Shannon divergence between full and truncated KL expansion', cm.Reds, ShiftColorMapStatus=False, LogNorm=False)

    # Plot Eario of energy for truncation
    East_EnergyRatio, North_EnergyRatio = RatioOfTruncationEnergy(Filename1, Filename2)
    PlotScalarFields(East_EnergyRatio, North_EnergyRatio, 'Ratio of truncation error energy over total energy of KL expansion', cm.Reds, ShiftColorMapStatus=False, LogNorm=False)

    plt.show()
