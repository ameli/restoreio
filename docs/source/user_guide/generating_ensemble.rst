.. _generating-ensemble:

Generating Ensemble
===================

Beyond its data reconstruction capabilities, |project| also provides the feature to create ensemble of the velocity vector field. These ensemble is crucial for quantifying uncertainties, which holds significance for various applications. For a more in-depth understanding of the ensemble generation algorithm, we direct interested readers to :ref:`[2] <ref2>`.

To create velocity ensemble, simply activate the ``uncertainty_quant`` option within :func:`restoreio.restore`. Do note that ensemble can be generated for **a single time point** only. This section elaborates on the utilization of :func:`restoreio.restore` specifically for ensemble generation purposes.

.. contents::
   :depth: 2

.. _ensemble-var-sec:

Required Variables
------------------

To generate ensemble, you should provide one of the following additional variables in your input file:

* :ref:`Ocean's Surface East and North Velocity Error Variables <ocean-vel-err-var-sec>`
* :ref:`Geometric Dilution of Precision Variables <ocean-gdop-var-sec>`

If you choose to provide GDOP variables instead of the velocity error variables, the velocity errors are calculated from GDOP as follows:

.. math::

    \begin{align}
    \sigma_e &= \sigma_r \mathrm{GDOP}_e, \\
    \sigma_n &= \sigma_r \mathrm{GDOP}_n,
    \end{align}

where :math:`\sigma_e` and :math:`\sigma_n` are the east and north components of the velocity error, :math:`\mathrm{GDOP_e}` and :math:`\mathrm{GDOP}_n` are the east and north components of the GDOP, respectively, and :math:`\sigma_r` is the radar's radial error. You can specify :math:`\sigma_r` using the ``scale_error`` argument within the function (also refer to :ref:`Scale Velocity Errors <scale-vel-error-sec>` section below).

.. _ensemble-settings-sec:

Ensemble Generation Settings
----------------------------

The following settings for ensemble generation can be set within the :func:`restoreio.restore` function:

.. _write-ensemble:

Write Ensemble to Output
~~~~~~~~~~~~~~~~~~~~~~~~

The ``write_samples`` option allows you to save the entire population of ensemble vector fields to the output file. If this option is not enabled, only the *mean* and *standard deviation* of the ensemble will be stored. For more details, please refer to the :ref:`Output Variables <output-var>` section.

.. _num-samples-sec:

Number of (Monte-Carlo) Samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``num_samples`` argument of the function enables you to specify the number of samples to be generated. This value should be greater than the number of velocity data points. Keep in mind that the processing time increases **linearly** with larger sample sizes.

.. _num-eigenmodes-sec:

Number of Eigen-Modes
~~~~~~~~~~~~~~~~~~~~~

To generate ensemble, the eigenvalues and eigenvectors of the covariance matrix of the velocity data need to be computed. For a velocity data with :math:`n` data points, this means the eigenvalues and eigenvectors of an :math:`n \times n` matrix have to be calculated. However, such a computation has a complexity of :math:`\mathcal{O}(n^3)`, which can be infeasible for large datasets.

To handle this, we employ a practical approach where we only compute a reduced number of :math:`m` eigenvalues and eigenvectors of the covariance matrix, where :math:`m` can be much smaller than :math:`n`. This simplification reduces the complexity to :math:`\mathcal{O}(n m^2)`, which enables us to process larger datasets while maintaining a reasonable level of accuracy. For a better understanding of this concept, we refer the interested reader to Section 4 of :ref:`[2] <ref2>`.

The ``ratio_num_modes`` argument of the function allows you to specify the number of eigenvectors of the data covariance to be utilized in the computations. The number of modes should be given as a percentage of the ratio :math:`m/n`.

Keep in mind that the processing time **quadratically** increases with the number of eigenmodes. We recommend setting this value to around 5% to 10% for most datasets.

.. _kernel-width-sec:

Kernel Width
~~~~~~~~~~~~

The ``kernel_width`` argument of the function represents the width of a spatial kernel used to construct the covariance matrix of the velocity data. The kernel width is measured in the unit of the velocity data points. For example, a kernel width of 5 on an HF radar dataset with a 2 km spatial resolution implies a kernel width of 10 km.

It is assumed that spatial distances larger than the kernel width are uncorrelated. Therefore, reducing the kernel width makes the covariance matrix of the data more sparse, resulting in more efficient processing. However, a smaller kernel width may lead to information loss within the dataset. As a general recommendation, we suggest setting this value to 5 to 20 data points.

.. _scale-vel-error-sec:

Scale Velocity Errors
~~~~~~~~~~~~~~~~~~~~~

The ``scale_error`` argument serves two purposes:

* If the :ref:`Ocean's Surface East and North Velocity Error Variables <ocean-vel-err-var-sec>` are included in the input dataset, the provided scale value is multiplied by the velocity error. This is useful to match the unit of the velocity error to the unit of the velocity data if they are not in the same unit. If you have velocity errors in the same unit as the velocity data, it is recommended to set this quantity to 1.
* If the :ref:`Geometric Dilution of Precision (GDOP) Variables <ocean-gdop-var-sec>` are included in the input dataset, the given scale value is interpreted as the HF radar's radial error, :math:`\sigma_r`. In this case, the velocity error is calculated by multiplying the radar's radial error by the GDOP variables. The typical range for the radial errors of HF radars is between 0.05 to 0.20 meters per second.
