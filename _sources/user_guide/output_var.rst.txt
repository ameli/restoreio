.. _output-var-sec:

Output Variables
================

.. contents::
   :depth: 2

The output is a NetCDF file in ``.nc`` format containing a selection of the following variables, contingent on the chosen configuration:

1. :ref:`Mask <output-mask>`
2. :ref:`Reconstructed East and North Velocities <output-vel-var>`
3. :ref:`East and North Velocity Errors <output-vel-err-var>`
4. :ref:`East and North Velocity Ensembles <output-vel-ens-var>`

.. _output-mask:

1. Mask
-------

The mask variable is a three-dimensional array with dimensions for *time*, *longitude*, and *latitude*. This variable is stored under the name ``mask`` in the output file.

Interpreting Variable over Segmented Domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mask variable includes information about the result of domain segmentation (refer to :ref:`Domain Segmentation <domain-seg-sec>` section). This array contains integer values ``-1``, ``0``, ``1``, and ``2`` that are interpreted as follows:

* The value ``-1`` indicates the location is identified to be on the **land** domain :math:`\Omega_l`. In these locations, the output velocity variable is masked.
* The value ``0`` indicates the location is identified to be on the **known** domain :math:`\Omega_k`. These locations have velocity data in the input file. The same velocity values are preserved in the output file.
* The value ``1`` indicates the location is identified to be on the **missing** domain :math:`\Omega_m`. These locations do not have a velocity data in the input file, but they do have a reconstructed velocity data on the output file.
* The value ``2`` indicates the location is identified to be on the **ocean** domain :math:`\Omega_0`. In these locations, the output velocity variable is masked.

.. _output-vel-var:

2. Reconstructed East and North Velocities
------------------------------------------

The reconstructed east and north velocity variables are stored in the output file under the names ``east_vel`` and ``north_vel``, respectively. These variables are three-dimensional arrays with dimensions for *time*, *longitude*, and *latitude*.

Interpreting Variable over Segmented Domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The velocity variables on each of the segmented domains are defined as follows:

* On locations where the :ref:`Mask <output-mask>` value is ``-1`` or ``2``, the output velocity variables are masked.
* On locations where the :ref:`Mask <output-mask>` value is ``0``, the output velocity variables have the same values as the corresponding variables in the input file.
* On locations where the :ref:`Mask <output-mask>` value is ``1``, the output velocity variables are reconstructed. If the ``uncertainty_quant`` option is enabled, these output velocity variables are obtained by the **mean** of the velocity ensembles, where the missing domain of each ensemble is reconstructed.

.. _output-vel-err-var:

3. East and North Velocity Errors
---------------------------------

If the ``uncertainty_quant`` option is enabled, the east and north velocity error variables will be included in the output file under the names ``east_err`` and ``north_err``, respectively. These variables are three-dimensional arrays with dimensions for *time*, *longitude*, and *latitude*.

Interpreting Variable over Segmented Domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The velocity error variables on each of the segmented domains are defined as follows:

* On locations where the :ref:`Mask <output-mask>` value is ``-1`` or ``2``, the output velocity error variables are masked.
* On locations where the :ref:`Mask <output-mask>` value is ``0``, the output velocity error variables are obtained from either the corresponding velocity error or GDOP variables in the input file scaled by the value of ``scale_error`` argument.
* On locations where the :ref:`Mask <output-mask>` value is ``1``, the output velocity error variables are obtained from the **standard deviation** of the ensembles, where the missing domain of each ensemble is reconstructed.

.. _output-vel-ens-var:

4. East and North Velocity Ensembles
------------------------------------

When you activate the ``uncertainty_quant`` option, a collection of velocity field ensembles is created. Yet, by default, the output file only contains the mean and standard deviation of these ensembles. To incorporate all ensembles into the output file, you should additionally enable the ``write_ensembles`` option. This action saves the east and north velocity ensemble variables in the output file as ``east_vel_ensembles`` and ``north_vel_ensembles``, respectively. These variables are four-dimensional arrays with dimensions for *ensemble*, *time*, *longitude*, and *latitude*. 

Ensemble Dimension
~~~~~~~~~~~~~~~~~~

The *ensemble* dimension of the array has the size :math:`s+1` where :math:`s` is the number of ensembles specified by ``num_ensembles`` (also refer to :ref:`Number of (Monte-Carlo) Samples <num-samples-sec>` section). The first ensemble with the index :math:`0` (assuming zero-based numbering) corresponds to the original input dataset. The other ensembles with the indices :math:`1, \dots, s` correspond to the generated ensembles.

Interpreting Variable over Segmented Domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The velocity ensemble variables on each of the segmented domains are defined similar to those presented for :ref:`Reconstructed East and North Velocities <output-vel-var>`. In particular, the missing domain of each ensemble is reconstructed independently.

Mean and Standard Deviation of Ensembles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that the *mean* and *standard deviation* of the velocity ensemble arrays over the ensemble dimension yield the :ref:`Reconstructed East and North Velocities <output-vel-var>` and :ref:`East and North Velocity Errors <output-vel-err-var>` variables, respectively.
