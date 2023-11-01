.. _examples:

Examples
********

The following set of instructions is a step-by-step guide to reproduce the result in :ref:`[2] <ref2>`, which is an example of using |project|.

Dataset
=======

The paper uses the HF radar dataset from the Monterey Bay region in California, USA. The HF radar can be accessed publicly through the `national HF radar network gateway <http://cordc.ucsd.edu/projects/mapping/>`__ maintained by the Coastal Observing Research and Development Center. Please follow these steps to obtain the data file.

**Dataset Webpage:** The studied data is available through its `OpenDap page <https://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd.html>`__. In particular, the **OpenDap URL** of this dataset is

.. code-block:: bash

   http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd

**Navigate to Dataset Webpage:** To navigate to the above webpage, visit `CORDC THREDDS Serever <https://hfrnet-tds.ucsd.edu/thredds/catalog.html>`__. From there, click on `HF RADAR, US West Coast <https://hfrnet-tds.ucsd.edu/thredds/HFRADAR_USWC.html>`__, and select `HFRADAR US West Coast 2km Resolution Hourly RTV <https://hfrnet-tds.ucsd.edu/thredds/catalog/HFR/USWC/2km/hourly/RTV/catalog.html>`__ and choose `Best Time Series <https://hfrnet-tds.ucsd.edu/thredds/catalog/HFR/USWC/2km/hourly/RTV/catalog.html?dataset=HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd>`__.

**Subseting Data:** The above dataset contains the HF radar covering the west coast of the US from 2012 to the present date with 2km resolution and hourly updates. In the examples below, we focus on a spatial subset of this dataset within the Monterey Bay region for January 2017.

Reproducing Results
===================

The scripts to reproduce the results can be found in the source code of |project| under the directory |script_dir|_.

Reproducing Figure 1
--------------------

.. prompt:: bash

    python plot_gdop_coverage.py

.. figure:: _static/images/plots/gdop_coverage.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 1:** *Panels (a) and (b) display the east and north GDOP values, respectively, calculated along the northern California coast using the radar configurations indicated by the red dots and their corresponding code names. Panel (c) shows the total GDOP, which is a measure of the overall quality of the estimation of the velocity vector. Panel (d) represents the average HF radar data availability for January 2017, indicating the locations with missing data that generally correspond to high total GDOP values.*

Reproducing Figure 2 to Figure 9
--------------------------------

.. prompt:: bash

    python restoreio_mwe.py

The above python script executes the following lines of code, which is given in **Listing 1** of the manuscript. Users may change the settings in the function arguments below. For further details, refer to the API reference documentation of the function :func:`restoreio.restore` function.

.. code-block:: python

    >>> # Install restoreio with: pip install restoreio
    >>> from restoreio import restore

    >>> # OpenDap URL of the remote netCDF data
    >>> url = 'http://hfrnet-tds.ucsd.edu/thredds/' + \
    ...       'dodsC/HFR/USWC/2km/hourly/RTV/HFRAD' + \
    ...       'AR_US_West_Coast_2km_Resolution_Hou' + \
    ...       'rly_RTV_best.ncd'

    >>> # Generate ensemble and reconstruct gaps
    >>> restore(input=url, output='output.nc',
    ...         min_lon=-122.344, max_lon=-121.781,
    ...         min_lat=36.507, max_lat=36.992,
    ...         time='2017-01-25T03:00:00',
    ...         uncertainty_quant=True, plot=True,
    ...         num_samples=2000, ratio_num_modes=1,
    ...         kernel_width=5, scale_error=0.08,
    ...         detect_land=True, fill_coast=True,
    ...         write_samples=True, verbose=True)

The above script generates the output file ``output.nc`` that contains all generated ensemble. Moreover, it creates a subdirectory called ``output_results`` and stores **Figure 2** to **Figure 9** of the manuscript. These plots are shown below.

.. figure:: _static/images/plots/orig_vel_and_error.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 2:** *Panels (a) and (b) show the east and north components of the ocean's current velocity as measured by HF radars in Monterey Bay on January 25-th, 2017, at 3:00 UTC. The regions inside the solid black curves represent missing data that was filtered out due to high GDOP values from the original measurement. Panels (c) and (d) respectively show the east and north components of the velocity error computed for the locations where velocity data is available in Panels (a) and (b).*

.. figure:: _static/images/plots/rbf_kernel_2d.png
   :align: left
   :figwidth: 100%
   :width: 90%
   :class: custom-dark

   **Figure 3:** *The red fields represent the calculated spatial autocorrelation α for the east (a) and north (b) velocity data. The elliptical contour curves are the best fit of the exponential kernel* :math:`\rho` *to the autocorrelation. The direction of the principal radii of ellipses is determined by the eigenvectors of* :math:`\boldsymbol{M}`, *representing the principal direction of correlation. The radii values are proportional to the eigenvalues of* :math:`\boldsymbol{M}`, *representing the correlation length scale. The axes are in the unit of data points spaced 2 km apart.*

.. figure:: _static/images/plots/cor_cov.png
   :align: left
   :figwidth: 100%
   :width: 90%
   :class: custom-dark

   **Figure 4:** *Correlation (first column) and covariance matrices (second column) of the east (first row) and north (second row) datasets are shown. The size of matrices are* :math:`n = 485`.

.. figure:: _static/images/plots/kl_eigenvectors.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 5:** *The first 12 spatial eigenfunctions* :math:`\phi_i` *for the east velocity dataset (first and second rows) and north velocity dataset (third and fourth rows) are shown in the domain* :math:`\Omega` *in the Monterey Bay. The black curves is indicate the boundary of the missing domain* :math:`\Omega_{\circ}`. *We note that the oblique pattern in the east eigenfunctions is attributed to the anisotropy of the east velocity data, as illustrated in Figure 3a.*
   
.. figure:: _static/images/plots/ensemble.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 6:** *The reconstructed central ensemble (first column), mean of reconstructed ensemble (second column), and the standard deviation of reconstructed ensemble (third column) are shown in both* :math:`\Omega` *and* :math:`\Omega_{\circ}`. *The boundary of* :math:`\Omega_{\circ}` *is shown by the solid black curve. The first and second rows correspond to the east and north velocity data, respectively.*

.. figure:: _static/images/plots/deviation.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 7:** *The left to right columns show the plots of deviations* :math:`d_1(\boldsymbol{x})`, :math:`d_2(\boldsymbol{x})`, :math:`d_3(\boldsymbol{x})`, *and* :math:`d_4(\boldsymbol{x})`, *displayed in both domains* :math:`\Omega` *and* :math:`\Omega_{\circ}` *with the first and second rows representing the east and north datasets, respectively. The solid black curve shows the boundary of* :math:`\Omega_{\circ}`. *The absolute values smaller than* :math:`10^{-8}` *are rendered as transparent and expose the ocean background, which includes the domain* :math:`\Omega` *for the first three deviations.*

.. figure:: _static/images/plots/ensemble_js_distance.png
   :align: left
   :figwidth: 100%
   :width: 90%
   :class: custom-dark

   **Figure 8:** *The JS distance between the expected distribution* :math:`q(\boldsymbol{x}, \xi)` *and the observed distribution* :math:`p(\boldsymbol{x}, \xi)` *is shown. The absolute values smaller than* :math:`10^{-8}` *are rendered as transparent and expose the ocean background, which includes the domain* :math:`\Omega` *where the JS distance between* :math:`p(\boldsymbol{x}, \xi)` *and* :math:`q(\boldsymbol{x}, \xi)` *is zero.*

.. figure:: _static/images/plots/kl_eigenvalues.png
   :align: left
   :figwidth: 100%
   :width: 70%
   :class: custom-dark

   **Figure 9:**  *The eigenvalues* :math:`\lambda_i`, :math:`i = 1, \dots , n` *(green curves using left ordinate) and the energy ratio* :math:`\gamma_m`, :math:`m = 1, \dots , n` *(blue curves using right ordinate) are shown for the east and north velocity data. The horizontal dashed lines correspond to the 60% and 90% energy ratio levels, respectively, which equate to utilizing nearly 10 and 100 eigenmodes.*

Reproducing Figure 10
---------------------

* First, run ``plot_js_divergence.sh`` script:

  .. prompt:: bash
  
      bash plot_js_divergence.sh
  
  The above script creates a directory called ``output_js_divergence`` and stores the output files ``output-001.nc`` to ``output-200.nc``.

* Next, run ``plot_js_divergence.py`` script:
  
  .. prompt:: bash
  
      python plot_js_divergence.py
  
.. figure:: _static/images/plots/js_distance.png
 :align: left
 :figwidth: 100%
 :width: 70%
 :class: custom-dark
 
 **Figure 10:** *The JS distance between the probability distributions* :math:`p_m(\boldsymbol{x}, \xi)` *and* :math:`p_n(\boldsymbol{x}, \xi)` *is shown as a function of* :math:`m = 0, \dots , n`. *These two distributions correspond to the ensemble generated by the* :math:`m` *term (truncated) and* :math:`n` *term (complete) KL expansions, respectively. We note that the abscissa of the figure is displayed as the percentage of the ratio* :math:`m/n` *where* :math:`n = 485`.

Reproducing Figure 11
---------------------

Run ``plot_vel_distribution.py`` script:

.. prompt:: bash

    python plot_vel_distribution.py
  
The above script creates the following plot:
  
.. figure:: _static/images/plots/vel_distribution.png
 :align: left
 :figwidth: 100%
 :width: 100%
 :class: custom-dark
 
 **Figure 11:** *Probability density functions of the east (a) and north (b) components of velocity data for January 2017. Circle points denote the empirical PDF, the solid black curve indicates the standard normal distribution* :math:`\mathcal{N}(0, 1)`, *and the dashed black curve shows the best-fit generalized Gaussian distribution (GGD) with the density function* :math:`p(v) \propto \exp(-\vert v \vert^{\beta})`. *The abscissa in both figures represents the normalized velocity components, with* :math:`\sigma` *being the standard deviation of the respective data.*

.. |script_dir| replace:: ``/examples/uncertainty_quant``
.. _script_dir: https://github.com/ameli/restoreio/blob/main/examples/uncertainty_quant/
