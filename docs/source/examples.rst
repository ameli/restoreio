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

    >>> # Generate ensembles and reconstruct gaps
    >>> restore(input=url, output='output.nc',
    ...         min_lon=-122.344, max_lon=-121.781,
    ...         min_lat=36.507, max_lat=36.992,
    ...         time='2017-01-25T03:00:00',
    ...         uncertainty_quant=True, plot=True,
    ...         num_ensembles=2000, ratio_num_modes=1,
    ...         kernel_width=5, scale_error=0.08,
    ...         detect_land=True, fill_coast=True,
    ...         write_ensembles=True, verbose=True)

The above script generates the output file ``output.nc`` that contains all generated ensembles. Moreover, it creates a subdirectory called ``output_results`` and stores **Figure 2** to **Figure 9** of the manuscript. These plots are shown below.

.. figure:: _static/images/plots/orig_vel_and_error.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 2:** *Panels (a) and (b) show the east and north components of the ocean’s current velocity in the upper 0.3 m th to 2.5 m range, as measured by HF radars in Monterey Bay on January 25 , 2017, at 3:00 UTC. The data has been averaged hourly and mapped to a 2 km resolution Cartesian grid using unweighted least squares. The regions inside the solid black curves represent missing data that was filtered out due to high GDOP values from the original measurement. Panels (c) and (d) respectively show the east and north components of the velocity error computed for the locations where velocity data is available in Panels (a) and (b).*

.. figure:: _static/images/plots/rbf_kernel_2d.png
   :align: left
   :figwidth: 100%
   :width: 90%
   :class: custom-dark

   **Figure 3:** *The red fields represent the calculated spatial autocorrelation α for the east (a) and north (b) velocity data. The elliptical contour curves are the best fit of the exponential kernel ρ to the autocorrelation. The direction of the principal radii of ellipses is determined by the eigenvectors of M, representing the principal direction of correlation. The radii values are proportional to the eigenvalues of M, representing the correlation length scale. The axes are in the unit of data points spaced 2 km apart.*

.. figure:: _static/images/plots/cor_cov.png
   :align: left
   :figwidth: 100%
   :width: 90%
   :class: custom-dark

   **Figure 4:** *Correlation (first column) and covariance matrices (second column) of the east (first row) and north (second row) datasets are shown. The size of matrices are n = 485.*

.. figure:: _static/images/plots/kl_eigenvectors.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 5:** *The first 12 spatial eigenfunctions φi for the east velocity dataset (first and second rows) and north velocity dataset (third and fourth rows) are shown in the domain Ω in the Monterey Bay. The black curves is indicate the boundary of the missing domain Ω◦. We note that the oblique pattern in the east eigenfunctions is attributed to the anisotropy of the east velocity data, as illustrated in Figure 3a.*
   
.. figure:: _static/images/plots/ensembles.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 6:** *The reconstructed central ensemble (first column), mean of reconstructed ensembles (second column), and the standard deviation of reconstructed ensembles (third column) are shown in both Ω and Ω◦. The boundary of Ω◦ is shown by the solid black curve. The first and second rows correspond to the east and north velocity data, respectively.*

.. figure:: _static/images/plots/deviation.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 7:** *The left to right columns show the plots of deviations d1(x), d2(x), d3(x), and d4(x), displayed in both domains Ω and Ω◦ with the first and second rows representing the east and north datasets, respectively. The solid black curve shows the boundary of Ω◦. The absolute values smaller than 10−8 are rendered as transparent and expose the ocean background, which includes the domain Ω for the first three deviations.*

.. figure:: _static/images/plots/ensembles_js_distance.png
   :align: left
   :figwidth: 100%
   :width: 90%
   :class: custom-dark

   **Figure 8:** *The JS distance between the expected distribution q(x, ξ) and the observed distribution p(x, ξ) is shown. The absolute values smaller than 10−8 are rendered as transparent and expose the ocean background, which includes the domain Ω where the JS distance between p(x, ξ) and q(x, ξ) is zero.*

.. figure:: _static/images/plots/kl_eigenvalues.png
   :align: left
   :figwidth: 100%
   :width: 70%
   :class: custom-dark

   **Figure 9:**  *The eigenvalues λi, i = 1, . . . , n (green curves using left ordinate) and the energy ratio γm, m = 1, . . . , n (blue curves using right ordinate) are shown for the east and north velocity data. The horizontal dashed lines correspond to the 60% and 90% energy ratio levels, respectively, which equate to utilizing nearly 10 and 100 eigenmodes.*

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
 
 **Figure 10:** *The JS distance between the probability distributions pm(x, ξ) and pn(x, ξ) is shown as a function of m = 0, . . . , n. These two distributions correspond to the ensembles generated by the m-term (truncated) and n-term (complete) KL expansions, respectively. We note that the abscissa of the figure is displayed as the percentage of the ratio m/n where n = 485.*

.. |script_dir| replace:: ``/examples/uncertainty_quant``
.. _script_dir: https://github.com/ameli/restoreio/blob/main/examples/uncertainty_quant/
