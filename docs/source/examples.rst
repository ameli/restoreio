.. _examples:

Examples
********

The following set of instructions is a step-by-step guide to reproduce the result in [R1]_, which is an example of using |project|.

Obtain Input Data
=================

The paper uses the HF radar dataset from Monterey Bay region in California, USA. The HF radar can be accessed publicly through the `national HF radar network gateway <http://cordc.ucsd.edu/projects/mapping/>`__ maintained by the Coastal Observing Research and Development Center. Please follow these steps to obtain the data file.

**Access Link to Data:** The direct link to access the dataset is: `NetcdfSubset <https://hfrnet-tds.ucsd.edu/thredds/ncss/grid/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd/dataset.html>`__.

While the above direct link is all we need, it can alternatively be accessed by navigating through the following links

* Go to `CORDC THREDDS Serever <https://hfrnet-tds.ucsd.edu/thredds/catalog.html>`__.
* From there, click on `HF RADAR, US West Coast <https://hfrnet-tds.ucsd.edu/thredds/HFRADAR_USWC.html>`__.
* Then, click on `HFRADAR US West Coast 2km Resolution Hourly RTV <https://hfrnet-tds.ucsd.edu/thredds/catalog/HFR/USWC/2km/hourly/RTV/catalog.html>`__.
* and then click on `Best Time Series <https://hfrnet-tds.ucsd.edu/thredds/catalog/HFR/USWC/2km/hourly/RTV/catalog.html?dataset=HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd>`__.
* From the *Access options*, click on `NetcdfSubset <https://hfrnet-tds.ucsd.edu/thredds/ncss/grid/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd/dataset.html>`__. 

**Subsetting Monterey Bay Region:** The above link contains the HF radar covering the west of the US and for a continuous time coverage to the present date. We generate a subset of the above data restricted to the Monterey Bay region and the month of January 2017. To do this, on that webpage, first selected the following variables to be included in the input files:

* ``dopx = longitudinal dilution of precision``
* ``dopy = latitudinal dilution of precision``
* ``number_of_radials = number of contributing radials``
* ``number_of_sites = number of contributing radars``
* ``u = surface_eastward_sea_water_velocity``
* ``v = surface_northward_sea_water_velocity``

For the time frame, choose the following start and end dates:

* Start: ``2017-01-01T00:00:00Z``
* End: ``2017-02-01T00:00:00Z``
* Stride: ``1``

We specify two subsets for the Monterey Bay region, with one being a large region covering an extent greater than Monterey Bay (used in **Figure 1**) and one being a smaller region that only focuses on the Monterey Bay (used in **Figure 2** to **Figure 10**).

1. For the *larger* domain, set the lon and lat bounds as:

   * West: ``-122.843``
   * East: ``-121.698``
   * North: ``37.2802``
   * South: ``36.3992``

   Click on `submit`, which then it downloads a netCDF file. Rename this file to ``Monterey_Large_2km_Hourly_2017_01.nc``.
   
2. For the *smaller* domain, set the lon and lat bounds as:

   * West: ``-122.344``
   * East: ``-121.781``
   * North: ``36.9925``
   * South: ``36.507``

   Click on `submit`, which then it downloads a netCDF file. Rename this file to ``Monterey_Small_2km_Hourly_2017_01.nc``.

.. note::

    The specific settings in the above aims to reproduce the results in the paper, however, users may choose other datasets to expriment with |project|. 

Reproducing Results
===================

The scripts to reproduce the results can be found in the source code of |project| under the directory |script_dir|_.

Reproducing Figure 1
--------------------

.. prompt:: bash

    python plot_gdop_coverage.py Monterey_Large_2km_Hourly_2017_01.nc

.. figure:: _static/images/plots/gdop_coverage.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 1:** *Panels (a) and (b) display the east and north GDOP values, respectively, calculated along the northern California coast using the radar configurations indicated by the red dots and their corresponding code names. Panel (c) shows the total GDOP, which is a measure of the overall quality of the estimation of the velocity vector. Panel (d) represents the average HF radar data availability for January 2017, indicating the locations with missing data that generally correspond to high total GDOP values.*

Reproducing Figure 2 to Figure 9
--------------------------------

.. prompt:: bash

    python plot_gdop_coverage.py Monterey_Small_2km_Hourly_2017_01.nc

The above python script executes the following lines of code, which is given in **Listing 1** of the manuscript. Users may change the settings in the function arguments below. For further details, refer to the API reference documentation of the function :func:`restoreio.restore` function.

.. code-block:: python

    >>> # Install restoreio with: pip install restoreio
    >>> from restoreio import restore

    >>> # Here we use the input file corresponding to the smaller domain
    >>> input_file = 'Monterey_Small_2km_Hourly_2017_01.nc'

    >>> # Generate ensembles and reconstruct gaps
    >>> restore(input_file, output='output.nc',
    ...         detect_land=True, fill_coast=True,
    ...         timeframe=117, uncertainty_quant=True,
    ...         scale_error=0.08, kernel_width=5,
    ...         num_ensembles=2000, ratio_num_modes=1,
    ...         plot=True, verbose=True)

The above script generates the output file ``output.nc`` that contains all generated ensembles. Moreover, it creates a subdirectory called ``output_results`` and stores **Figure 2** to **Figure 9** of the manuscript. These plots are shown below.

.. figure:: _static/images/plots/orig_vel_and_error.png
   :align: left
   :figwidth: 100%
   :class: custom-dark

   **Figure 2:** *Panels (a) and (b) show the east and north components of the ocean’s current velocity in the upper 0.3 m th to 2.5 m range, as measured by HF radars in Monterey Bay on January 25 , 2017, at 3:00 UTC. The data has been averaged hourly and mapped to a 2 km resolution Cartesian grid using unweighted least squares. The regions inside the solid black curves represent missing data that was filtered out due to high GDOP values from the original measurement. Panels (c) and (d) respectively show the east and north components of the velocity error computed for the locations where velocity data is available in Panels (a) and (b).*

.. figure:: _static/images/plots/rbf_kernel_2d.png
   :align: left
   :figwidth: 90%
   :class: custom-dark

   **Figure 3:** *The red fields represent the calculated spatial autocorrelation α for the east (a) and north (b) velocity data. The elliptical contour curves are the best fit of the exponential kernel ρ to the autocorrelation. The direction of the principal radii of ellipses is determined by the eigenvectors of M, representing the principal direction of correlation. The radii values are proportional to the eigenvalues of M, representing the correlation length scale. The axes are in the unit of data points spaced 2 km apart.*

.. figure:: _static/images/plots/cor_cov.png
   :align: left
   :figwidth: 90%
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
   :figwidth: 90%
   :class: custom-dark

   **Figure 8:** *The JS distance between the expected distribution q(x, ξ) and the observed distribution p(x, ξ) is shown. The absolute values smaller than 10−8 are rendered as transparent and expose the ocean background, which includes the domain Ω where the JS distance between p(x, ξ) and q(x, ξ) is zero.*

.. figure:: _static/images/plots/kl_eigenvalues.png
   :align: left
   :figwidth: 80%
   :class: custom-dark

   **Figure 9:**  *The eigenvalues λi, i = 1, . . . , n (green curves using left ordinate) and the energy ratio γm, m = 1, . . . , n (blue curves using right ordinate) are shown for the east and north velocity data. The horizontal dashed lines correspond to the 60% and 90% energy ratio levels, respectively, which equate to utilizing nearly 10 and 100 eigenmodes.*

Reproducing Figure 10
---------------------

* First, run ``plot_js_distance.sh`` script:

  .. prompt:: bash
  
      bash plot_js_divergence.sh Monterey_Small_2km_Hourly_2017_01.nc
  
  The above script creates a directory called ``output_js_divergence`` and stores the output files ``output-001.nc`` to ``output-200.nc``.

* Next, run ``plot_js_divergence.py`` script:
  
  .. prompt:: bash
  
      python plot_js_divergence.py
  
.. figure:: _static/images/plots/js_distance.png
 :align: left
 :figwidth: 80%
 :class: custom-dark
 
 **Figure 10:** *The JS distance between the probability distributions pm(x, ξ) and pn(x, ξ) is shown as a function of m = 0, . . . , n. These two distributions correspond to the ensembles generated by the m-term (truncated) and n-term (complete) KL expansions, respectively. We note that the abscissa of the figure is displayed as the percentage of the ratio m/n where n = 485.*

References
==========

.. [R1] Ameli, S., Shadden, S. C. (2023). *Stochastic Modeling of HF Radar Data for Uncertainty Quantification and Gap Filling*. `arXiv: 2206.09976 [physics.ao-ph] <https://arxiv.org/abs/2206.09976>`_


.. |script_dir| replace:: ``/examples/uncertainty_quant``
.. _script_dir: https://github.com/ameli/restoreio/blob/main/examples/uncertainty_quant/
