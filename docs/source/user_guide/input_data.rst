.. _input-data:

Input Data
**********

This section offers comprehensive guidance on preparing your datasets to ensure their compatibility as input for |project|.

.. contents::
   :depth: 2

Preparing Input Data
====================

.. _file-format:

File Format
-----------

The input dataset can consist of **one or multiple files**, which should adhere to the following formats:

* **NetCDF** file format with file extensions ``.nc``, ``.nc4``, ``.ncd``, or ``.nc.gz``.
* **NcML** file format with file extensions ``.ncml`` or ``.ncml.gz``. For more information on NcML files, see :ref:`Single Dataset Stored Across Multiple Files <multi-file-single-data-sec>`.

Note that it is also acceptable to provide a NetCDF file without a file extension.

.. _best-practice-sec:

Best Practice for NetCDF Files
------------------------------

It is highly recommended to save your data file in **NetCDF4** format instead of **NetCDF3**. For more information, refer to `NetCDF4 Package documentation <https://unidata.github.io/netcdf4-python/>`__ in Python or `NetCDF Files <https://www.mathworks.com/help/matlab/network-common-data-form.html>`__ in MATLAB.

Also, you may follow the best practice of preparing a NetCDF file, which involves passing the **CF 1.8 compliance** test using `online CF-convention compliance checker <https://compliance.ioos.us/index.html>`__. In particular, this involves adding the ``standard_name`` attribute to your variables (see :ref:`Required NetCDF Variables <req-var-sec>` section below). For reference, you can consult the comprehensive list of `CF compliance standard names <http://cfconventions.org/Data/cf-standard-names/43/build/cf-standard-name-table.html>`__.
.


.. _req-var-sec:

Required NetCDF Variables
-------------------------

An input NetCDF file to |project| should include all the variables listed in the table below. To ensure proper detection of these variables by |project|, each variable should include at least one of the attributes: ``standard_name``, ``name``, or both, as listed in the table. Note that checking the (standard) name is done in a case-insensitive manner.

.. |br| raw:: html

    <br>

.. +--------------------------------+----------------------------------------------------------------------------------------+----------------------------------+
.. | Variable                       | Acceptable Standard Names                                                              | Acceptable Names                 |
.. +================================+========================================================================================+==================================+
.. | Time                           | ``time``                                                                               | ``t``, ``time``, ``datetime``    |
.. +--------------------------------+----------------------------------------------------------------------------------------+----------------------------------+
.. | Longitude                      | ``longitude``                                                                          | ``lon``, ``long``, ``longitude`` |
.. +--------------------------------+----------------------------------------------------------------------------------------+----------------------------------+
.. | Latitude                       | ``latitude``                                                                           | ``lat``, ``latitude``            |
.. +--------------------------------+----------------------------------------------------------------------------------------+----------------------------------+
.. | Ocean's Surface East Velocity  | ``surface_eastward_sea_water_velocity``                                                | ``east_vel``                     |
.. |                                | ``eastward_sea_water_velocity``                                                        | ``eastward_vel``                 |
.. |                                | ``surface_geostrophic_eastward_sea_water_velocity``                                    | ``u``                            |
.. |                                | ``surface_geostrophic_sea_water_x_velocity``                                           | ``ugos``                         |
.. |                                | ``surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid``       | ``east_velocity``                |
.. |                                | ``surface_eastward_geostrophic_sea_water_velocity_assuming_sea_level_for_geoid``       | ``eastward_velocity``            |
.. |                                | ``surface_geostrophic_sea_water_x_velocity_assuming_mean_sea_level_for_geoid``         | ``u-velocity``                   |
.. |                                | ``surface_geostrophic_sea_water_x_velocity_assuming_sea_level_for_geoid``              |                                  |
.. |                                | ``surface_geostrophic_eastward_sea_water_velocity_assuming_mean_sea_level_for_geoid``  |                                  |
.. |                                | ``sea_water_x_velocity``                                                               |                                  |
.. |                                | ``x_sea_water_velocity``                                                               |                                  |
.. +--------------------------------+----------------------------------------------------------------------------------------+----------------------------------+
.. | Ocean's Surface North Velocity | ``surface_northward_sea_water_velocity``                                               | ``north_vel``                    |
.. |                                | ``northward_sea_water_velocity``                                                       | ``northward_vel``                |
.. |                                | ``surface_geostrophic_northward_sea_water_velocity``                                   | ``v``                            |
.. |                                | ``surface_geostrophic_sea_water_y_velocity``                                           | ``vgos``                         |
.. |                                | ``surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid``      | ``north_velocity``               |
.. |                                | ``surface_northward_geostrophic_sea_water_velocity_assuming_sea_level_for_geoid``      | ``northward_velocity``           |
.. |                                | ``surface_geostrophic_sea_water_y_velocity_assuming_mean_sea_level_for_geoid``         | ``v-velocity``                   |
.. |                                | ``surface_geostrophic_sea_water_y_velocity_assuming_sea_level_for_geoid``              |                                  |
.. |                                | ``surface_geostrophic_northward_sea_water_velocity_assuming_mean_sea_level_for_geoid`` |                                  |
.. |                                | ``sea_water_y_velocity``                                                               |                                  |
.. |                                | ``y_sea_water_velocity``                                                               |                                  |
.. +--------------------------------+----------------------------------------------------------------------------------------+----------------------------------+

+--------------------------------+-----------------------+-----------------------------------------------------------------+
| Variable                       | Acceptable Names and Standard Names                                                     |
+================================+=======================+=================================================================+
| Time                           |  *Standard Names:*    | ``time``                                                        |
|                                +-----------------------+-----------------------------------------------------------------+
|                                |  *Names:*             | ``t``, ``time``, ``datetime``                                   |
+--------------------------------+-----------------------+-----------------------------------------------------------------+
| Longitude                      |  *Standard Names:*    | ``longitude``                                                   |
|                                +-----------------------+-----------------------------------------------------------------+
|                                |  *Names:*             | ``lon``, ``long``, ``longitude``                                |
+--------------------------------+-----------------------+-----------------------------------------------------------------+
| Latitude                       |  *Standard Names:*    | ``latitude``                                                    |
|                                +-----------------------+-----------------------------------------------------------------+
|                                |  *Names:*             | ``lat``, ``latitude``                                           |
+--------------------------------+-----------------------+-----------------------------------------------------------------+
| Ocean's Surface East Velocity  | *Standard Names:* |br|                                                                  |
|                                | ``surface_eastward_sea_water_velocity``,                                                |
|                                | ``eastward_sea_water_velocity``,                                                        |
|                                | ``surface_geostrophic_eastward_sea_water_velocity``,                                    |
|                                | ``surface_geostrophic_sea_water_x_velocity``,                                           |
|                                | ``surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid``,       |
|                                | ``surface_eastward_geostrophic_sea_water_velocity_assuming_sea_level_for_geoid``,       |
|                                | ``surface_geostrophic_sea_water_x_velocity_assuming_mean_sea_level_for_geoid``,         |
|                                | ``surface_geostrophic_sea_water_x_velocity_assuming_sea_level_for_geoid``,              |
|                                | ``surface_geostrophic_eastward_sea_water_velocity_assuming_mean_sea_level_for_geoid``,  |
|                                | ``sea_water_x_velocity``,                                                               |
|                                | ``x_sea_water_velocity``                                                                |
+                                +-----------------------+-----------------------------------------------------------------+
|                                | *Names:* |br|                                                                           |
|                                | ``east_vel``,                                                                           |
|                                | ``eastward_vel``,                                                                       |
|                                | ``u``,                                                                                  |
|                                | ``ugos``,                                                                               |
|                                | ``east_velocity``,                                                                      |
|                                | ``eastward_velocity``,                                                                  |
|                                | ``u-velocity``                                                                          |
+--------------------------------+-----------------------+-----------------------------------------------------------------+
| Ocean's Surface North Velocity | *Standard Names:* |br|                                                                  |
|                                | ``surface_northward_sea_water_velocity``,                                               |
|                                | ``northward_sea_water_velocity``,                                                       |
|                                | ``surface_geostrophic_northward_sea_water_velocity``,                                   |
|                                | ``surface_geostrophic_sea_water_y_velocity``,                                           |
|                                | ``surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid``,      |
|                                | ``surface_northward_geostrophic_sea_water_velocity_assuming_sea_level_for_geoid``,      |
|                                | ``surface_geostrophic_sea_water_y_velocity_assuming_mean_sea_level_for_geoid``,         |
|                                | ``surface_geostrophic_sea_water_y_velocity_assuming_sea_level_for_geoid``,              |
|                                | ``surface_geostrophic_northward_sea_water_velocity_assuming_mean_sea_level_for_geoid``, |
|                                | ``sea_water_y_velocity``,                                                               |
|                                | ``y_sea_water_velocity``,                                                               |
+                                +-----------------------+-----------------------------------------------------------------+
|                                | *Names:* |br|                                                                           |
|                                | ``north_vel``,                                                                          |
|                                | ``northward_vel``,                                                                      |
|                                | ``v``,                                                                                  |
|                                | ``vgos``,                                                                               |
|                                | ``north_velocity``,                                                                     |
|                                | ``northward_velocity``,                                                                 |
|                                | ``v-velocity``                                                                          |
+--------------------------------+-----------------------+-----------------------------------------------------------------+

.. _opt-var-sec:

Optional NetCDF Variables
-------------------------

Apart from the required variables mentioned above, you have the option to include the following additional variables in your input file. Note that there is no standard name established for these variables, so you should provide a <code>name</code> attribute according to the table. These variables are used exclusively for the purposes of **uncertainty quantification** by **generating data ensemble**. For more details, you may refer to the :ref:`Generating Ensemble <generating-ensemble>` section.

+---------------------------------------------------+---------------------------+------------------------------+
| Variable                                          | Acceptable Standard Names | Acceptable Names             |
+===================================================+===========================+==============================+
| Ocean's Surface East Velocity Error               | N/A                       | ``east_err``, ``east_error`` |
+---------------------------------------------------+---------------------------+------------------------------+
| Ocean's Surface North Velocity Error              | N/A                       | ``east_err``, ``east_error`` |
+---------------------------------------------------+---------------------------+------------------------------+
| Geometric Dilution of Precision (East Component)  | N/A                       | ``dopx``, ``gdopx``          |
+---------------------------------------------------+---------------------------+------------------------------+
| Geometric Dilution of Precision (North Component) | N/A                       | ``dopx``, ``gdopx``          |
+---------------------------------------------------+---------------------------+------------------------------+

The following provides further details for each of the variables listed in the tables above.

.. _time-var-sec:

1. Time Variable
----------------

The time variable should be a one-dimensional array and strictly increases in values.

Optional Attributes
~~~~~~~~~~~~~~~~~~~

* ``units``: a string specifying both the time unit (such as ``years``, ``months``, ``days``, ``hours``, ``minutes``, ``seconds`` or ``microseconds``) and the origin of the time axis (such as ``since 1970-01-01 00:00:00 UTC``). If this attribute is not provided, the default assumption is ``days since 1970-01-01 00:00:00 UTC``.
* ``calendar``: a string indicating the time calendar. If this attribute is not provided, the default assumption is ``gregorian``.

Masking
~~~~~~~

Ensure that the time variable is not masked. If the ``_FillValue`` attribute is included, the variable will be masked. Therefore, make sure this attribute is not present for the time variable.

.. _lon-lat-var-sec:

2. Longitude and Latitude Variables
-----------------------------------

These variables should be one-dimensional arrays, each representing an axis of a rectilinear grid. The values in both longitude and latitude arrays should either strictly increase or strictly decrease. The units of the arrays should be degrees positive eastward (for longitude) and degrees positive northward (for latitude).

Data on Irregular Grids
~~~~~~~~~~~~~~~~~~~~~~~

|project| is designed to process data on rectilinear grids which are presented by **one-dimensional longitude and latitude arrays**. However, if your data is on irregular grids represented by **two-dimensional longitude and latitude arrays**, you can remap the data to a rectilinear grid by using interpolation functions such as `scipy.interpolate.griddata <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`__ in Python or `griddata <https://www.mathworks.com/help/matlab/ref/griddata.html>`__ in MATLAB.

Masking
~~~~~~~

Ensure that the longitude and latitude variables are not masked. The presence of ``_FillValue`` attribute, for example, will cause these variables to be masked. Therefore, make sure this attribute is not present for the longitude and latitude variables.

.. _ocean-vel-var-sec:

3. Ocean's Surface East and North Velocity Variables
----------------------------------------------------

Unit
~~~~

There is no restriction on the physical unit of the velocity variables; however, they should be oriented positive eastward (for the east component) and positive northward (for the north component).

Array Dimensions
~~~~~~~~~~~~~~~~

The east and north ocean's surface velocity variables should be **three-dimensional** arrays that include dimensions for *time*, *longitude*, and *latitude*. However, you can also provide **four-dimensional** arrays, where an additional dimension represents *depth*. In the latter case, only the **first index** of the depth dimension (representing the surface at near zero depth) will be read from these variables.

Dimensions Order
~~~~~~~~~~~~~~~~

The order of dimensions for a velocity variable, named ``east_vel`` for instance, is as follows:

* For three dimensional arrays, the order should be ``east_vel[time, lat, lon]`` in Python and ``east_vel(lon, lat, time)`` in MATLAB.
* For four dimensional arrays, the order should be ``east_vel[time, depth, lat, lon]`` in Python and ``east_vel(lon, lat, depth, time)`` in MATLAB.

Note that the order of dimensions in MATLAB is reversed compared to Python.

Masking
~~~~~~~

In areas where the velocity is unknown (either due to being located on land or having incomplete data coverage), the velocity variable should be masked using one of the following methods:

* The **recommended approach** is to use **masked arrays** such as by `numpy.ma.MaskArray <https://numpy.org/doc/stable/reference/maskedarray.baseclass.html#maskedarray-baseclass>`__ class in Python or `netcdf.defVarFill <https://www.mathworks.com/help/matlab/ref/netcdf.defvarfill.html>`__ function in MATLAB (only for **NetCDF4**).
*  Set the velocity value on such locations to a large number such as ``9999.0`` and assign the attribute ``missing_value`` or ``_FillValue`` with this value.
*  Set the velocity value on such locations to ``NaN``.

.. _ocean-vel-err-var-sec:

4. Ocean's Surface East and North Velocity Error Variables (Optional)
---------------------------------------------------------------------

When you enable the ``uncertainty_quant`` option in :func:`restoreio.restore` to generate ensemble of velocity field for uncertainty quantification, the east and north velocity error variables are used. However, for uncertainty quantification purposes, you have the alternative option of providing the :ref:`Geometric Dilution of Precision Variables <ocean-gdop-var-sec>` instead of the velocity error variables.

For further details, refer to :ref:`Generating Ensemble <generating-ensemble>` section.

Unit
~~~~

The velocity error variables should be expressed as **non-negative** values and use the **same unit as the velocity** variable, such as both being in meters per second. If your velocity error values are not in the same unit as the velocity variables (e.g., velocity in **meters per second** and velocity error in **centimeters per second**), you can convert the velocity error unit by using the ``scale_error`` argument in :func:`restoreio.restore`. This scale factor will be directly multiplied to the error variables in your files.

Array Dimensions
~~~~~~~~~~~~~~~~

The east and north ocean's surface velocity error variables should be **three-dimensional** arrays that include dimensions for *time*, *longitude*, and *latitude*. However, you can also provide **four-dimensional** arrays, where an additional dimension represents *depth*. In the latter case, only the **first index** of the depth dimension (representing the surface at near zero depth) will be read from these variables.

Dimensions Order
~~~~~~~~~~~~~~~~

The order of dimensions for a velocity error variable, named ``east_vel`` for instance, is as follows:

* For three dimensional arrays, the order should be ``east_vel[time, lat, lon]`` in Python and ``east_vel(lon, lat, time)`` in MATLAB.
* For four dimensional arrays, the order should be ``east_vel[time, depth, lat, lon]`` in Python and ``east_vel(lon, lat, depth, time)`` in MATLAB.

Note that the order of dimensions in MATLAB is reversed compared to Python.

Masking
~~~~~~~

Unlike the velocity variable, masking the velocity error variables is not mandatory. However, if you choose to apply masks to the velocity error variables, the same rules that apply to the velocity variable should also be followed for the velocity error variables.

.. _ocean-gdop-var-sec:

5. Geometric Dilution of Precision Variables (Optional)
-------------------------------------------------------

The Geometric Dilution of Precision (GDOP) is relevant to HF radar datasets, and it quantifies the effect of the geometric configuration of the HF radars on the uncertainty in velocity estimates. To gain a better understanding of the GDOP variables, we recommend referring to Section 2 of :ref:`[2] <ref2>`.

When you enable the ``uncertainty_quant`` option in :func:`restoreio.restore` to generate ensemble of velocity field for uncertainty quantification, the :ref:`Ocean's East and North Velocity Error Variables <ocean-vel-err-var-sec>` are used. However, for uncertainty quantification purposes, you have the alternative option of providing the GDOP variables instead of the velocity error variables.

For further details on the usage of GDOP variables, refer to :ref:`Generating Ensemble <generating-ensemble>` section.

Set Scale Velocity Error Entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When utilizing the GDOP variables instead of the velocity error variables, ensure to specify the ``scale_error`` argument in :func:`restoreio.restore`. This value should be set to the **radial error of HF radars**. The velocity error is then calculated as the product of this scale factor and the GDOP variables.

Unit
~~~~

The GDOP variables should be expressed as **non-negative** values. The GDOP variables are **dimensionless**, however, when the GDOP variables are provided instead of the velocity error, the unit of the ``scale_error`` argument in :func:`restoreio.restore` should be the same unit as your velocity variable. 

Array Dimensions
~~~~~~~~~~~~~~~~

The east and north ocean's surface velocity error variables should be **three-dimensional** arrays that include dimensions for *time*, *longitude*, and *latitude*. However, you can also provide **four-dimensional** arrays, where an additional dimension represents *depth*. In the latter case, only the **first index** of the depth dimension (representing the surface at near zero depth) will be read from these variables.

Dimensions Order
~~~~~~~~~~~~~~~~

The order of dimensions for a velocity error variable, named ``east_vel`` for instance, is as follows:

* For three dimensional arrays, the order should be ``east_vel[time, lat, lon]`` in Python and ``east_vel(lon, lat, time)`` in MATLAB.
* For four dimensional arrays, the order should be ``east_vel[time, depth, lat, lon]`` in Python and ``east_vel(lon, lat, depth, time)`` in MATLAB.

Note that the order of dimensions in MATLAB is reversed compared to Python.

Masking
~~~~~~~

Unlike the velocity variable, masking the velocity error variables is not mandatory. However, if you choose to apply masks to the velocity error variables, the same rules that apply to the velocity variable should also be followed for the velocity error variables.

.. _provide-input-sec:

Providing Input Data
====================

You can provide the input dataset in two different ways:

1. Using files from your local machine.
2. By specifying the URL of data hosted on remote THREDDS data servers.

You can provide either the full path file name of you local files or the *OpenDap* URL of a remote dataset using the ``input`` argument in :func:`restoreio.restore`.

Finding the OpenDap URL from THREDDS Catalogs
---------------------------------------------

Many providers of geophysical data host their datasets on `THREDDS Data servers <https://www.unidata.ucar.edu/software/tds/>`__ , which offer OpenDap protocols. The following steps guide you to obtain the OpenDap URL of a remote dataset hosted on a THREDDS server. In the example below, we use a sample HF radar data hosted on our THREDDS server available at `https://transport.me.berkeley.edu/thredds <https://transport.me.berkeley.edu/thredds>`__.

1. Visit the `catalog webpage <https://transport.me.berkeley.edu/thredds/catalog/catalog.html?dataset=WHOI-HFR/WHOI_HFR_2014_original.nc>`__ of the dataset.
2. From the list of *Service*, select the *OPENDAP* service. This brings you to the `OPENDAP Dataset Access Form <https://transport.me.berkeley.edu/thredds/dodsC/root/WHOI-HFR/WHOI_HFR_2014_original.nc.html>`__ for this dataset.
3. From the OPENDAP Dataset Access Form, find the *Data URL* text box. This contains the OpenDap URL of this dataset, which is:

   .. prompt::

    https://transport.me.berkeley.edu/thredds/dodsC/root/WHOI-HFR/WHOI_HFR_2014_restored.nc 

For a visual demonstration of the steps described above, you may refer to the animated clip.

.. image:: ../_static/images/user-guide/OpenDap.gif
   :align: center
   :class: custom-dark

.. _multi-file-sec:

Multi-File Datasets
===================

You have the option to provide multiple files. A multi-file datasets can appear in two scenarios:

.. _multi-file-single-data-sec:

Single Dataset Stored Across Multiple Files
-------------------------------------------

If your dataset is divided into multiple files, where each file represents a distinct part of the data (e.g., different time frames), you can use the NetCDF Markup Language (NcML) to create an ``ncml`` file that aggregates all the individual NetCDF files into a single dataset. To provide this multi-file dataset, simply specify the URL of the NcML file. For detailed guidance on using NcML, you can consult the `NcML Tutorial <https://docs.unidata.ucar.edu/netcdf-java/4.6/userguide/ncml/Tutorial.html>`__.

.. _multi-file-multiple-data-sec:

Multiple Separate Datasets, Each within a File
----------------------------------------------

Alternatively, you may have several files, with each file representing an independent dataset. An example of such multiple files could be ensemble members obtained from ocean models, where each file corresponds to a velocity ensemble.

The following steps guide you to provide multiple files.

1. Name Your Files with a Numeric Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When providing multiple files, the name of your files (or the URLs) should include a numeric pattern. For instance, you can use the file name format like ``MyInputxxxxFile.nc`` where ``xxxx`` is the numeric pattern. An example of such data URLs where the pattern ranges from ``0000`` to ``0020`` could be:

.. prompt::

    https://transport.me.berkeley.edu/thredds/dodsC/public/SomeDirectory/MyInput0000File.nc
    https://transport.me.berkeley.edu/thredds/dodsC/public/SomeDirectory/MyInput0001File.nc
    https://transport.me.berkeley.edu/thredds/dodsC/public/SomeDirectory/MyInput0002File.nc
    ...
    https://transport.me.berkeley.edu/thredds/dodsC/public/SomeDirectory/MyInput0020File.nc

2. Provide File Iterator Range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide the ``min_file_index`` and ``max_file_index`` arguments in :func:`restoreio.restore` function to define the range of files to be processed. This allows |project| to search through your uploaded files or generate new URLs based on the provided URL to access the other datasets.

For example, in the case of the URLs mentioned earlier, you can enter ``0`` as the minimum file index and ``20`` as the maximum file index. Alternatively, you can specify the full iterator pattern with the leading zeros as ``0000`` to ``0020``.

.. _scan-input-data-sec:

Scan Input Data
===============

It is recommended that you perform a scan of your dataset using the :func:`restoreio.scan` function. This function performs a simple check on your data to make sure required variables exists and are readable. This is often useful if you do not have a priori knowledge on the time and spatial extent of your data. The following code demonstrate scanning of a dataset:

.. code-block:: python
    :emphasize-lines: 9

    >>> # Import package
    >>> from restoreio import scan

    >>> # OpenDap URL of HF radar data
    >>> input = 'https://transport.me.berkeley.edu/thredds/dodsC/' + \
    ...         'root/MontereyBay/MontereyBay_2km_original.nc'

    >>> # Run script
    >>> info = scan(input, scan_velocity=True)

The ``info`` dictionary in the above contains information about the input dataset, such as its spatial extent, time span, and the range of velocity field values. Here is an example of printing this variable:

.. code-block:: python

    >>> import json
    >>> json_obj = json.dumps(info, indent=4)
    >>> print(json_obj)
    {
        "Scan": {
            "ScanStatus": true,
            "Message": ""
        },
        "TimeInfo": {
            "InitialTime": {
                "Year": "2017",
                "Month": "01",
                "Day": "20",
                "Hour": "06",
                "Minute": "00",
                "Second": "00",
                "Microsecond": "000000"
            },
            "FinalTime": {
                "Year": "2017",
                "Month": "01",
                "Day": "25",
                "Hour": "21",
                "Minute": "00",
                "Second": "00",
                "Microsecond": "000000"
            },
            "TimeDuration": {
                "Day": "5",
                "Hour": "15",
                "Minute": "00",
                "Second": "00"
            },
            "TimeDurationInSeconds": "486000.0",
            "DatetimeSize": "136"
        },
        "SpaceInfo": {
            "DataResolution": {
                "LongitudeResolution": "96",
                "LatitudeResolution": "84"
            },
            "DataBounds": {
                "MinLatitude": "36.29128",
                "MidLatitude": "37.03744888305664",
                "MaxLatitude": "37.78362",
                "MinLongitude": "-123.59292",
                "MidLongitude": "-122.6038818359375",
                "MaxLongitude": "-121.614845"
            },
            "DataRange": {
                "LongitudeRange": "175771.3634036201",
                "LatitudeRange": "166126.5386743735",
                "ViewRange": "246079.90876506813",
                "PitchAngle": "45.373085021972656"
            },
            "CameraBounds": {
                "MinLatitude": "35.995285353686455",
                "MaxLatitude": "38.079612412426826",
                "MinLongitude": "-123.98635925709257",
                "MaxLongitude": "-121.22140441478243"
            }
        },
        "VelocityInfo": {
            "EastVelocityName": "u",
            "NorthVelocityName": "v",
            "EastVelocityStandardName": "surface_eastward_sea_water_velocity",
            "NorthVelocityStandardName": "surface_northward_sea_water_velocity",
            "VelocityStandardName": "surface_sea_water_velocity",
            "MinEastVelocity": "-0.6357033237814904",
            "MaxEastVelocity": "0.5624764338135719",
            "MinNorthVelocity": "-0.6599066462367773",
            "MaxNorthVelocity": "0.8097501311451196",
            "TypicalVelocitySpeed": "1.005058422798982"
        }
    }
