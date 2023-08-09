.. _getting-started:

Getting Started
===============

.. contents::
   :depth: 2

This user guide offers both a quick overview of package usage and more in-depth details. We recommend starting with the :ref:`Quick Start <quick-start-sec>` section and then acquainting yourself with the function arguments detailed in the following sections. You may find it helpful to read this user guide alongside the :ref:`API reference <api>` for a comprehensive understanding.

Overview of Usage
-----------------

An installation of |project| can be used in two ways:

1. :ref:`As a python package <as_python_package>`
2. :ref:`As a standalone executable <as_standalone_exec>`

.. _as_python_package:

1. As a Python Package
~~~~~~~~~~~~~~~~~~~~~~

You can import |project| in python with ``import restoreio``. This package contains the following functions:

* :func:`restoreio.restore`: This is the main function of the package which can restore incomplete data, generates ensembles, and performs statistical analysis. You may import this function as

  .. code-block:: python

      >>> from restoreio import restore

* :func:`restoreio.scan`: This function performs a pre-scan of your NetCDF dataset (see :ref:`Scan Input Sata <scan-input-data-sec>` for more details). You may import this function as

  .. code-block:: python

      >>> from restoreio import scan

.. _as_standalone_exec:

2. As a Standalone Executable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you may use |project| as a standalone executable (outside of python environment) which can be executed in command line. When this package is installed, the following executables will be available on your Python path:

* `restore <https://ameli.github.io/restoreio/cli_restore.html>`__: This executable is equivalent to :func:`restoreio.restore` function in the Python interface.
* `restore-scan <https://ameli.github.io/restoreio/cli_scan.html>`__: This executable is equivalent to :func:`restoreio.scan` function in the Python interface.

To use these executables, make sure the ``/bin`` directory of your python installation is set on your ``PATH`` environment variable. For instance, if your python is installed on ``/opt/minicinda3/``, add this path ``/opt/miniconda3/bin`` directory to ``PATH`` by

.. prompt:: bash

    export PATH=/opt/minicinda/bin:$PATH

You may place the above line in ``~/.bashrc`` to make the above change permanently.

.. _quick-start-sec:

Quick Start
-----------

|project| has two main purposes:

1. :ref:`Restore incomplete data <quick_restore>`, and
2. :ref:`Generate data ensembles <quick_ensemble>`.

The function :func:`restoreio.restore` serves both of the above purposes. In the followings two sections, we demonstrate simple examples on how to use this function for each of these applications.

.. _quick_restore:

1. Restoring Incomplete Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following demonstrates a minimalistic example of restoring the missing data of an HF radar dataset. We provide this example using both the Python interface and command-line interface of |project|.

Using Python Interface
......................

.. _quick-code-1:

The code below uses the :func:`restoreio.restore` function in the Python interface of |project|:

.. code-block:: python

    >>> # Import package
    >>> from restoreio import restore

    >>> # OpenDap URL of HF radar data, south side of Martha's Vineyard
    >>> input = 'https://transport.me.berkeley.edu/thredds/dodsC/' + \
    ...          'root/MontereyBay/MontereyBay_2km_original.nc'

    >>> # Specify output
    >>> output = '/tmp/output.nc'

    >>> # Specify a time point
    >>> time_point = '2017-01-25T09:00:00'

    >>> # Restore missing velocity data
    >>> restore(input, output=output, time=time_point, detect_land=True,
    ...         fill_coast=True, plot=True, save=False)

The above code makes the following plots where you can compare the input data (left column) and output data (right column). Also, the result of the above is an output file called ``output.nc`` which stores the reconstructed east and north components of the velocity data.

.. image:: ../_static/images/user-guide/velocities.png
   :align: center
   :class: custom-dark

The above code processed one time point specific by ``time`` argument. You can, also process a time interval within the input dataset using ``min_time`` and ``max_time`` arguments:

.. code-block:: python

    >>> # Import package
    >>> from restoreio import restore

    >>> # OpenDap URL of HF radar data, south side of Martha's Vineyard
    >>> input = 'https://transport.me.berkeley.edu/thredds/dodsC/' + \
    ...          'root/MontereyBay/MontereyBay_2km_original.nc'

    >>> # Specify output
    >>> output = '/tmp/output.nc'

    >>> # Subsetting time
    >>> min_time = '2017-01-25T03:00:00'
    >>> max_time = '2017-01-25T09:00:00'

    >>> # Restore missing velocity data
    >>> restore(input, output=output, min_time=min_time, max_time=max_time,
    ...         detect_land=True, save=False)

The output file contains reconstructed variables named ``east_vel`` and ``north_vel``. The following reads the output file and prints the variables within the output file:

.. code-block:: python

    >>> # Reading the output file
    >>> import netCDF4
    >>> nc = netCDF4.Dataset(output)
    >>> nc.variables.keys()
    dict_keys(['time', 'lon', 'lat', 'mask', 'east_vel', 'north_vel'])

For more information about the output variables, see :ref:`Output Variables <output-var-sec>` in this user guide.

Using Command-Line Interface
............................

The same code above can also be invoked using the `restore <https://ameli.github.io/restoreio/cli_restore.html>`__ executable:

.. code-block:: bash

    # OpenDap URL of the dataset
    input='https://transport.me.berkeley.edu/thredds/dodsC/'\
            'root/MontereyBay/MontereyBay_2km_original.nc'

    # Specify output
    output='/tmp/output.nc'

    # Subsetting time
    min_time='2017-01-25T03:00:00'
    max_time='2017-01-25T09:00:00'

    # Restore missing velociy data
    restore -i $input -o $output --min-time $min_time --max-time $max_time -L 2

.. _quick_ensemble:

2. Generation Data Ensembles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second purpose of :func:`restoreio.restore` function serves to generate ensembles of the velocity data together with restoring the missing data within each generate ensemble. Here we demonstrate its usage both in Python and command-line interface.

Using Python Interface
......................

.. code-block:: python

    >>> # Import package
    >>> from restoreio import restore

    >>> # OpenDap URL of HF radar data, south side of Martha's Vineyard
    >>> input = 'https://transport.me.berkeley.edu/thredds/dodsC/' + \
    ...          'root/MontereyBay/MontereyBay_2km_original.nc'

    >>> # Specify output
    >>> output = '/tmp/output.nc'

    >>> # Subsetting spatial domain to the Monterey Bay region, California
    >>> min_lon = -122.344
    >>> max_lon = -121.781
    >>> min_lat = 36.507
    >>> max_lat = 36.992

    >>> # Specify a time point
    >>> time_point = '2017-01-25T09:00:00'

    >>> # Generate ensembles and reconstruct gaps
    >>> restore(input=input, output=output, min_lon=min_lon,
    ...         max_lon=max_lon, min_lat=min_lat, max_lat=max_lat,
    ...         time=time_point, uncertainty_quant=True, num_ensembles=2000,
    ...         ratio_num_modes=1, kernel_width=5, scale_error=0.08,
    ...         detect_land=True, fill_coast=True, write_ensembles=True)

In the above code, we subset the data to the Monterey Bay region in California. The above code generates 2000 ensembles of the velocity data. You may refer to :ref:`Generating Ensembles <generating-ensembles>` for more details. The mean and standard deviation of the ensembles are shown in the left and right panels of the figure below.

.. image:: ../_static/images/user-guide/ensembles.png
   :align: center
   :class: custom-dark

The output file contains the mean of the reconstructed ensembles under the names ``east_vel`` and ``north_vel``. Also, the mean and standard deviation of the ensembles are stored under the names ``east_err`` and ``north_err``. All ensembles are stored by the variables ``east_vel_ensembles`` and ``north_vel_ensembles``.

.. code-block:: python

    >>> # Reading the output file
    >>> import netCDF4
    >>> nc = netCDF4.Dataset(output)
    >>> nc.variables.keys()
    dict_keys(['time', 'lon', 'lat', 'mask', 'east_vel', 'north_vel', 'east_err',
    'north_err', 'east_vel_ensembles', 'north_vel_ensembles'])

Using Command-Line Interface
............................

The same code above can also be invoked using the `restore <https://ameli.github.io/restoreio/cli_restore.html>`__ executable:

.. code-block:: bash

    # OpenDap URL of the dataset
    input='https://transport.me.berkeley.edu/thredds/dodsC/'\
            'root/MontereyBay/MontereyBay_2km_original.nc'

    # Specify output
    output='/tmp/output.nc'

    # Subsetting spatial domain to the Monterey Bay region, California
    min_lon=-122.344
    max_lon=-121.781
    min_lat=36.507
    max_lat=36.992

    # Specifying a time point
    time_point='2017-01-25T03:00:00'

    # Restore missing velociy data
    restore -i $input -o $output --min-lon $min_lon --max-lon $max_lon \
            --min-lat $min_lat --max-lat $max_lat --time $time_point -L 2 -l \
            -u -e 2000 -m 1 -w 5 -E 0.08 -W
