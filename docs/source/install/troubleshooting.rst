.. _troubleshooting:

Troubleshooting
===============

Issue with ``basemap``
----------------------

When using this package, You may encountered this error:

.. prompt::

    ModuleNotFoundError: No module named 'mpl_toolkits.basemap'

or the following error:

.. prompt::

    FileNotFoundError: [Errno 2] No such file or directory: '/opt/miniconda3/lib/python3.10/site-packages/basemap_data_hires-1.3.2-py3.10.egg/mpl_toolkits/basemap_data/epsg'

To solve these issues, first, install ``libgeos`` library by

.. prompt::

    sudo apt install libgeos3.10.2 libgeos-dev -y


Next, install ``basemap`` package directly thought its `GitHub repository <https://github.com/matplotlib/basemap>`__ as follows. 

.. prompt::

    python -m pip install git+https://github.com/matplotlib/basemap#subdirectory=packages/basemap
    python -m pip install git+https://github.com/matplotlib/basemap#subdirectory=packages/basemap_data
    python -m pip install git+https://github.com/matplotlib/basemap#subdirectory=packages/basemap_data_hires

If the issue is not yet resolved with the above solutions, try reinstalling all prerequisite packages using ``conda`` instead of ``pip`` as follows:

.. prompt::

    conda install -c conda-forge --file conda-recipe/requirements_conda.txt

In the above command, the file ``requirements_conda.txt`` is located in the `source code <https://github.com/ameli/restoreio>`__ under ``/conda-receipe`` directory.

Issue with ``geos``
-------------------

When building the sphinx documentation, you may get this error:

.. prompt::

    Extension error (pydata_sphinx_theme):
    Handler <function _overwrite_pygments_css at 0x7fb8efce2cb0> for event 'build-finished' threw an exception (exception: [Errno 13] Permission denied: '/opt/miniconda3/lib/python3.10/site-packages/geos-0.2.3-py3.10.egg/EGG-INFO/entry_points.txt')
    make: *** [Makefile:20: html] Error 2

To resolve this issue, uninstall, then install the ``geos`` package:

.. prompt::

    python -m pip uninstall geos
    python -m pip install --upgrade geos
