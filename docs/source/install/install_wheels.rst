.. _install-wheels:

Install from Wheels
===================

|project| offers Python wheels for a variety of operating systems and Python versions. These wheels are available on both `PyPI <https://pypi.org/project/restoreio>`_ and `Anaconda Cloud <https://anaconda.org/s-ameli/restoreio>`_, providing a convenient way to install the package using either ``pip`` or ``conda``.

Required Libraries
------------------

|project| requires the ``libgeos`` library and the GNU C++ library. When installing |project| via ``pip``, these dependencies must be installed manually as shown below. However, if you are using ``conda``, these dependencies will be installed automatically during the package installation, so no manual installation is necessary.

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install libgeos-dev gcc libstdc++6 -y

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install geos geos-devel gcc libstdc++ -y

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install geos geos-devel gcc libstdc++ -y

    .. tab-item:: macOS
        :sync: osx

        .. prompt:: bash

            brew install geos gcc


Install with ``pip``
--------------------

|pypi|

Install |project| using ``pip`` by

.. prompt:: bash
    
    pip install restoreio


Install with ``conda``
----------------------

|conda-version|

Alternatively, you can install |project| along with its dependencies by:

.. prompt:: bash

    conda install -c s-ameli -c conda-forge restoreio -y

.. |pypi| image:: https://img.shields.io/pypi/v/restoreio
   :target: https://pypi.org/project/restoreio
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/restoreio
   :target: https://anaconda.org/s-ameli/restoreio
