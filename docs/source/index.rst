.. module:: restoreio

|project| Documentation
***********************

|deploy-docs|

|project| is a python package to **Restore** **I**\ ncomplete **O**\ ceanographic datasets, with a specific focus on ocean surface velocity data. This package can also generate data ensemble and perform statistical analysis, which allows uncertainty qualification of such datasets.

.. grid:: 4

    .. grid-item-card:: GitHub
        :link: https://github.com/ameli/restoreio
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: PyPI
        :link: https://pypi.org/project/restoreio/
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Anaconda Cloud
        :link: https://anaconda.org/s-ameli/restoreio
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Online Interface
        :link: https://restoreio.org
        :text-align: center
        :class-card: custom-card-link

.. grid:: 4

    .. grid-item-card:: Install
        :link: install
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: User Guide
        :link: user_guide
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: API reference
        :link: api
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Publications
        :link: index_publications
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

Supported Platforms
===================

Successful installation and tests performed on the following operating systems, architectures, and Python versions:

.. |y| unicode:: U+2714
.. |n| unicode:: U+2716

+----------+----------+-------+-------+-------+-----------------+
| Platform | Arch     | Python Version        | Continuous      |
+          |          +-------+-------+-------+ Integration     +
|          |          |  3.9  |  3.10 |  3.11 |                 |
+==========+==========+=======+=======+=======+=================+
| Linux    | X86-64   |  |y|  |  |y|  |  |y|  | |build-linux|   |
+          +----------+-------+-------+-------+                 +
|          | AARCH-64 |  |y|  |  |y|  |  |y|  |                 |
+----------+----------+-------+-------+-------+-----------------+
| macOS    | X86-64   |  |y|  |  |y|  |  |y|  | |build-macos|   |
+          +----------+-------+-------+-------+                 +
|          | ARM-64   |  |y|  |  |y|  |  |y|  |                 |
+----------+----------+-------+-------+-------+-----------------+
| Windows  | X86-64   |  |y|  |  |y|  |  |y|  | |build-windows| |
+          +----------+-------+-------+-------+                 +
|          | ARM-64   |  |y|  |  |y|  |  |y|  |                 |
+----------+----------+-------+-------+-------+-----------------+

.. |build-linux| image:: https://img.shields.io/github/actions/workflow/status/ameli/restoreio/build-linux.yml
   :target: https://github.com/ameli/restoreio/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://img.shields.io/github/actions/workflow/status/ameli/restoreio/build-macos.yml
   :target: https://github.com/ameli/restoreio/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://img.shields.io/github/actions/workflow/status/ameli/restoreio/build-windows.yml
   :target: https://github.com/ameli/restoreio/actions?query=workflow%3Abuild-windows

Python wheels for |project| for all supported platforms and versions in the above are available through `PyPI <https://pypi.org/project/restoreio/>`_ and `Anaconda Cloud <https://anaconda.org/s-ameli/restoreio>`_. If you need |project| on other platforms, architectures, and Python versions, `raise an issue <https://github.com/ameli/restoreio/issues>`_ on GitHub and we build its Python Wheel for you.

Install
=======

|conda-downloads|

.. grid:: 2

    .. grid-item-card:: 

        Install with ``pip`` from `PyPI <https://pypi.org/project/restoreio/>`_:

        .. prompt:: bash
            
            pip install restoreio

    .. grid-item-card::

        Install with ``conda`` from `Anaconda Cloud <https://anaconda.org/s-ameli/restoreio>`_:

        .. prompt:: bash
            
            conda install -c s-ameli -c conda-forge restoreio

For complete installation guide, see:

.. toctree::
    :maxdepth: 2

    Install <install/install>

User Guide
==========

.. toctree::
    :maxdepth: 2

    User Guide <user_guide/user_guide>

API Reference
=============

Check the list of functions, classes, and modules of |project| with their usage, options, and examples.

.. toctree::
    :maxdepth: 2
   
    API Reference <api/api>

Examples
========

.. toctree::
    :maxdepth: 2

    Examples <examples>

Online Web-Based Interface
==========================

Alongside |project| python package, we also offer an online service as a web-based interface for this software. This platform is available at: `https://restoreio.org <https://restoreio.org>`__.

This online gateway allows users to efficiently process both local and remote datasets. The computational tasks are executed on the server side, leveraging the parallel processing capabilities of a high-performance computing cluster. Moreover, the web-based interface seamlessly integrates an interactive globe map, empowering sophisticated visualization of the results within the online platform.

Technical Notes
===============

|tokei|

How to Contribute
=================

We welcome contributions via `GitHub's pull request <https://github.com/ameli/restoreio/pulls>`_. If you do not feel comfortable modifying the code, we also welcome feature requests and bug reports as `GitHub issues <https://github.com/ameli/restoreio/issues>`_.

.. _index_publications:

Publications
============

For information on how to cite |project|, publications, and software packages that used |project|, see:

.. toctree::
    :maxdepth: 2

    Publications <cite>

License
=======

|license|

This project uses a `BSD 3-clause license <https://github.com/ameli/restoreio/blob/main/LICENSE.txt>`_, in hopes that it will be accessible to most projects. If you require a different license, please raise an `issue <https://github.com/ameli/restoreio/issues>`_ and we will consider a dual license.

.. Companion Applications
.. ======================
..
.. .. grid:: 3
..
..    .. grid-item-card:: |traceflows-light| |traceflows-dark|
..        :link: https://ameli.github.io/traceflows/index.html
..        :text-align: center
..        :class-card: custom-card-link
..    
..        An online high-performance computational service for Lagrangian analysis of geophysical flows.

.. |deploy-docs| image:: https://img.shields.io/github/actions/workflow/status/ameli/restoreio/deploy-docs.yml?label=docs
   :target: https://github.com/ameli/restoreio/actions?query=workflow%3Adeploy-docs
.. |deploy-docker| image:: https://img.shields.io/github/actions/workflow/status/ameli/restoreio/deploy-docker.yml?label=build%20docker
   :target: https://github.com/ameli/restoreio/actions?query=workflow%3Adeploy-docker
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/restoreio
   :target: https://codecov.io/gh/ameli/restoreio
.. |license| image:: https://img.shields.io/github/license/ameli/restoreio
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |implementation| image:: https://img.shields.io/pypi/implementation/restoreio
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/restoreio
.. |format| image:: https://img.shields.io/pypi/format/restoreio
.. |pypi| image:: https://img.shields.io/pypi/v/restoreio
.. |conda| image:: https://anaconda.org/s-ameli/traceinv/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/traceinv
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/traceinv?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/traceinv
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/traceinv
   :target: https://anaconda.org/s-ameli/traceinv
.. |conda-downloads| image:: https://img.shields.io/conda/dn/s-ameli/restoreio
   :target: https://anaconda.org/s-ameli/restoreio
.. |tokei| image:: https://tokei.ekzhang.com/b1/github/ameli/restoreio?category=lines
   :target: https://github.com/ameli/restoreio
.. |languages| image:: https://img.shields.io/github/languages/count/ameli/restoreio
   :target: https://github.com/ameli/restoreio
.. |traceflows-light| image:: _static/images/icons/logo-traceflows-light.svg
   :height: 23
   :class: only-light
.. |traceflows-dark| image:: _static/images/icons/logo-traceflows-dark.svg
   :height: 23
   :class: only-dark
.. .. |binder| image:: https://mybinder.org/badge_logo.svg
..    :target: https://mybinder.org/v2/gh/ameli/restoreio/HEAD?filepath=notebooks%2Fquick_start.ipynb
