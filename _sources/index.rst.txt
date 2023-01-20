.. module:: restoreio

|project| Documentation
***********************

|deploy-docs|

|project| is a modular and high-performance Python package for machine learning using **G**\ aussian process regression with novel algorithms capable of petascale computation on multi-GPU devices.

.. .. toctree::
    :maxdepth: 1

    old/ComputeLogDeterminant.rst
    old/ComputeTraceOfInverse.rst
    old/examples.rst
    old/generate_matrix.rst
    old/InterpolateTraceOfInverse.rst
    old/introduction.rst

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

    .. grid-item-card:: Docker Hub
        :link: https://hub.docker.com/r/sameli/restoreio
        :text-align: center
        :class-card: custom-card-link

.. grid:: 4

    .. grid-item-card:: Install
        :link: install
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Tutorials
        :link: index_tutorials
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

.. Content for performance are not ready. I cnaged this to Publications temporarily.
.. .. grid-item-card:: Performance
..     :link: index_performance
..     :link-type: ref
..     :text-align: center
..     :class-card: custom-card-link

.. Overview
.. ========
..
.. To learn more about |project| functionality, see:
..
.. .. toctree::
..
..     overview

Supported Platforms
===================

Successful installation and tests performed on the following operating systems, architectures, and Python versions:

.. |y| unicode:: U+2714
.. |n| unicode:: U+2716

+----------+--------+-------+-------+-------+-------+-------+-----------------+
| Platform | Arch   | Python Version                        | Continuous      |
+          |        +-------+-------+-------+-------+-------+ Integration     +
|          |        |  3.7  |  3.8  |  3.9  |  3.10 |  3.11 |                 |
+==========+========+=======+=======+=======+=======+=======+=================+
| Linux    | X86-64 |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-linux|   |
+----------+--------+-------+-------+-------+-------+-------+-----------------+
| macOS    | X86-64 |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-macos|   |
+----------+--------+-------+-------+-------+-------+-------+-----------------+
| Windows  | X86-64 |  |n|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-windows| |
+----------+--------+-------+-------+-------+-------+-------+-----------------+

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
            
            conda install -c s-ameli restoreio

For complete installation guide, see:

.. toctree::
    :maxdepth: 2
    :numbered:

    Install <install>

.. _index_tutorials:

Tutorials
=========

|binder|

.. toctree::
    :maxdepth: 1

    Quick Start (jupyter notebook) <notebooks/quick_start.ipynb>

Launch `online interactive notebook <https://mybinder.org/v2/gh/ameli/restoreio/HEAD?filepath=notebooks%2Fquick_start.ipynb>`_ with Binder.

API Reference
=============

Check the list of functions, classes, and modules of |project| with their usage, options, and examples.

.. toctree::
   :maxdepth: 2
   
   API Reference <api>

.. Features
.. ========

.. * **Randomized algorithms** using Hutchinson and stochastic Lanczos quadrature algorithms (see :ref:`Overview <overview>`)
.. * Novel method to **interpolate** matrix functions. See :ref:`Interpolation of Affine Matrix Functions <interpolation>`.
.. * Parallel processing both on **shared memory** and CUDA Capable **multi-GPU** devices.
.. * Sparse covariance
.. * Mixed covariance model, object
.. * Automatic Relevance Determination (ARD)
.. * Jacobian and Hessian based optimization
.. * Learn hyperparameters in reduced space (profile likelihood)
.. * Prediction in dual space with with :math:`\mathcal{O}(n)` complexity.

Technical Notes
===============

|tokei| |languages|

.. Some notable implementation techniques used to develop |project| are:


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

.. Related Projects
.. ================
..
.. .. grid:: 3
..
..    .. grid-item-card:: |imate-light| |imate-dark|
..        :link: https://ameli.github.io/imate/index.html
..        :text-align: center
..        :class-card: custom-card-link
..    
..        A high-performance python package for scalable randomized algorithms for matrix functions in machine learning.
..
..    .. grid-item-card:: |detkit-light| |detkit-dark|
..        :link: https://ameli.github.io/detkit/index.html
..        :text-align: center
..        :class-card: custom-card-link
..
..        A python package for matrix determinant functions used in machine learning.
..
..    .. grid-item-card:: |special-light| |special-dark|
..       :link: https://ameli.github.io/special_functions/index.html
..       :text-align: center
..       :class-card: custom-card-link
..
..       A python package providing both Python and Cython interface for special mathematical functions.

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
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/restoreio/HEAD?filepath=notebooks%2Fquick_start.ipynb
.. |conda-downloads| image:: https://img.shields.io/conda/dn/s-ameli/restoreio
   :target: https://anaconda.org/s-ameli/restoreio
.. |tokei| image:: https://tokei.rs/b1/github/ameli/restoreio?category=lines
   :target: https://github.com/ameli/restoreio
.. .. |tokei-2| image:: https://img.shields.io/badge/code%20lines-31.9k-blue
..    :target: https://github.com/ameli/restoreio
.. |languages| image:: https://img.shields.io/github/languages/count/ameli/restoreio
   :target: https://github.com/ameli/restoreio
.. .. |imate-light| image:: _static/images/icons/logo-imate-light.svg
..    :height: 23
..    :class: only-light
.. .. |imate-dark| image:: _static/images/icons/logo-imate-dark.svg
..    :height: 23
..    :class: only-dark
.. .. |detkit-light| image:: _static/images/icons/logo-detkit-light.svg
..    :height: 27
..    :class: only-light
.. .. |detkit-dark| image:: _static/images/icons/logo-detkit-dark.svg
..    :height: 27
..    :class: only-dark
.. .. |special-light| image:: _static/images/icons/logo-special-light.svg
..    :height: 24
..    :class: only-light
.. .. |special-dark| image:: _static/images/icons/logo-special-dark.svg
..    :height: 24
..    :class: only-dark
