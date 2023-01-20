******
|logo|
******

``restoreio`` is a Python package to **Restore** **I**\ ncomplete **O**\ ceanographic dataset.

``restoreio`` can be installed and used as a standalone Python package or in your browser through the `online gateway interface <http://transport.me.berkeley.edu/restore>`__.

Links
=====

* `Online Gateway <http://transport.me.berkeley.edu/restore>`_
* `Documentation <https://ameli.github.io/restoreio>`_
* `PyPI <https://pypi.org/project/restoreio/>`_
* `Anaconda <https://anaconda.org/s-ameli/restoreio>`_
* `Git Hub <https://github.com/ameli/restore>`_

Install
=======

Install with ``pip``
--------------------

|pypi|

::

    pip install restoreio

Install with ``conda``
----------------------

|conda-version|

::

    conda install -c s-ameli restoreio

Supported Platforms
===================

Successful installation and tests performed on the following operating systems and Python versions:

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

.. |build-linux| image:: https://img.shields.io/github/actions/workflow/status/ameli/restore/build-linux.yml
   :target: https://github.com/ameli/restore/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://img.shields.io/github/actions/workflow/status/ameli/restore/build-macos.yml
   :target: https://github.com/ameli/restore/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://img.shields.io/github/actions/workflow/status/ameli/restore/build-windows.yml
   :target: https://github.com/ameli/restore/actions?query=workflow%3Abuild-windows

Documentation
=============

|deploy-docs| |binder|

See `documentation <https://ameli.github.io/restore/index.html>`__, including:

* `What This Packages Does? <https://ameli.github.io/restore/overview.html>`_
* `Comprehensive Installation Guide <https://ameli.github.io/restore/tutorials/install.html>`_
* `How to Work with Docker Container? <https://ameli.github.io/restore/tutorials/docker.html>`_
* `How to Deploy on GPU Devices? <https://ameli.github.io/restore/tutorials/gpu.html>`_
* `API Reference <https://ameli.github.io/restore/api.html>`_
* `Interactive Notebook Tutorials <https://mybinder.org/v2/gh/ameli/restore/HEAD?filepath=notebooks%2Fquick_start.ipynb>`_
* `Publications <https://ameli.github.io/restore/cite.html>`_

How to Contribute
=================

We welcome contributions via `GitHub's pull request <https://github.com/ameli/restore/pulls>`_. If you do not feel comfortable modifying the code, we also welcome feature requests and bug reports as `GitHub issues <https://github.com/ameli/restore/issues>`_.

How to Cite
===========

If you publish work that uses ``restoreio``, please consider citing the manuscripts available `here <https://ameli.github.io/restore/cite.html>`_.

License
=======

|license|

This project uses a `BSD 3-clause license <https://github.com/ameli/restore/blob/main/LICENSE.txt>`_, in hopes that it will be accessible to most projects. If you require a different license, please raise an `issue <https://github.com/ameli/restore/issues>`_ and we will consider a dual license.

.. |logo| image:: https://raw.githubusercontent.com/ameli/restore/main/docs/source/_static/images/icons/logo-restoreio-light.svg
   :width: 200
.. |license| image:: https://img.shields.io/github/license/ameli/restore
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |deploy-docs| image:: https://img.shields.io/github/actions/workflow/status/ameli/restore/deploy-docs.yml?label=docs
   :target: https://github.com/ameli/restore/actions?query=workflow%3Adeploy-docs
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/restore/HEAD?filepath=notebooks%2Fquick_start.ipynb
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/restore
   :target: https://codecov.io/gh/ameli/restore
.. |pypi| image:: https://img.shields.io/pypi/v/restoreio
   :target: https://pypi.org/project/restoreio/
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/restoreio
   :target: https://anaconda.org/s-ameli/restoreio
