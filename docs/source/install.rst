.. _install:

Install
*******

.. contents::

.. _install-wheels:

Install |project| From Wheels
=============================

Python wheels for |project| are available for various operating systems and Python versions on both PyPI and Anaconda Cloud.

Install with ``pip``
--------------------

|pypi|

Install |project| and its Python dependencies through `PyPI <https://pypi.org/project/restoreio>`_ by

.. prompt:: bash
    
    python -m pip install --upgrade pip
    python -m pip install restoreio

If you are using PyPy instead of Python, install with

.. prompt:: bash
    
    pypy -m pip install --upgrade pip
    pypy -m pip install restoreio

Install with ``conda``
----------------------

|conda-version|

Alternately, install |project| and its Python dependencies from `Anaconda Cloud <https://anaconda.org/s-ameli/restoreio>`_ by

.. prompt:: bash

    conda install -c s-ameli restoreio -y

.. _virtual-env:

Install |project| in Virtual Environments
=========================================

If you do not want the installation to occupy your main python's site-packages (either you are testing or the dependencies may clutter your existing installed packages), install the package in an isolated virtual environment. Two common virtual environments are :ref:`virtualenv <virtualenv_env>` and :ref:`conda <conda_env>`.

.. _virtualenv_env:

Install in ``virtualenv`` Environment
-------------------------------------

1. Install ``virtualenv``:

   .. prompt:: bash

       python -m pip install virtualenv

2. Create a virtual environment and give it a name, such as ``restoreio_env``

   .. prompt:: bash

       python -m virtualenv restoreio_env

3. Activate python in the new environment

   .. prompt:: bash

       source restoreio_env/bin/activate

4. Install ``restoreio`` package with any of the :ref:`above methods <install-wheels>`. For instance:

   .. prompt:: bash

       python -m pip install restoreio
   
   Then, use the package in this environment.

5. To exit from the environment

   .. prompt:: bash

       deactivate

.. _conda_env:

Install in ``conda`` Environment
--------------------------------

In the followings, it is assumed `anaconda <https://www.anaconda.com/products/individual#Downloads>`_ (or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_) is installed.

1. Initialize conda

   .. prompt:: bash

       conda init

   You may need to close and reopen your terminal after the above command. Alternatively, instead of the above, you can do

   .. prompt:: bash

       sudo sh $(conda info --root)/etc/profile.d/conda.sh

2. Create a virtual environment and give it a name, such as ``restoreio_env``

   .. prompt:: bash

       conda create --name restoreio_env -y

   The command ``conda info --envs`` shows the list of all environments. The current environment is marked by an asterisk in the list, which should be the default environment at this stage. In the next step, we will change the current environment to the one we created.

3. Activate the new environment

   .. prompt:: bash

       source activate restoreio_env

4. Install ``restoreio`` with any of the :ref:`above methods <install-wheels>`. For instance:

   .. prompt:: bash

       conda install -c s-ameli restoreio
   
   Then, use the package in this environment.

5. To exit from the environment

   .. prompt:: bash

       conda deactivate

Generate Documentation
======================

Before generating the Sphinx documentation, you should compile the package.

Get the source code from the GitHub repository.

.. prompt:: bash

    git clone https://github.com/ameli/restoreio.git
    cd restoreio

Generate Sphinx Documentation
-----------------------------

Install `Pandoc <https://pandoc.org/>`_ by

.. tab-set::

   .. tab-item:: Ubuntu/Debian
      :sync: ubuntu

      .. prompt:: bash

            sudo apt install pandoc -y

   .. tab-item:: CentOS 7
      :sync: centos

      .. prompt:: bash

          sudo yum install pandoc -y

   .. tab-item:: RHEL 9
      :sync: rhel

      .. prompt:: bash

          sudo dnf install pandoc -y

   .. tab-item:: macOS
      :sync: osx

      .. prompt:: bash

          sudo brew install pandoc -y

   .. tab-item:: Windows (Powershell)
      :sync: win

      .. prompt:: powershell

          scoop install pandoc

Install the requirements for the Sphinx documentation by

.. prompt:: bash

    python -m pip install -r docs/requirements.txt

The above command installs the required packages in Python's path directory. Make sure python's directory is on the `PATH`, for instance, by

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            PYTHON_PATH=`python -c "import os, sys; print(os.path.dirname(sys.executable))"`
            export PATH=${PYTHON_PATH}:$PATH

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: powershell

            $PYTHON_PATH = (python -c "import os, sys; print(os.path.dirname(sys.executable))")
            $env:Path += ";$PYTHON_PATH"

Now, build the documentation:

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            make clean html --directory=docs

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: powershell

            cd docs
            make.bat clean html

The main page of the documentation can be found in ``/docs/build/html/index.html``. 

Test with ``pytest``
====================

|codecov-devel|

The package can be tested by running several `test scripts <https://github.com/ameli/restoreio/tree/main/tests>`_, which test all `sub-packages <https://github.com/ameli/restoreio/tree/main/restoreio>`_ and `examples <https://github.com/ameli/restoreio/tree/main/examples>`_.

Clone the source code from the repository and install the required test packages by

.. prompt:: bash

    git clone https://github.com/ameli/restoreio.git
    cd restoreio
    python -m pip install -r tests/requirements.txt
    python setup.py install

To automatically run all tests, use ``pytest`` which is installed by the above commands.

.. prompt:: bash

    mv restoreio restoreio-do-not-import
    pytest

.. attention::

    To properly run ``pytest``, rename ``/restoreio/restoreio`` directory as shown in the above code. This makes ``pytest`` to properly import |project| from the installed location, not from the source code directory.

Test with ``tox``
=================

To run a test in a virtual environment, use ``tox`` as follows:

1. Clone the source code from the repository:
   
   .. prompt:: bash
       
       git clone https://github.com/ameli/restoreio.git

2. Install `tox <https://tox.wiki/en/latest/>`_:
   
   .. prompt:: bash
       
       python -m pip install tox

3. Run tests by
   
   .. prompt:: bash
       
       cd restoreio
       tox

.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/restoreio
   :target: https://codecov.io/gh/ameli/restoreio
.. |implementation| image:: https://img.shields.io/pypi/implementation/restoreio
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/restoreio
.. |format| image:: https://img.shields.io/pypi/format/restoreio
.. |pypi| image:: https://img.shields.io/pypi/v/restoreio
.. |conda| image:: https://anaconda.org/s-ameli/restoreio/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/restoreio
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/restoreio?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/restoreio
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/restoreio
   :target: https://anaconda.org/s-ameli/restoreio
.. |release| image:: https://img.shields.io/github/v/tag/ameli/restoreio
   :target: https://github.com/ameli/restoreio/releases/
.. |conda-platform| image:: https://anaconda.org/s-ameli/restoreio/badges/platforms.svg
   :target: https://anaconda.org/s-ameli/restoreio
.. |repo-size| image:: https://img.shields.io/github/repo-size/ameli/restoreio
   :target: https://github.com/ameli/restoreio
