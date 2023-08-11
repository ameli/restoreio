.. _api:

API Reference
*************

The main functions offered by |project| are as follows. Note that these functions are accessible through both the :ref:`Python Interface <python_interface>` and :ref:`Command-line interface <cli_interface>`.

.. _python_interface:

Python Interface
================

You can import the |project| package using the command ``import traceflows``. The key functions of this package include the following. For further information, refer to the :ref:`Python Interface <as_python_package>` section.

.. autosummary::
    :toctree: generated
    :caption: Functions
    :recursive:
    :template: autosummary/member.rst

    restoreio.restore
    restoreio.scan
    
.. _cli_interface:

Command Line Interface (CLI)
============================

The primary functions listed above are also accessible via the command-line interface. Further information can be found in the :ref:`Command Line Intrerface <as_standalone_exec>` section.

.. toctree:: 
   :maxdepth: 1
   
   restore <cli_restore>
   restore-scan <cli_scan>
