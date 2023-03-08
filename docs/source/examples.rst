.. _examples:

Examples
********

.. contents::


.. code-block:: python

    >>> # Install restoreio with ;#"\texttt{pip install restoreio}"#;
    >>> from restoreio import restore

    >>> # Generate ensembles and reconstruct gaps at ;#$ \OmegaMissing $#;
    >>> restore('input.nc', output='output.nc',
    ...         detect_land=True, fill_coast=True,
    ...         timeframe=117, uncertainty_quant=True,
    ...         scale_error=0.08, kernel_width=5,
    ...         num_ensembles=2000, ratio_num_modes=1,
    ...         plot=True, verbose=True)
