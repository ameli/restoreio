Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[0.6.8] - 2023-08-08
--------------------

Changed
~~~~~~~

* `build-macos.yml` workflow fails, because ``basemap`` cannot compile from source due to a deprecation warning that recently took into effect and elevated from deprecation warning to into error. This does not raise issue for the installation of ``restoreio`` in PyPI, since this is a purely python-based package and does not need to be compiled in MacOS to build macosx wheel. All wheels are built on ubuntu and distributable on other OS.

[0.0.1] - 2020-10-11
--------------------

Added
~~~~~

* Initial working code.
* README is added.
