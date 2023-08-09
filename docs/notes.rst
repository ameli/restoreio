*****
Notes
*****

Journals:

- Computers and Geoscience
- Environmental Modeling and Software
- Journal of Atmospheric and Oceanic Technology
- SoftwareX

- Citation:

  * Add Tom's students thesis, some papers used this software
  * Add conferences I went to present this work, including the "poster",
    and "slides".

Issues
======

**Building on MacOS:**

Since 2023/08, it seems I cannot build restoreio on macos as the
``build-macos.yml`` github action fails. This, however, does not affects macos
users who installed restoreio through pypi or anaconda, because restoreio is
pure python package, hence, its wheels do not depend on OS. The wheels that are
uploaded to PyPI, for instance, are built on ubuntu-latest instance. But they
can be used on other OS, such as macos.

Thus, this issue only affects if someone wants to build restoreio from source
code, not using its binary wheel.
