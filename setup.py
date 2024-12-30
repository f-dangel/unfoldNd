"""Setup file for unfoldNd.

Use setup.cfg to configure the project.
"""

import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version
from setuptools import setup

try:
    setuptools_version = Version(version("setuptools"))
    required_version = Version("38.3")
    if setuptools_version < required_version:
        print("Error: version of setuptools is too old (<38.3)!")
        sys.exit(1)
except PackageNotFoundError:
    print("setuptools not found")
    sys.exit(1)

if __name__ == "__main__":
    setup(use_scm_version=True)
