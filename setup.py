#!/usr/bin/env python3

# Copyright (C) 2022:
#   Helmholtz-Zentrum Potsdam Deutsches GeoForschungsZentrum GFZ
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
# General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from setuptools import setup, find_packages

tests_require = ["pytest"]

setup(
    name="real-time-loss-tools",
    version="0.0.1",
    description="",
    keywords="earthquake damage, earthquake loss, rapid loss assessment, operational earthquake loss forecasting",
    author="Helmholtz-Zentrum Potsdam Deutsches GeoForschungsZentrum GFZ",
    license="AGPLv3+",
    install_requires=[
        "pyyaml",
        "numpy",
        "pandas",
        "rtree",
        "shapely",
        "pyproj",
        "geopandas",
        "openquake.engine==3.14.0",
    ],
    extras_require={
        "tests": tests_require,
    },
    packages=find_packages(),
    entry_points={"console_scripts": ["rtlt = realtimelosstools.realtimelosstools:main"]},
    python_requires=">=3.8",
)
