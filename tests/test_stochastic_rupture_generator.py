#!/usr/bin/env python3

# Copyright (C) 2022:
#   Cecilia Nievas: cecilia.nievas@gfz-potsdam.de
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
import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from openquake.hazardlib import nrml
from openquake.hazardlib.sourceconverter import RuptureConverter
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.geo.nodalplane import NodalPlane
from realtimelosstools.stochastic_rupture_generator import (StochasticRuptureSet,
                                                            validate_ruptures,
                                                            export_ruptures_to_xml)


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


# Simple source model (3 sources)
SOURCE_MODEL = gpd.GeoDataFrame(
    {"SRC_ID": ["A", "B", "C"],
     "NAME": ["SOURCE A", "SOURCE B", "SOURCE C"],
     "TRT": ["Active Shallow Crust"] * 3,
     "USD": [0.0, 0.0, 0.0],
     "LSD": [20.0, 20.0, 20.0],
     "RATE": [1.0, 1.0, 1.0]},
    geometry=gpd.GeoSeries([
        Polygon(((9.5, 30.5), (10.5, 30.5), (10.5, 29.5), (9.5, 29.5))),
        Polygon(((10.5, 31.5), (11.5, 31.5), (11.5, 30.5), (10.5, 30.5))),
        Polygon(((11.5, 32.5), (12.5, 32.5), (12.5, 31.5), (11.5, 31.5))),
    ]),
    crs="EPSG:4326"
)


# Simple source model PMFS
PMFS = {
    # Strike Slip, 15 km depth
    "A": {"hdd": PMF([(1.0, 15.0)]),
          "npd": PMF([(1.0, NodalPlane(0.0, 90.0, 0.0))])},
    # Reverse, 10 km depth
    "B": {"hdd": PMF([(1.0, 10.0)]),
          "npd": PMF([(1.0, NodalPlane(45.0, 40.0, 90.0))])},
    # Normal, 5 km depth
    "C": {"hdd": PMF([(1.0, 5.0)]),
          "npd": PMF([(1.0, NodalPlane(90.0, 60.0, -90.0))])},
}


# Test catalogue
BASIC_CATALOGUE = pd.DataFrame({
        "longitude": np.array([10.0, 11.0, 12.0]),
        "latitude": np.array([30.0, 31.0, 32.0]),
        "datetime": ["2010-01-01 12:30:00",
                     "2010-01-01 13:30:00",
                     "2010-01-02 14:30:00"],
        "magnitude": np.array([4.0, 4.5, 5.0]),
        "ses_id": np.array([1, 1, 1]),
        "event_id": np.array([1, 2, 3])
    })


def test_rupture_generator_no_depth():
    """Tests generation of ruptures when no depth is available
    """
    catalogue = BASIC_CATALOGUE.copy()
    rupture_generator = StochasticRuptureSet(source_model=SOURCE_MODEL.copy(),
                                             pmfs=PMFS)
    ruptures = rupture_generator.generate_ruptures(catalogue)
    # Expected depths from source model depths
    depths = [15.0, 10.0, 5.0]
    # Expected rakes from source model nodal plane distributions
    rakes = [0.0, 90.0, -90.0]
    mags = [4.0, 4.5, 5.0]
    for rup, depth, rake, mag in zip(ruptures, depths, rakes, mags):
        assert np.isclose(ruptures[rup]["hypocenter"]["depth"], depth)
        assert np.isclose(ruptures[rup]["rake"], rake)
        assert np.isclose(ruptures[rup]["magnitude"], mag)


def test_rupture_generator_depth():
    """Tests generation of ruptures when depth is available
    """
    catalogue = BASIC_CATALOGUE.copy()
    catalogue["depth"] = np.array([4.0, 8.0, 12.0])
    rupture_generator = StochasticRuptureSet(source_model=SOURCE_MODEL.copy(),
                                             pmfs=PMFS)
    ruptures = rupture_generator.generate_ruptures(catalogue)
    # Expected depths the same as those in the catalogue
    depths = [4.0, 8.0, 12.0]
    # Expected rakes from the source model nodal plane distributions
    rakes = [0.0, 90.0, -90.0]
    mags = [4.0, 4.5, 5.0]
    for rup, depth, rake, mag in zip(ruptures, depths, rakes, mags):
        assert np.isclose(ruptures[rup]["hypocenter"]["depth"], depth)
        assert np.isclose(ruptures[rup]["rake"], rake)
        assert np.isclose(ruptures[rup]["magnitude"], mag)


def _rupture_round_trip_validation(catalogue_file, source_model_file, mmin=4.5):
    """Generates ruptures from the test catalogue used for OELF integration tests,
    validates them, exports to xml and then re-loads from the xml using OpenQuake's
    Rupture parser (checks that OpenQuake should be able to read the ruptures)
    """
    # Load catalogue and add ses ID and event IDs
    catalogue = pd.read_csv(catalogue_file, sep=",")
    catalogue["ses_id"] = np.ones(catalogue.shape[0], dtype=int)
    catalogue["event_id"] = np.arange(0, catalogue.shape[0], 1)
    # Generate the ruptures
    stoch_rup = StochasticRuptureSet.from_xml(source_model_file, mmin=mmin)
    ruptures = stoch_rup.generate_ruptures(catalogue)
    # Use in-built validation first
    assert validate_ruptures(ruptures)
    # Export the ruptures xml
    temp_rupture_file = os.path.join(BASE_DATA_PATH, "temp_ruptures")
    if os.path.exists(temp_rupture_file):
        shutil.rmtree(temp_rupture_file)
    export_ruptures_to_xml(ruptures, export_folder=temp_rupture_file)
    # Re-load the ruptures one-by-one using OpenQuake rupture converter
    rupture_files = os.listdir(temp_rupture_file)
    conv = RuptureConverter(0.1)
    for rup_file in rupture_files:
        # Parse file to node object
        [rupture_node] = nrml.read(os.path.join(temp_rupture_file, rup_file))
        # Build rupture from node
        _ = conv.convert_node(rupture_node)
    # Cleanup
    shutil.rmtree(temp_rupture_file)
    return


def test_catalogue_1_round_trip():
    """Uses the catalogue from the integration_oelf test
    """
    cat_file = os.path.join(
        BASE_DATA_PATH,
        os.path.join("integration_oelf", "catalogues", "test_oelf_01.csv")
        )
    source_model_file = os.path.join(
        BASE_DATA_PATH,
        os.path.join("integration_oelf", "ruptures", "source_model_for_oelf.xml")
    )
    _rupture_round_trip_validation(cat_file, source_model_file)


def test_catalogue_2_round_trip():
    """Uses the catalogue from the integration_oelf_no_GMFs test
    """
    cat_file = os.path.join(
        BASE_DATA_PATH,
        os.path.join("integration_oelf_no_GMFs", "catalogues", "test_oelf_01.csv")
        )
    source_model_file = os.path.join(
        BASE_DATA_PATH,
        os.path.join("integration_oelf_no_GMFs", "ruptures", "source_model_for_oelf.xml")
    )
    _rupture_round_trip_validation(cat_file, source_model_file)
