#!/usr/bin/env python3

# Copyright (C) 2026:
#   Cecilia Nievas: cecilia.nievas@gfz.de
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
from openquake.hazardlib import nrml
from realtimelosstools.point_source_generator import PointSourceSet

def test_fill_in_missing_fields():
    # Test 1: default depth ignored if catalogue has a "depth" column
    catalogue = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_for_point_sources.csv")
    )

    default_depth = -99.0

    source_gen = PointSourceSet(default_depth)

    assert round(source_gen.default_depth, 5) == round(default_depth, 5)
    assert round(source_gen.default_strike, 5) == 0.0
    assert round(source_gen.default_dip, 5) == 90.0
    assert round(source_gen.default_rake, 5) == 0.0

    catalogue_full = source_gen.fill_in_missing_fields(catalogue)

    # "default_depth" should have been ignored because the catalogue has a "depth" column
    assert (catalogue_full["depth"] != default_depth).all()


    # Test 2: all default values assigned because the four columns are missing from catalogue
    catalogue.drop(columns=["depth", "strike", "dip", "rake"], inplace=True)

    default_depth = 2.0
    default_rake = -90.0

    source_gen = PointSourceSet(default_depth, default_rake=default_rake)

    assert round(source_gen.default_depth, 5) == round(default_depth, 5)
    assert round(source_gen.default_strike, 5) == 0.0
    assert round(source_gen.default_dip, 5) == 90.0
    assert round(source_gen.default_rake, 5) == default_rake

    catalogue_full = source_gen.fill_in_missing_fields(catalogue)

    assert np.isclose(catalogue_full["depth"], default_depth).all()
    assert np.isclose(catalogue_full["strike"], 0.0).all()
    assert np.isclose(catalogue_full["dip"], 90.0).all()
    assert np.isclose(catalogue_full["rake"], default_rake).all()


def test_generate_ruptures():
    catalogue = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_for_point_sources.csv")
    )

    default_depth = -99.0
    source_gen = PointSourceSet(default_depth)

    # Create temporary directory to save the ruptures
    temp_path_to_ruptures = os.path.join(
        os.path.dirname(__file__), "data", "temp_point_ruptures"
    )
    os.makedirs(temp_path_to_ruptures, exist_ok=True)

    ruptures = source_gen.generate_ruptures(
        catalogue,
        temp_path_to_ruptures,
        export_type='xml'
    )

    # Check that all rupture files have been generated 
    ruptures_should_exist = [
        "RUP_1-1-1.xml", "RUP_1-1-2.xml", "RUP_1-1-3.xml", "RUP_1-1-4.xml",
        "RUP_1-2-1.xml",
        "RUP_1-3-1.xml", "RUP_1-3-2.xml", "RUP_1-3-3.xml",
        "RUP_1-4-1.xml", "RUP_1-4-2.xml"
    ]

    for rupt_filename in ruptures_should_exist:
        assert os.path.exists(os.path.join(temp_path_to_ruptures, rupt_filename))

    # Comparison of XML files
    xml_pairs = [
        ("RUP_1-1-1.xml", "expected_pointsource_RUP_1-1-1.xml"),
        ("RUP_1-1-2.xml", "expected_pointsource_RUP_1-1-2.xml")
    ]

    def load_nrml_file(filename):
        with open(filename, "rb") as f:
            return nrml.read(f) # nrml.read returns a list of objects

    for xml_pair in xml_pairs:
        # File created by this test
        returned_objects = load_nrml_file(os.path.join(temp_path_to_ruptures, xml_pair[0]))
        # Expected file
        expected_objects = load_nrml_file(
            os.path.join(os.path.dirname(__file__), "data", xml_pair[1])
        )

        assert len(returned_objects) == len(expected_objects)

        for idx, (obj1, obj2) in enumerate(zip(returned_objects, expected_objects)):
            assert obj1 == obj2

    # Remove "temp_path_to_ruptures"
    shutil.rmtree(temp_path_to_ruptures)
