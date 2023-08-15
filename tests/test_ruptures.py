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
import pytest
import pandas as pd
from realtimelosstools.ruptures import Rupture, RLA_Ruptures
from realtimelosstools.utils import Files, Loader


def test_distance_between_coordinates():
    returned_distance = Rupture.distance_between_coordinates(13.4761, 42.2713, 13.506, 42.2772)

    assert round(returned_distance, 3) == 2.546


def test_calculate_depth_of_rupture_bottom():
    returned_bottom_depth = Rupture.calculate_depth_of_rupture_bottom(
        13.4761, 42.2713, 13.506, 42.2772, 13.8, 63.0
    )

    assert round(returned_bottom_depth, 4) == 18.7969


    returned_bottom_depth = Rupture.calculate_depth_of_rupture_bottom(
        13.556, 42.283, 13.466, 42.227, 0.5, 50.0
    )

    assert round(returned_bottom_depth, 4) == 12.0324


def test_build_rupture_from_ITACA_parameters():

    source_params = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "ruptures_data.csv")
    )
    source_params = source_params.set_index(source_params["ITACA_event_id"])

    returned_vals = Rupture.build_rupture_from_ITACA_parameters(
        "IT-YYYY-VVVV", source_params
    )

    returned_strike, returned_dip, returned_rake, returned_hypocenter, returned_rupt_plane = (
        returned_vals
    )

    expected_rupt_plane = {
        "topLeft": {"lon": 13.4, "lat": 42.421, "depth": 0.5},
        "topRight": {"lon": 13.556, "lat": 42.283, "depth": 0.5},
        "bottomLeft": {"lon": 13.31, "lat": 42.366, "depth": 11.9825},
        "bottomRight": {"lon": 13.466, "lat": 42.227, "depth": 11.9825},
    }

    assert round(returned_strike, 5) == 140.0
    assert round(returned_dip, 5) == 50.0
    assert round(returned_rake, 5) == -90.0
    assert round(returned_hypocenter["lon"], 4) == 13.4193
    assert round(returned_hypocenter["lat"], 4) == 42.314
    assert round(returned_hypocenter["depth"], 4) == 8.2

    for corner in expected_rupt_plane:
        for attr in expected_rupt_plane[corner]:
            assert (
                round(expected_rupt_plane[corner][attr], 4)
                == round(returned_rupt_plane[corner][attr], 4)
            )


def test_RLA_Ruptures():
    # Test 1:
    # One earthquake with XML input by user, one earthquake with XML built from CSV, no errors
    main_path = os.path.join(os.path.dirname(__file__), "data", "rla_ruptures_01")
    triggers = Loader.load_triggers(
        os.path.join(main_path, "triggering.csv"),
        os.path.join(main_path, "catalogues")
    )

    returned_rla_ruptures = RLA_Ruptures(triggers, main_path)

    expected_rla_ruptures_mapping = {
        "triggering_01_rla_01.csv": "earthquake_01.xml",
        "triggering_01_rla_02.csv": "built_rupture_IT-2009-0009.xml",
    }

    assert len(returned_rla_ruptures.mapping.keys()) == len(expected_rla_ruptures_mapping.keys())

    for cat_filename in expected_rla_ruptures_mapping:
        assert (
            returned_rla_ruptures.mapping[cat_filename]
            == expected_rla_ruptures_mapping[cat_filename]
        )

    created_xml_path = os.path.join(main_path, "ruptures", "rla", "built_rupture_IT-2009-0009.xml")

    assert os.path.isfile(created_xml_path)
    os.remove(created_xml_path)

    existing_xml_path = os.path.join(main_path, "ruptures", "rla", "earthquake_01.xml")

    assert os.path.isfile(existing_xml_path)

    # Test 2:
    # The rupture XML file indicated in the triggers does not exist
    main_path = os.path.join(os.path.dirname(__file__), "data", "rla_ruptures_02")
    triggers = Loader.load_triggers(
        os.path.join(main_path, "triggering.csv"),
        os.path.join(main_path, "catalogues")
    )

    with pytest.raises(OSError) as excinfo:
        RLA_Ruptures(triggers, main_path)
    assert "OSError" in str(excinfo.type)

    # Test 3:
    # The source parameters CSV cannot be found
    main_path = os.path.join(os.path.dirname(__file__), "data", "rla_ruptures_03")
    triggers = Loader.load_triggers(
        os.path.join(main_path, "triggering.csv"),
        os.path.join(main_path, "catalogues")
    )

    with pytest.raises(OSError) as excinfo:
        RLA_Ruptures(triggers, main_path)
    assert "OSError" in str(excinfo.type)

    # Test 4:
    # The event ID cannot be found in the source parameters CSV
    main_path = os.path.join(os.path.dirname(__file__), "data", "rla_ruptures_04")
    triggers = Loader.load_triggers(
        os.path.join(main_path, "triggering.csv"),
        os.path.join(main_path, "catalogues")
    )

    with pytest.raises(OSError) as excinfo:
        RLA_Ruptures(triggers, main_path)
    assert "OSError" in str(excinfo.type)

    # Test 5:
    # The XML file to be built from the CSV already exists
    main_path = os.path.join(os.path.dirname(__file__), "data", "rla_ruptures_05")
    triggers = Loader.load_triggers(
        os.path.join(main_path, "triggering.csv"),
        os.path.join(main_path, "catalogues")
    )

    with pytest.raises(OSError) as excinfo:
        RLA_Ruptures(triggers, main_path)
    assert "OSError" in str(excinfo.type)
