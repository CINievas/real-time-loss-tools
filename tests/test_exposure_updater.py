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
from copy import deepcopy
import pytest
import numpy as np
import pandas as pd
from realtimelosstools.exposure_updater import ExposureUpdater


def test_create_mapping_asset_id_building_id():
    filepath = os.path.join(os.path.dirname(__file__), "data", "exposure_model.csv")
    exposure = pd.read_csv(filepath)

    # Test case in which the index of 'exposure' is not 'asset_id'
    with pytest.raises(OSError) as excinfo:
        ExposureUpdater.create_mapping_asset_id_building_id(exposure)
    assert "OSError" in str(excinfo.type)

    # Test "normal" case
    exposure.index = exposure["id"]
    exposure.index = exposure.index.rename("asset_id")
    exposure = exposure.drop(columns=["id"])

    # Expected mapping
    expected_mapping = pd.DataFrame(
        {"building_id": ["osm_1", "osm_1", "tile_8", "tile_8", "shm_1"]},
        index=["res_1", "res_2", "res_3", "res_4", "res_5"],
    )
    expected_mapping.index = expected_mapping.index.rename("asset_id")

    # Execute the method
    returned_mapping = ExposureUpdater.create_mapping_asset_id_building_id(exposure)

    assert returned_mapping.shape[0] == expected_mapping.shape[0]
    assert (returned_mapping.index == expected_mapping.index).all
    assert returned_mapping.index.name == expected_mapping.index.name

    for asset_id in expected_mapping.index:
        assert asset_id in returned_mapping.index
        assert (
            returned_mapping.loc[asset_id, "building_id"]
            == expected_mapping.loc[asset_id, "building_id"]
        )


def test_merge_damage_results_OQ_SHM():
    # First test case: straightforward replacement (one building_id of SHM corresponds to one
    # asset_id of OQ

    # Damage results from OpenQuake
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_OQ_0.csv")
    damage_results_OQ = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_OQ["asset_id"], damage_results_OQ["dmg_state"]]
    )
    damage_results_OQ.index = new_index
    damage_results_OQ = damage_results_OQ.drop(columns=["asset_id", "dmg_state"])

    # Damage results from Structural Health Monitoring
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_SHM_0.csv")
    damage_results_SHM = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_SHM["building_id"], damage_results_SHM["dmg_state"]]
    )
    damage_results_SHM.index = new_index
    damage_results_SHM = damage_results_SHM.drop(columns=["dmg_state"])

    # Mapping of asset_id and building_id
    id_asset_building_mapping = pd.DataFrame(
        {"building_id": ["osm_1", "osm_1", "tile_8", "tile_8", "shm_1"]},
        index=["res_1", "res_2", "res_3", "res_4", "res_5"],
    )
    id_asset_building_mapping.index = id_asset_building_mapping.index.rename("asset_id")

    # Expected merged damage results
    expected_damage_results_merged = deepcopy(damage_results_OQ)
    for dmg in ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"]:
        expected_damage_results_merged.loc[("res_5", dmg), "value"] = damage_results_SHM.loc[
            ("shm_1", dmg), "value"
        ]

    # Execute the method
    returned_damage_results_merged = ExposureUpdater.merge_damage_results_OQ_SHM(
        damage_results_OQ, pd.Series(damage_results_SHM.loc[:, "value"]), id_asset_building_mapping
    )

    assert returned_damage_results_merged.shape[0] == expected_damage_results_merged.shape[0]
    assert (returned_damage_results_merged.index == expected_damage_results_merged.index).all
    assert (
        returned_damage_results_merged.index.name == expected_damage_results_merged.index.name
    )

    for multiindex in expected_damage_results_merged.index:
        assert multiindex in returned_damage_results_merged.index
        assert round(returned_damage_results_merged.loc[multiindex, "value"], 5) == round(
            expected_damage_results_merged.loc[multiindex, "value"], 5
        )

    # Second test case: one building_id of SHM corresponds to several values of asset_id of OQ

    # Damage results from OpenQuake
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_OQ_1.csv")
    damage_results_OQ = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_OQ["asset_id"], damage_results_OQ["dmg_state"]]
    )
    damage_results_OQ.index = new_index
    damage_results_OQ = damage_results_OQ.drop(columns=["asset_id", "dmg_state"])

    # Damage results from Structural Health Monitoring
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_SHM_1.csv")
    damage_results_SHM = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_SHM["building_id"], damage_results_SHM["dmg_state"]]
    )
    damage_results_SHM.index = new_index
    damage_results_SHM = damage_results_SHM.drop(columns=["dmg_state"])

    # Mapping of asset_id and building_id
    id_asset_building_mapping = pd.DataFrame(
        {
            "building_id": [
                "osm_1", "osm_1", "osm_1", "osm_1", "osm_1",
                "osm_1", "osm_1", "osm_1", "osm_1","osm_1",
                "tile_8", "tile_8", "tile_8", "tile_8", "tile_8",
                "tile_8", "tile_8", "tile_8", "tile_8", "tile_8",
                "shm_1", "shm_1", "shm_1", "shm_1", "shm_1",
            ]
        },
        index=[
            "res_1", "res_2", "res_3", "res_4", "res_5", "res_6", "res_7", "res_8", "res_9",
            "res_10", "res_11", "res_12", "res_13", "res_14", "res_15", "res_16", "res_17",
            "res_18", "res_19", "res_20", "res_21", "res_22", "res_23", "res_24", "res_25",
        ],
    )
    id_asset_building_mapping.index = id_asset_building_mapping.index.rename("asset_id")

    # Expected merged damage results
    expected_damage_results_merged = deepcopy(damage_results_OQ)
    for dmg in ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"]:
        expected_damage_results_merged.loc[("res_21", dmg), "value"] = (
            damage_results_SHM.loc[("shm_1", dmg), "value"] / 5.0
        )
        expected_damage_results_merged.loc[("res_22", dmg), "value"] = (
            damage_results_SHM.loc[("shm_1", dmg), "value"] / 5.0
        )
        expected_damage_results_merged.loc[("res_23", dmg), "value"] = (
            damage_results_SHM.loc[("shm_1", dmg), "value"] / 5.0
        )
        expected_damage_results_merged.loc[("res_24", dmg), "value"] = (
            damage_results_SHM.loc[("shm_1", dmg), "value"] / 5.0
        )
        expected_damage_results_merged.loc[("res_25", dmg), "value"] = (
            damage_results_SHM.loc[("shm_1", dmg), "value"] / 5.0
        )

    # Execute the method
    returned_damage_results_merged = ExposureUpdater.merge_damage_results_OQ_SHM(
        damage_results_OQ, pd.Series(damage_results_SHM.loc[:, "value"]), id_asset_building_mapping
    )

    assert returned_damage_results_merged.shape[0] == expected_damage_results_merged.shape[0]
    assert (returned_damage_results_merged.index == expected_damage_results_merged.index).all
    assert (
        returned_damage_results_merged.index.name == expected_damage_results_merged.index.name
    )

    for multiindex in expected_damage_results_merged.index:
        assert multiindex in returned_damage_results_merged.index
        assert round(returned_damage_results_merged.loc[multiindex, "value"], 5) == round(
            expected_damage_results_merged.loc[multiindex, "value"], 5
        )


def test_update_exposure_with_damage_states():
    """
    The test carries out two cycles of update, because the second cycle needs to re-group assets
    but the first cycle does not.
    """

    # Time of the day of the earthquake
    earthquake_time_of_day = "night"

    # Columns to check
    cols_to_check_numeric = ["lon", "lat", "number", "census", earthquake_time_of_day]
    cols_to_check_numeric_lower_precision = ["structural"]
    cols_to_check_str = ["taxonomy", "building_id"]

    # Mapping between the names of damage states
    mapping_aux = {
        "dmg_state": ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"],
        "fragility": ["DS0", "DS1", "DS2", "DS3", "DS4"],
    }
    mapping_damage_states = pd.DataFrame(
        mapping_aux, columns=["fragility"], index=mapping_aux["dmg_state"]
    )
    mapping_damage_states.index = mapping_damage_states.index.rename("asset_id")

    # Initial exposure model
    filepath = os.path.join(os.path.dirname(__file__), "data", "exposure_model.csv")
    initial_exposure = pd.read_csv(filepath)
    initial_exposure.index = initial_exposure["id"]
    initial_exposure.index = initial_exposure.index.rename("asset_id")
    initial_exposure = initial_exposure.drop(columns=["id"])

    # Damage results from OpenQuake, first cycle
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_OQ_0.csv")
    damage_results_OQ = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_OQ["asset_id"], damage_results_OQ["dmg_state"]]
    )
    damage_results_OQ.index = new_index
    damage_results_OQ = damage_results_OQ.drop(columns=["asset_id", "dmg_state"])

    # Damage results from Structural Health Monitoring, first cycle
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_SHM_0.csv")
    damage_results_SHM = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_SHM["building_id"], damage_results_SHM["dmg_state"]]
    )
    damage_results_SHM.index = new_index
    damage_results_SHM = damage_results_SHM.drop(columns=["dmg_state"])

    # Expected updated exposure model, first cycle
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_1.csv"
    )
    expected_exposure_model_1 = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [expected_exposure_model_1["asset_id"], expected_exposure_model_1["dmg_state"]]
    )
    expected_exposure_model_1.index = new_index
    expected_exposure_model_1 = expected_exposure_model_1.drop(
        columns=["asset_id", "dmg_state"]
    )

    # Execute the method, first cycle
    returned_exposure_model_1 = ExposureUpdater.update_exposure_with_damage_states(
        initial_exposure,
        initial_exposure,
        damage_results_OQ,
        mapping_damage_states,
        earthquake_time_of_day,
        damage_results_SHM=pd.Series(damage_results_SHM.loc[:, "value"]),
    )

    assert returned_exposure_model_1.shape[0] == expected_exposure_model_1.shape[0]

    for multiindex in expected_exposure_model_1.index:
        assert multiindex in returned_exposure_model_1.index

        for col in cols_to_check_str:
            assert (
                returned_exposure_model_1.loc[multiindex, col]
                == expected_exposure_model_1.loc[multiindex, col]
            )

        for col in cols_to_check_numeric:
            assert round(returned_exposure_model_1.loc[multiindex, col], 5) == round(
                expected_exposure_model_1.loc[multiindex, col], 5
            )

        for col in cols_to_check_numeric_lower_precision:
            assert round(returned_exposure_model_1.loc[multiindex, col], 2) == round(
                expected_exposure_model_1.loc[multiindex, col], 2
            )

    for col in ["day", "transit"]:
        assert col not in returned_exposure_model_1

    # Initial exposure model, second cycle
    initial_exposure_updated = deepcopy(returned_exposure_model_1)
    initial_exposure_updated.index = initial_exposure_updated["id"]
    initial_exposure_updated.index = initial_exposure_updated.index.rename("asset_id")
    initial_exposure_updated = initial_exposure_updated.drop(columns=["id"])

    # Damage results from OpenQuake, second cycle
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_OQ_1.csv")
    damage_results_OQ = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_OQ["asset_id"], damage_results_OQ["dmg_state"]]
    )
    damage_results_OQ.index = new_index
    damage_results_OQ = damage_results_OQ.drop(columns=["asset_id", "dmg_state"])

    # Damage results from Structural Health Monitoring, second cycle
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_SHM_1.csv")
    damage_results_SHM = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_SHM["building_id"], damage_results_SHM["dmg_state"]]
    )
    damage_results_SHM.index = new_index
    damage_results_SHM = damage_results_SHM.drop(columns=["dmg_state"])

    # Expected updated exposure model, second cycle
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_2.csv"
    )
    expected_exposure_model_2 = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [expected_exposure_model_2["asset_id"], expected_exposure_model_2["dmg_state"]]
    )
    expected_exposure_model_2.index = new_index
    expected_exposure_model_2 = expected_exposure_model_2.drop(
        columns=["asset_id", "dmg_state"]
    )

    # Execute the method, second cycle
    returned_exposure_model_2 = ExposureUpdater.update_exposure_with_damage_states(
        initial_exposure_updated,
        initial_exposure,
        damage_results_OQ,
        mapping_damage_states,
        earthquake_time_of_day,
        damage_results_SHM=pd.Series(damage_results_SHM.loc[:, "value"]),
    )

    assert returned_exposure_model_2.shape[0] == expected_exposure_model_2.shape[0]

    for multiindex in expected_exposure_model_2.index:
        assert multiindex in returned_exposure_model_2.index

        for col in cols_to_check_str:
            assert (
                returned_exposure_model_2.loc[multiindex, col]
                == expected_exposure_model_2.loc[multiindex, col]
            )

        for col in cols_to_check_numeric:
            assert round(returned_exposure_model_2.loc[multiindex, col], 5) == round(
                expected_exposure_model_2.loc[multiindex, col], 5
            )

        for col in cols_to_check_numeric_lower_precision:
            assert round(returned_exposure_model_2.loc[multiindex, col], 2) == round(
                returned_exposure_model_2.loc[multiindex, col], 2
            )

    for col in ["day", "transit"]:
        assert col not in returned_exposure_model_2


def test_ensure_no_negative_damage_results_OQ():
    # Test case in which values are adjusted
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_OQ_negative.csv")
    damage_results_OQ = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_OQ["asset_id"], damage_results_OQ["dmg_state"]]
    )
    damage_results_OQ.index = new_index
    damage_results_OQ = damage_results_OQ.drop(columns=["asset_id", "dmg_state"])

    expected_damage_results_OQ = deepcopy(damage_results_OQ)
    expected_damage_results_OQ.loc[("res_11", "no_damage"), "value"] = 0.0
    expected_damage_results_OQ.loc[("res_11", "dmg_1"), "value"] = 18.47959741
    expected_damage_results_OQ.loc[("res_11", "dmg_2"), "value"] = 0.113728624
    expected_damage_results_OQ.loc[("res_11", "dmg_3"), "value"] = 0.011640353
    expected_damage_results_OQ.loc[("res_11", "dmg_4"), "value"] = 0.010561345
    expected_damage_results_OQ.loc[("res_13", "no_damage"), "value"] = 0.0
    expected_damage_results_OQ.loc[("res_13", "dmg_1"), "value"] = 0.0
    expected_damage_results_OQ.loc[("res_13", "dmg_2"), "value"] = 0.0
    expected_damage_results_OQ.loc[("res_13", "dmg_3"), "value"] = 1.588763707
    expected_damage_results_OQ.loc[("res_13", "dmg_4"), "value"] = 0.321509105

    returned_damage_results_OQ = ExposureUpdater.ensure_no_negative_damage_results_OQ(
        damage_results_OQ, tolerance=0.0001
    )

    assert returned_damage_results_OQ.shape == expected_damage_results_OQ.shape

    for multiindex in expected_damage_results_OQ.index:
        assert round(returned_damage_results_OQ.loc[multiindex, "value"], 5) == round(
            expected_damage_results_OQ.loc[multiindex, "value"], 5
        )

    # Test case in which there is nothing to adjust (input the already adjusted case)
    returned_damage_results_OQ = ExposureUpdater.ensure_no_negative_damage_results_OQ(
        expected_damage_results_OQ, tolerance=0.0001
    )

    for multiindex in expected_damage_results_OQ.index:
        assert round(returned_damage_results_OQ.loc[multiindex, "value"], 5) == round(
            expected_damage_results_OQ.loc[multiindex, "value"], 5
        )

    # Test case in which the negative values do not comply with the tolerance
    with pytest.raises(ValueError) as excinfo:
        ExposureUpdater.ensure_no_negative_damage_results_OQ(
            damage_results_OQ, tolerance=0.00001
        )
    assert "ValueError" in str(excinfo.type)


def test_summarise_damage_states_per_building_id():
    # Read exposure model
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_2.csv"
    )
    exposure = pd.read_csv(filepath)

    returned_damage_summary = ExposureUpdater.summarise_damage_states_per_building_id(exposure)

    # Expected output
    building_id = [
        "osm_1", "osm_1", "osm_1", "osm_1", "osm_1",
        "tile_8", "tile_8", "tile_8", "tile_8", "tile_8",
        "shm_1", "shm_1", "shm_1", "shm_1", "shm_1",
    ]
    damage_state = [
        "DS1", "DS2", "DS3", "DS4", "DS0",
        "DS1", "DS2", "DS3", "DS4", "DS0",
        "DS1", "DS2", "DS3", "DS4", "DS0",
    ]
    number = [
        0.26460062, 0.05589356, 0.03028228, 0.10033112, 0.54889240,
        22.29490758, 4.89183173, 2.26042970, 5.83527250, 64.71756050,
        0.2, 0.4, 0.1, 0.03, 0.27
    ]
    expected_damage_summary = pd.DataFrame(
        {"number": number},
        index=pd.MultiIndex.from_arrays([building_id, damage_state]),
    )

    for index in expected_damage_summary.index:
        assert round(returned_damage_summary.loc[index, "number"], 5) == round(
            expected_damage_summary.loc[index, "number"], 5
        )


def test_get_unique_exposure_locations():
    filepath = os.path.join(os.path.dirname(__file__), "data", "exposure_model.csv")
    exposure = pd.read_csv(filepath)

    returned_lons, returned_lats = ExposureUpdater.get_unique_exposure_locations(exposure)

    expected_lons = np.array([13.400949, 13.3888, 13.400949])
    expected_lats = np.array([42.344967, 42.344967 ,42.3358])

    # The order of the expected and returned values might not be the same --> re-order them first
    new_order_expected = (expected_lons + expected_lats).argsort()
    new_order_returned = (returned_lons + returned_lats).argsort()

    expected_lons = expected_lons[new_order_expected]
    expected_lats = expected_lats[new_order_expected]
    returned_lons = returned_lons[new_order_returned]
    returned_lats = returned_lats[new_order_returned]

    for i in range(len(expected_lons)):

        assert round(returned_lons[i], 6) == round(expected_lons[i], 6)
        assert round(returned_lats[i], 6) == round(expected_lats[i], 6)


def test_update_exposure_occupants():
    """
    The test comprises three cases:
        1) A first earthquake for which no previous earthquakes have been run.
        2) An earthquake for which previous earthquakes have been run and for which the
        occupancy factors that take into account damage and inspection times are not all null.
        3) An earthquake for which previous earthquakes have been run and for which the
        occupancy factors that take into account damage and inspection times are all null.
    """

    # PARAMETERS COMMON TO ALL TESTS
    # Columns to check
    cols_to_check_numeric = ["lon", "lat", "number", "census", "night"]
    cols_to_check_numeric_lower_precision = ["structural"]
    cols_to_check_str = ["taxonomy", "building_id"]

    # Time of day factors
    time_of_day_factors = {
        "residential": {"day": 0.242853, "night": 0.9517285, "transit": 0.532079,
        }
    }

    # Time of the day of the earthquake
    earthquake_time_of_day = "night"

    # Mapping between the names of damage states
    mapping_aux = {
        "dmg_state": ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"],
        "fragility": ["DS0", "DS1", "DS2", "DS3", "DS4"],
    }
    mapping_damage_states = pd.DataFrame(
        mapping_aux, columns=["fragility"], index=mapping_aux["dmg_state"]
    )
    mapping_damage_states.index = mapping_damage_states.index.rename("asset_id")

    # TEST 1
    # Earthquake UTC
    earthquake_datetime = np.datetime64("2010-04-10T00:00:00")

    # Read exposure model
    filepath = os.path.join(os.path.dirname(__file__), "data", "exposure_model.csv")
    exposure_full_occupants = pd.read_csv(filepath)
    exposure_full_occupants.index = exposure_full_occupants["id"]
    exposure_full_occupants.index = exposure_full_occupants.index.rename("asset_id")
    exposure_full_occupants = exposure_full_occupants.drop(columns=["id", "night"])

    # Expected output
    expected_output = os.path.join(
        os.path.dirname(__file__), "data", "exposure_model.csv"
    )
    expected_output = pd.read_csv(filepath)
    expected_output.index = expected_output["id"]
    expected_output.index = expected_output.index.rename("asset_id")
    expected_output = expected_output.drop(columns=["id"])

    # Execute the method
    returned_exposure_updated_occupants = ExposureUpdater.update_exposure_occupants(
        exposure_full_occupants,
        time_of_day_factors,
        earthquake_time_of_day,
        earthquake_datetime,
        mapping_damage_states,
        os.path.join(os.path.dirname(__file__), "data", "intentionally_no_files"),
    )

    assert returned_exposure_updated_occupants.shape[0] == expected_output.shape[0]

    for multiindex in expected_output.index:
        assert multiindex in returned_exposure_updated_occupants.index

        for col in cols_to_check_str:
            assert (
                returned_exposure_updated_occupants.loc[multiindex, col]
                == expected_output.loc[multiindex, col]
            )

        for col in cols_to_check_numeric:
            assert round(returned_exposure_updated_occupants.loc[multiindex, col], 5) == round(
                expected_output.loc[multiindex, col], 5
            )

        for col in cols_to_check_numeric_lower_precision:
            assert round(returned_exposure_updated_occupants.loc[multiindex, col], 2) == round(
                expected_output.loc[multiindex, col], 2
            )

    # TEST 2
    # Earthquake UTC
    earthquake_datetime = np.datetime64("2010-04-10T00:00:00")

    # Read exposure model
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_1.csv"
    )
    exposure_full_occupants = pd.read_csv(filepath)
    exposure_full_occupants.index = exposure_full_occupants["id"]
    exposure_full_occupants.index = exposure_full_occupants.index.rename("asset_id")
    exposure_full_occupants = exposure_full_occupants.drop(columns=["id", "night"])

    # Expected output
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_occupants_update.csv"
    )
    expected_output = pd.read_csv(filepath)
    expected_output.index = expected_output["id"]
    expected_output.index = expected_output.index.rename("asset_id")
    expected_output = expected_output.drop(columns=["id"])

    # Execute the method
    returned_exposure_updated_occupants = ExposureUpdater.update_exposure_occupants(
        exposure_full_occupants,
        time_of_day_factors,
        earthquake_time_of_day,
        earthquake_datetime,
        mapping_damage_states,
        os.path.join(os.path.dirname(__file__), "data"),
    )

    assert returned_exposure_updated_occupants.shape[0] == expected_output.shape[0]

    for multiindex in expected_output.index:
        assert multiindex in returned_exposure_updated_occupants.index

        for col in cols_to_check_str:
            assert (
                returned_exposure_updated_occupants.loc[multiindex, col]
                == expected_output.loc[multiindex, col]
            )

        for col in cols_to_check_numeric:
            assert round(returned_exposure_updated_occupants.loc[multiindex, col], 5) == round(
                expected_output.loc[multiindex, col], 5
            )

        for col in cols_to_check_numeric_lower_precision:
            assert round(returned_exposure_updated_occupants.loc[multiindex, col], 2) == round(
                expected_output.loc[multiindex, col], 2
            )

    # TEST 3
    # Earthquake UTC
    earthquake_datetime = np.datetime64("2009-04-06T01:32:00")

    # Read exposure model
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_1.csv"
    )
    exposure_full_occupants = pd.read_csv(filepath)
    exposure_full_occupants.index = exposure_full_occupants["id"]
    exposure_full_occupants.index = exposure_full_occupants.index.rename("asset_id")
    exposure_full_occupants = exposure_full_occupants.drop(columns=["id", "night"])

    # Expected output (modified manually in code)
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_occupants_update.csv"
    )
    expected_output = pd.read_csv(filepath)
    expected_output.index = expected_output["id"]
    expected_output.index = expected_output.index.rename("asset_id")
    expected_output = expected_output.drop(columns=["id"])
    expected_output["night"] = np.zeros([expected_output.shape[0]])

    # Execute the method
    returned_exposure_updated_occupants = ExposureUpdater.update_exposure_occupants(
        exposure_full_occupants,
        time_of_day_factors,
        earthquake_time_of_day,
        earthquake_datetime,
        mapping_damage_states,
        os.path.join(os.path.dirname(__file__), "data"),
    )

    assert returned_exposure_updated_occupants.shape[0] == expected_output.shape[0]

    for multiindex in expected_output.index:
        assert multiindex in returned_exposure_updated_occupants.index

        for col in cols_to_check_str:
            assert (
                returned_exposure_updated_occupants.loc[multiindex, col]
                == expected_output.loc[multiindex, col]
            )

        for col in cols_to_check_numeric:
            assert round(returned_exposure_updated_occupants.loc[multiindex, col], 5) == round(
                expected_output.loc[multiindex, col], 5
            )

        for col in cols_to_check_numeric_lower_precision:
            assert round(returned_exposure_updated_occupants.loc[multiindex, col], 2) == round(
                expected_output.loc[multiindex, col], 2
            )
