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
        index=["exp_1", "exp_2", "exp_3", "exp_4", "exp_5"],
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


def test_create_mapping_asset_id_to_original_asset_id():
    # Test 1
    filepath = os.path.join(os.path.dirname(__file__), "data", "exposure_model.csv")
    exposure = pd.read_csv(filepath)
    exposure.index = exposure["id"]
    exposure.index = exposure.index.rename("asset_id")
    exposure = exposure.drop(columns=["id"])

    returned_mapping = ExposureUpdater.create_mapping_asset_id_to_original_asset_id(
        exposure
    )

    expected_mapping = pd.DataFrame(
        {
            "original_asset_id": ["exp_1", "exp_2", "exp_3", "exp_4", "exp_5"],
            "number": [0.7, 0.3, 90., 10., 1.]
        },
        index=["exp_1", "exp_2", "exp_3", "exp_4", "exp_5"]
    )
    expected_mapping.index.name = "asset_id"

    assert returned_mapping.index.name == expected_mapping.index.name
    assert len(returned_mapping.index) == len(expected_mapping.index)
    assert len(returned_mapping.columns) == len(expected_mapping.columns)

    for original_asset_id in expected_mapping.index:
        assert original_asset_id in returned_mapping.index
        assert (
            returned_mapping.loc[original_asset_id, "original_asset_id"]
            == expected_mapping.loc[original_asset_id, "original_asset_id"]
        )
        assert (
            round(returned_mapping.loc[original_asset_id, "number"], 6)
            == round(expected_mapping.loc[original_asset_id, "number"], 6)
        )

    # Test 2
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_1.csv"
    )
    exposure = pd.read_csv(filepath)
    exposure.index = exposure["id"]
    exposure.index = exposure.index.rename("asset_id")
    exposure = exposure.drop(columns=["id", "asset_id"])

    returned_mapping = ExposureUpdater.create_mapping_asset_id_to_original_asset_id(
        exposure
    )

    expected_mapping = pd.DataFrame(
        {
            "original_asset_id": [
                "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                "exp_2", "exp_2", "exp_2", "exp_2", "exp_2",
                "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
                "exp_4", "exp_4", "exp_4", "exp_4", "exp_4",
                "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
            ],
            "number": [
                0.17096311, 0.04218159, 0.01607385, 0.06130928, 0.40947217,
                0.06580585, 0.01408029, 0.00908880, 0.02430329, 0.18672177,
                18.61722600, 4.69119550, 1.91062280, 4.89723900, 59.88371700,
                1.80822290, 0.41971886, 0.27867640, 0.48487490, 7.00850700,
                0.3, 0.2, 0.08, 0.02, 0.4,
            ]
        },
        index=["exp_%s" % (i) for i in range(1, 26)]
    )
    expected_mapping.index.name = "asset_id"

    assert returned_mapping.index.name == expected_mapping.index.name
    assert len(returned_mapping.index) == len(expected_mapping.index)
    assert len(returned_mapping.columns) == len(expected_mapping.columns)

    for original_asset_id in expected_mapping.index:
        assert original_asset_id in returned_mapping.index
        assert (
            returned_mapping.loc[original_asset_id, "original_asset_id"]
            == expected_mapping.loc[original_asset_id, "original_asset_id"]
        )
        assert (
            round(returned_mapping.loc[original_asset_id, "number"], 6)
            == round(expected_mapping.loc[original_asset_id, "number"], 6)
        )

    # Test 3
    exposure_add = exposure.loc[["exp_2", "exp_3"], :]
    exposure_add.loc["exp_2", "original_asset_id"] = "something_wrong"
    exposure = pd.concat((exposure, exposure_add))

    with pytest.raises(OSError) as excinfo:
        ExposureUpdater.create_mapping_asset_id_to_original_asset_id(exposure)
    assert "OSError" in str(excinfo.type)


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
        index=["exp_1", "exp_2", "exp_3", "exp_4", "exp_5"],
    )
    id_asset_building_mapping.index = id_asset_building_mapping.index.rename("asset_id")

    # Expected merged damage results
    expected_damage_results_merged = deepcopy(damage_results_OQ)
    for dmg in ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"]:
        expected_damage_results_merged.loc[("exp_5", dmg), "value"] = damage_results_SHM.loc[
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
            "exp_1", "exp_2", "exp_3", "exp_4", "exp_5", "exp_6", "exp_7", "exp_8", "exp_9",
            "exp_10", "exp_11", "exp_12", "exp_13", "exp_14", "exp_15", "exp_16", "exp_17",
            "exp_18", "exp_19", "exp_20", "exp_21", "exp_22", "exp_23", "exp_24", "exp_25",
        ],
    )
    id_asset_building_mapping.index = id_asset_building_mapping.index.rename("asset_id")

    # Expected merged damage results
    expected_damage_results_merged = deepcopy(damage_results_OQ)
    for dmg in ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"]:
        expected_damage_results_merged.loc[("exp_21", dmg), "value"] = (
            damage_results_SHM.loc[("shm_1", dmg), "value"] / 5.0
        )
        expected_damage_results_merged.loc[("exp_22", dmg), "value"] = (
            damage_results_SHM.loc[("shm_1", dmg), "value"] / 5.0
        )
        expected_damage_results_merged.loc[("exp_23", dmg), "value"] = (
            damage_results_SHM.loc[("shm_1", dmg), "value"] / 5.0
        )
        expected_damage_results_merged.loc[("exp_24", dmg), "value"] = (
            damage_results_SHM.loc[("shm_1", dmg), "value"] / 5.0
        )
        expected_damage_results_merged.loc[("exp_25", dmg), "value"] = (
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


def test_get_damage_results_by_orig_asset_id():
    # Damage results from OpenQuake
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_OQ_1.csv")
    damage_results = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results["asset_id"], damage_results["dmg_state"]]
    )
    damage_results.index = new_index
    damage_results = damage_results.drop(columns=["asset_id", "dmg_state"])

    # Mapping between asset_id and original_asset_id
    id_original_asset_building_mapping = pd.DataFrame(
        {"original_asset_id": [
            "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
            "exp_2", "exp_2", "exp_2", "exp_2", "exp_2",
            "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
            "exp_4", "exp_4", "exp_4", "exp_4", "exp_4",
            "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
        ]},
        index=["exp_%s" % (i) for i in range(1, 26)]
    )
    id_original_asset_building_mapping.index.name = "asset_id"

    returned_damage_by_orig_asset_id = (
        ExposureUpdater.get_damage_results_by_orig_asset_id(
            damage_results, id_original_asset_building_mapping
        )
    )

    expected_damage_by_orig_asset_id = pd.DataFrame(
        {
            "value": [
                0.191258291, 0.04211284, 0.0200105, 0.071417303, 0.37520105,
                0.073342333, 0.013780722, 0.010271783, 0.028913818, 0.17369135,
                20.34272390, 4.49482887, 1.987013535, 5.296540463, 57.878895,
                1.95218368, 0.397002856, 0.273416157, 0.538732039, 6.8386655,
                0.317141207, 0.179084241, 0.08580086, 0.045739075, 0.37223464
            ]
        },
        index=pd.MultiIndex.from_arrays(
            [
                id_original_asset_building_mapping["original_asset_id"],
                [
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                ]
            ]
        )
    )
    expected_damage_by_orig_asset_id.index = expected_damage_by_orig_asset_id.index.rename(
        ["original_asset_id", "dmg_state"]
    )

    assert (
        len(
            returned_damage_by_orig_asset_id.index.get_level_values("original_asset_id")
        )
        == len(
            expected_damage_by_orig_asset_id.index.get_level_values("original_asset_id")
        )
    )

    assert (
        len(
            returned_damage_by_orig_asset_id.index.get_level_values("dmg_state")
        )
        == len(
            expected_damage_by_orig_asset_id.index.get_level_values("dmg_state")
        )
    )

    assert returned_damage_by_orig_asset_id.shape[0] == expected_damage_by_orig_asset_id.shape[0]

    for original_asset_id in id_original_asset_building_mapping["original_asset_id"]:
        for dmg_state in ["dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage"]:
            assert (
                round(returned_damage_by_orig_asset_id.loc[(original_asset_id, dmg_state), "value"], 6)
                == round(expected_damage_by_orig_asset_id.loc[(original_asset_id, dmg_state), "value"], 6)
            )


def test_ensure_all_damage_states():
    # Input 'occurrence_by_orig_asset_id'
    occurrence_by_orig_asset_id = pd.DataFrame(
        {
            "value": [
                0.191258291, 0.04211284, 0.0200105, 0.071417303, 0.37520105,
                0.073342333, 0.010271783, 0.028913818, 0.17369135,
                20.34272390, 4.49482887, 1.987013535, 5.296540463, 57.878895,
                1.95218368, 0.397002856, 0.273416157, 0.538732039, 6.8386655,
                0.317141207, 0.179084241, 0.08580086, 0.045739075, 0.37223464
            ]
        },
        index=pd.MultiIndex.from_arrays(
            [
                [
                    "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                    "exp_2", "exp_2", "exp_2", "exp_2",  # one value missing on purpose
                    "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
                    "exp_4", "exp_4", "exp_4", "exp_4", "exp_4",
                    "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
                ],
                [
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_3", "dmg_4", "no_damage",  # dmg_2 missing on purpose
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                ]
            ]
        )
    )
    occurrence_by_orig_asset_id.index = occurrence_by_orig_asset_id.index.rename(
        ["original_asset_id", "dmg_state"]
    )

    # Mapping between the names of damage states
    mapping_aux = {
        "dmg_state": ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"],
        "fragility": ["DS0", "DS1", "DS2", "DS3", "DS4"],
    }
    mapping_damage_states = pd.DataFrame(
        mapping_aux, columns=["fragility"], index=mapping_aux["dmg_state"]
    )
    mapping_damage_states.index = mapping_damage_states.index.rename("asset_id")

    returned_filled = ExposureUpdater.ensure_all_damage_states(
        occurrence_by_orig_asset_id, mapping_damage_states
    )

    expected_filled = pd.DataFrame(
        {
            "value": [
                0.191258291, 0.04211284, 0.0200105, 0.071417303, 0.37520105,
                0.073342333, 0.0, 0.010271783, 0.028913818, 0.17369135,
                20.34272390, 4.49482887, 1.987013535, 5.296540463, 57.878895,
                1.95218368, 0.397002856, 0.273416157, 0.538732039, 6.8386655,
                0.317141207, 0.179084241, 0.08580086, 0.045739075, 0.37223464
            ]
        },
        index=pd.MultiIndex.from_arrays(
            [
                [
                    "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                    "exp_2", "exp_2", "exp_2", "exp_2", "exp_2",
                    "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
                    "exp_4", "exp_4", "exp_4", "exp_4", "exp_4",
                    "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
                ],
                [
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                ]
            ]
        )
    )
    expected_filled.index = expected_filled.index.rename(
        ["original_asset_id", "dmg_state"]
    )

    assert (
        len(returned_filled.index.get_level_values("original_asset_id"))
        == len(expected_filled.index.get_level_values("original_asset_id"))
    )

    assert (
        len(returned_filled.index.get_level_values("dmg_state"))
        == len(expected_filled.index.get_level_values("dmg_state"))
    )

    assert returned_filled.shape[0] == expected_filled.shape[0]

    original_asset_ids = expected_filled.index.get_level_values("original_asset_id").unique()

    for original_asset_id in original_asset_ids:
        for dmg_state in ["dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage"]:
            assert (
                round(returned_filled.loc[(original_asset_id, dmg_state), "value"], 6)
                == round(expected_filled.loc[(original_asset_id, dmg_state), "value"], 6)
            )


def test_get_non_exceedance_by_orig_asset_id():
    # Input 'occurrence_by_orig_asset_id'
    occurrence_by_orig_asset_id = pd.DataFrame(
        {
            "value": [
                0.191258291, 0.04211284, 0.0200105, 0.071417303, 0.37520105,
                0.073342333, 0.013780722, 0.010271783, 0.028913818, 0.17369135,
                20.34272390, 4.49482887, 1.987013535, 5.296540463, 57.878895,
                1.95218368, 0.397002856, 0.273416157, 0.538732039, 6.8386655,
                0.317141207, 0.179084241, 0.08580086, 0.045739075, 0.37223464
            ]
        },
        index=pd.MultiIndex.from_arrays(
            [
                [
                    "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                    "exp_2", "exp_2", "exp_2", "exp_2", "exp_2",
                    "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
                    "exp_4", "exp_4", "exp_4", "exp_4", "exp_4",
                    "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
                ],
                [
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                    "dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage",
                ]
            ],
            names=["original_asset_id", "dmg_state"]
        )
    )

    # Mapping between the names of damage states
    mapping_aux = {
        "dmg_state": ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"],
        "fragility": ["DS0", "DS1", "DS2", "DS3", "DS4"],
    }
    mapping_damage_states = pd.DataFrame(
        mapping_aux, columns=["fragility"], index=mapping_aux["dmg_state"]
    )
    mapping_damage_states.index = mapping_damage_states.index.rename("asset_id")

    returned_prob_non_exceedance = ExposureUpdater.get_non_exceedance_by_orig_asset_id(
        occurrence_by_orig_asset_id, mapping_damage_states
    )

    expected_prob_non_exceedance = pd.DataFrame(
        {
            "prob_non_exceedance": [
                0.0, 0.53600151, 0.80922765, 0.86938885, 0.89797528,
                0.0, 0.57897116, 0.82344559, 0.86938133, 0.90362061,
                0.0, 0.64309882, 0.86912908, 0.91907162, 0.94114955,
                0.0, 0.68386653, 0.87908490, 0.91878518, 0.94612680,
                0.0, 0.37223463, 0.68937583, 0.86846007, 0.95426093,
            ]
        },
        index=pd.MultiIndex.from_arrays(
            [
                [
                    "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                    "exp_2", "exp_2", "exp_2", "exp_2", "exp_2",
                    "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
                    "exp_4", "exp_4", "exp_4", "exp_4", "exp_4",
                    "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
                ],
                [
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                ]
            ],
            names=["original_asset_id", "dmg_state"]
        )
    )

    assert (
        len(returned_prob_non_exceedance.index.get_level_values("original_asset_id"))
        == len(expected_prob_non_exceedance.index.get_level_values("original_asset_id"))
    )

    assert (
        len(returned_prob_non_exceedance.index.get_level_values("dmg_state"))
        == len(expected_prob_non_exceedance.index.get_level_values("dmg_state"))
    )

    assert returned_prob_non_exceedance.shape[0] == expected_prob_non_exceedance.shape[0]

    original_asset_ids = expected_prob_non_exceedance.index.get_level_values(
        "original_asset_id"
    ).unique()

    for original_asset_id in original_asset_ids:
        for dmg_state in ["dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage"]:
            assert (
                round(returned_prob_non_exceedance.loc[
                    (original_asset_id, dmg_state), "prob_non_exceedance"
                ], 6)
                == round(expected_prob_non_exceedance.loc[
                    (original_asset_id, dmg_state), "prob_non_exceedance"
                ], 6)
            )


def test_get_prob_occurrence_from_independent_non_exceedance():
    # prob_non_exceedance_current
    prob_non_exceedance_current = pd.DataFrame(
        {
            "prob_non_exceedance": [
                0.0, 0.53600151, 0.80922765, 0.86938885, 0.89797528,
                0.0, 0.37223463, 0.68937583, 0.86846007, 0.95426093,
            ]
        },
        index=pd.MultiIndex.from_arrays(
            [
                [
                    "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                    "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
                ],
                [
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                ]
            ],
            names=["original_asset_id", "dmg_state"]
        )
    )

    # prob_non_exceedance_previous
    prob_non_exceedance_previous = pd.DataFrame(
        {
            "prob_non_exceedance": [
                0.0, 0.25, 0.50, 0.75, 0.90,
                0.0, 0.35, 0.55, 0.68, 0.92,
            ]
        },
        index=pd.MultiIndex.from_arrays(
            [
                [
                    "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                    "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
                ],
                [
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                ]
            ],
            names=["original_asset_id", "dmg_state"]
        )
    )

    # id_original_asset_building_mapping
    id_original_asset_building_mapping = pd.DataFrame(
        {
            "original_asset_id": [
                "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
            ],
            "number": [
                0.17096311, 0.04218159, 0.01607385, 0.06130928, 0.40947217,
                0.3, 0.2, 0.08, 0.02, 0.4,
            ]
        },
        index=[
                "exp_1", "exp_2", "exp_3", "exp_4", "exp_5",
                "exp_21", "exp_22", "exp_23", "exp_24", "exp_25",
        ]
    )
    id_original_asset_building_mapping.index.name = "asset_id"

    # Mapping between the names of damage states
    mapping_aux = {
        "dmg_state": ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"],
        "fragility": ["DS0", "DS1", "DS2", "DS3", "DS4"],
    }
    mapping_damage_states = pd.DataFrame(
        mapping_aux, columns=["fragility"], index=mapping_aux["dmg_state"]
    )
    mapping_damage_states.index = mapping_damage_states.index.rename("asset_id")

    returned_occurrence_by_orig_asset_id = (
        ExposureUpdater.get_prob_occurrence_from_independent_non_exceedance(
            prob_non_exceedance_previous,
            prob_non_exceedance_current,
            id_original_asset_building_mapping,
            mapping_damage_states,
        )
    )

    expected_occurrence_by_orig_asset_id = pd.DataFrame(
        {
            "prob_occurrence_cumulative": [
                0.13400038, 0.27061345, 0.24742781, 0.15613611, 0.19182225,
                0.13028212, 0.24887459, 0.21139614, 0.28736721, 0.12207995,
            ],
            "number_occurrence_cumulative": [
                0.09380026, 0.18942941, 0.17319947, 0.10929528, 0.13427557,
                0.13028212, 0.24887459, 0.21139614, 0.28736721, 0.12207995,
            ]
        },
        index=pd.MultiIndex.from_arrays(
            [
                [
                    "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                    "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
                ],
                [
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                ]
            ],
            names=["original_asset_id", "dmg_state"]
        )
    )

    assert (
        len(returned_occurrence_by_orig_asset_id.index.get_level_values("original_asset_id"))
        == len(expected_occurrence_by_orig_asset_id.index.get_level_values("original_asset_id"))
    )

    assert (
        len(returned_occurrence_by_orig_asset_id.index.get_level_values("dmg_state"))
        == len(expected_occurrence_by_orig_asset_id.index.get_level_values("dmg_state"))
    )

    assert (
        returned_occurrence_by_orig_asset_id.shape[0]
        == expected_occurrence_by_orig_asset_id.shape[0]
    )

    original_asset_ids = expected_occurrence_by_orig_asset_id.index.get_level_values(
        "original_asset_id"
    ).unique()

    for original_asset_id in original_asset_ids:
        for dmg_state in ["dmg_1", "dmg_2", "dmg_3", "dmg_4", "no_damage"]:
            assert (
                round(returned_occurrence_by_orig_asset_id.loc[
                    (original_asset_id, dmg_state), "prob_occurrence_cumulative"
                ], 6)
                == round(expected_occurrence_by_orig_asset_id.loc[
                    (original_asset_id, dmg_state), "prob_occurrence_cumulative"
                ], 6)
            )
            assert (
                round(returned_occurrence_by_orig_asset_id.loc[
                    (original_asset_id, dmg_state), "number_occurrence_cumulative"
                ], 6)
                == round(expected_occurrence_by_orig_asset_id.loc[
                    (original_asset_id, dmg_state), "number_occurrence_cumulative"
                ], 6)
            )


def test_update_damage_results():
    # damage_results_original (due to one earthquake, state-independent fragilities)
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_OQ_1.csv")
    damage_results_original = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_original["asset_id"], damage_results_original["dmg_state"]]
    )
    damage_results_original.index = new_index
    damage_results_original = damage_results_original.drop(columns=["asset_id", "dmg_state"])

    # damage_occurrence_by_orig_asset_id
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "damage_occurrence_by_orig_asset_id.csv"
    )
    damage_occurrence_by_orig_asset_id = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [
            damage_occurrence_by_orig_asset_id["original_asset_id"],
            damage_occurrence_by_orig_asset_id["dmg_state"]
        ]
    )
    damage_occurrence_by_orig_asset_id.index = new_index
    damage_occurrence_by_orig_asset_id = damage_occurrence_by_orig_asset_id.drop(
        columns=["original_asset_id", "dmg_state"]
    )

    # asset_id_original_asset_id_mapping
    asset_id_original_asset_id_mapping = pd.DataFrame(
        {
            "original_asset_id": [
                "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                "exp_2", "exp_2", "exp_2", "exp_2", "exp_2",
                "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
                "exp_4", "exp_4", "exp_4", "exp_4", "exp_4",
                "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
            ],
            "number": [
                0.17096311, 0.04218159, 0.01607385, 0.06130928, 0.40947217,
                0.06580585, 0.01408029, 0.00908880, 0.02430329, 0.18672177,
                18.61722600, 4.69119550, 1.91062280, 4.89723900, 59.88371700,
                1.80822290, 0.41971886, 0.27867640, 0.48487490, 7.00850700,
                0.3, 0.2, 0.08, 0.02, 0.4,
            ]
        },
        index=["exp_%s" % (i) for i in range(1, 26)]
    )
    asset_id_original_asset_id_mapping.index.name = "asset_id"

    returned_damage_results_updated = ExposureUpdater.update_damage_results(
        damage_results_original,
        damage_occurrence_by_orig_asset_id,
        asset_id_original_asset_id_mapping
    )

    # Expected result
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_damage_results_updated_OQ_1.csv"
    )
    expected_damage_results_updated = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [expected_damage_results_updated["asset_id"],
         expected_damage_results_updated["dmg_state"]]
    )
    expected_damage_results_updated.index = new_index
    expected_damage_results_updated = expected_damage_results_updated.drop(
        columns=["asset_id", "dmg_state"]
    )
    dmg_states = expected_damage_results_updated.index.get_level_values("dmg_state").unique()

    for asset_id in asset_id_original_asset_id_mapping.index:
        for dmg_state in dmg_states:
            assert (
                round(returned_damage_results_updated.loc[(asset_id,  dmg_state), "value"], 6)
                == round(
                    expected_damage_results_updated.loc[(asset_id,  dmg_state), "value"], 6
                )
            )


def test_update_exposure_with_damage_states():
    """
    The test is split into the state-dependent fragilities and the state-independent fragilities
    cases (controled by the boolean input 'state_dependent').

    TEST 1: using state-dependent fragilities
        The test carries out two cycles of update, because the second cycle needs to re-group
        assets but the first cycle does not.

    TEST 2: state-independent fragilities
    """

    # TEST 1
    state_dependent = True

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
    returned_exposure_model_1 = (
        ExposureUpdater.update_exposure_with_damage_states(
            state_dependent,
            initial_exposure,
            initial_exposure,
            damage_results_OQ,
            mapping_damage_states,
            earthquake_time_of_day,
            damage_results_SHM=pd.Series(damage_results_SHM.loc[:, "value"]),
        )
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
    returned_exposure_model_2 = (
        ExposureUpdater.update_exposure_with_damage_states(
            state_dependent,
            initial_exposure_updated,
            initial_exposure,
            damage_results_OQ,
            mapping_damage_states,
            earthquake_time_of_day,
            damage_results_SHM=pd.Series(damage_results_SHM.loc[:, "value"]),
        )
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

    # TEST 2
    state_dependent = False
    # TO BE IMPLEMENTED


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
    expected_damage_results_OQ.loc[("exp_11", "no_damage"), "value"] = 0.0
    expected_damage_results_OQ.loc[("exp_11", "dmg_1"), "value"] = 18.47959741
    expected_damage_results_OQ.loc[("exp_11", "dmg_2"), "value"] = 0.113728624
    expected_damage_results_OQ.loc[("exp_11", "dmg_3"), "value"] = 0.011640353
    expected_damage_results_OQ.loc[("exp_11", "dmg_4"), "value"] = 0.010561345
    expected_damage_results_OQ.loc[("exp_13", "no_damage"), "value"] = 0.0
    expected_damage_results_OQ.loc[("exp_13", "dmg_1"), "value"] = 0.0
    expected_damage_results_OQ.loc[("exp_13", "dmg_2"), "value"] = 0.0
    expected_damage_results_OQ.loc[("exp_13", "dmg_3"), "value"] = 1.588763707
    expected_damage_results_OQ.loc[("exp_13", "dmg_4"), "value"] = 0.321509105

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


def test_update_OQ_damage_w_logic_tree_weights():
    filepath = os.path.join(os.path.dirname(__file__), "data", "damages_OQ_logic_tree.csv")
    damage_results_OQ = pd.read_csv(filepath)

    logic_tree_weights = {
        0: 0.00188042, 1: 0.00749916, 2: 0.00188042,
        3: 0.03708736, 4: 0.14790528, 5: 0.03708736,
        6: 0.08906444, 7: 0.35519111, 8: 0.08906444,
        9: 0.03708736, 10: 0.14790528, 11: 0.03708736,
        12: 0.00188042, 13: 0.00749916, 14: 0.00188042,
    }

    returned_damage_results_OQ_weighted = ExposureUpdater.update_OQ_damage_w_logic_tree_weights(
        damage_results_OQ, logic_tree_weights
    )

    # Expected result
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_damages_OQ_logic_tree_processed.csv"
    )
    expected_damage_results_weighted = pd.read_csv(filepath)
    new_index = pd.MultiIndex.from_arrays(
        [expected_damage_results_weighted["asset_id"],
         expected_damage_results_weighted["dmg_state"]]
    )
    expected_damage_results_weighted.index = new_index
    expected_damage_results_weighted = expected_damage_results_weighted.drop(
        columns=["asset_id", "dmg_state"]
    )
    dmg_states = expected_damage_results_weighted.index.get_level_values("dmg_state").unique()
    asset_ids = expected_damage_results_weighted.index.get_level_values("asset_id").unique()
    #import pdb
    #pdb.set_trace()

    for asset_id in asset_ids:
        for dmg_state in dmg_states:
            assert (
                round(
                    returned_damage_results_OQ_weighted.loc[(asset_id,  dmg_state), "value"], 6
                )
                == round(
                    expected_damage_results_weighted.loc[(asset_id,  dmg_state), "value"], 6
                )
            )


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
        False,
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
        False,
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
        False,
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


def test_create_OQ_existing_damage():
    # Test 1, with initially undamaged exposure model
    filepath = os.path.join(os.path.dirname(__file__), "data", "exposure_model.csv")
    exposure = pd.read_csv(filepath)
    exposure.index = exposure["id"]
    exposure.index = exposure.index.rename("asset_id")
    exposure = exposure.drop(columns=["id"])

    mapping_damage_states = pd.DataFrame(
        {
            "fragility": ["DS0", "DS1", "DS2", "DS3", "DS4"]
        },
        index=["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"]
    )
    mapping_damage_states.index = mapping_damage_states.index.rename("asset_id")

    returned_oq_no_damage = ExposureUpdater.create_OQ_existing_damage(
            exposure,
            mapping_damage_states,
            loss_type="structural"
    )

    expected_oq_no_damage = pd.DataFrame(
            {
                "asset_id": [
                    "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                    "exp_2", "exp_2", "exp_2", "exp_2", "exp_2",
                    "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
                    "exp_4", "exp_4", "exp_4", "exp_4", "exp_4",
                    "exp_5", "exp_5", "exp_5", "exp_5", "exp_5",
                ],
                "rlz": [
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                ],
                "loss_type": [
                    "structural", "structural", "structural", "structural", "structural",
                    "structural", "structural", "structural", "structural", "structural",
                    "structural", "structural", "structural", "structural", "structural",
                    "structural", "structural", "structural", "structural", "structural",
                    "structural", "structural", "structural", "structural", "structural",
                ],
                "dmg_state": [
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                ],
                "value": [
                    0.7, 0.0,  0.0,  0.0,  0.0,
                    0.3, 0.0,  0.0,  0.0,  0.0,
                    90., 0.0,  0.0,  0.0,  0.0,
                    10., 0.0,  0.0,  0.0,  0.0,
                    1., 0.0,  0.0,  0.0,  0.0,
                ],
            },
        index=range(25),
    )

    for row in expected_oq_no_damage.index:
        assert row in returned_oq_no_damage.index

        for column in expected_oq_no_damage.columns:
            assert column in returned_oq_no_damage.columns

            if column in ["asset_id", "loss_type", "dmg_state"]:
                assert (
                    returned_oq_no_damage.loc[row, column]
                    == expected_oq_no_damage.loc[row, column]
                )
            if column in ["rlz", "value"]:
                assert (
                    round(returned_oq_no_damage.loc[row, column], 6)
                    == round(expected_oq_no_damage.loc[row, column], 6)
                )

    # Test 2, with previously damaged exposure model
    exposure.loc["exp_2", "taxonomy"] = exposure.loc["exp_2", "taxonomy"].replace("DS0", "DS2")
    exposure.loc["exp_3", "taxonomy"] = exposure.loc["exp_3", "taxonomy"].replace("DS0", "DS3")

    expected_oq_no_damage["value"] = [
        0.7, 0.0,  0.0,  0.0,  0.0,
        0.0, 0.0,  0.3,  0.0,  0.0,
        0.0, 0.0,  0.0,  90.,  0.0,
        10., 0.0,  0.0,  0.0,  0.0,
        1., 0.0,  0.0,  0.0,  0.0,
    ]

    returned_oq_no_damage = ExposureUpdater.create_OQ_existing_damage(
            exposure,
            mapping_damage_states,
            loss_type="structural"
    )

    for row in expected_oq_no_damage.index:
        assert row in returned_oq_no_damage.index

        for column in expected_oq_no_damage.columns:
            assert column in returned_oq_no_damage.columns

            if column in ["asset_id", "loss_type", "dmg_state"]:
                assert (
                    returned_oq_no_damage.loc[row, column]
                    == expected_oq_no_damage.loc[row, column]
                )
            if column in ["rlz", "value"]:
                assert (
                    round(returned_oq_no_damage.loc[row, column], 6)
                    == round(expected_oq_no_damage.loc[row, column], 6)
                )
