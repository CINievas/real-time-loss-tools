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
import numpy as np
import pandas as pd
from realtimelosstools.losses import Losses


def test_expected_economic_loss():
    # Read exposure model
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_2.csv"
    )
    exposure = pd.read_csv(filepath)

    # Read economic consequence model
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "consequences_economic.csv"
    )
    consequence_model = pd.read_csv(filepath)
    consequence_model.set_index(
        consequence_model["Taxonomy"], drop=True, inplace=True
    )
    consequence_model = consequence_model.drop(columns=["Taxonomy"])

    returned_loss_summary = Losses.expected_economic_loss(exposure, consequence_model)

    # Expected output
    building_id = ["osm_1", "tile_8", "shm_1"]
    loss = [43362.325643, 2765141.383750, 31697.200000]

    expected_loss_summary = pd.DataFrame(
        {"loss": loss},
        index=building_id,
    )

    assert len(returned_loss_summary.columns) == len(expected_loss_summary.columns)
    assert len(returned_loss_summary.index) == len(expected_loss_summary.index)

    for index in expected_loss_summary.index:
        assert round(returned_loss_summary.loc[index, "loss"], 5) == round(
            expected_loss_summary.loc[index, "loss"], 5
        )


def test_expected_human_loss_per_asset_id():
    # Read exposure model
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_2.csv"
    )
    exposure = pd.read_csv(filepath)

    # Read human consequence models
    injuries_scale = ["1", "2", "3", "4"]

    consequence_injuries = {}
    for severity in injuries_scale:
        consequence_injuries[severity] = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "consequences_injuries_severity_%s.csv" % (severity),
            )
        )
        consequence_injuries[severity].set_index(
            consequence_injuries[severity]["Taxonomy"], drop=True, inplace=True
        )
        consequence_injuries[severity] = consequence_injuries[severity].drop(
            columns=["Taxonomy"]
        )

    returned_losses_per_asset = Losses.expected_human_loss_per_asset_id(
        exposure, "night", consequence_injuries
    )

    # Expected output
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_injuries_cycle_2.csv"
    )
    expected_losses_per_asset = pd.read_csv(filepath)
    expected_losses_per_asset.set_index(
        expected_losses_per_asset["id"], drop=True, inplace=True
    )
    expected_losses_per_asset = expected_losses_per_asset.drop(columns=["id"])

    assert len(returned_losses_per_asset.columns) == len(expected_losses_per_asset.columns)
    assert len(returned_losses_per_asset.index) == len(expected_losses_per_asset.index)

    for index in expected_losses_per_asset.index:
        for col in ["injuries_1", "injuries_2", "injuries_3", "injuries_4"]:
            assert round(returned_losses_per_asset.loc[index, col], 8) == round(
                expected_losses_per_asset.loc[index, col], 8
            )


def test_expected_human_loss_per_building_id():
    # Read human losses per asset ID
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_injuries_cycle_2.csv"
    )
    human_losses_per_asset = pd.read_csv(filepath)
    human_losses_per_asset.set_index(
        human_losses_per_asset["id"], drop=True, inplace=True
    )
    human_losses_per_asset = human_losses_per_asset.drop(columns=["id"])

    returned_losses_human = Losses.expected_human_loss_per_building_id(human_losses_per_asset)

    # Expected output
    building_id = ["osm_1", "tile_8", "shm_1"]
    injuries_1 = [0.0277947127, 1.6384286247, 0.0096997012]
    injuries_2 = [0.0053323589, 0.3094904906, 0.0014555925]
    injuries_3 = [0.0002597157, 0.0148709547, 0.0000560824]
    injuries_4 = [0.0026484613, 0.1514596335, 0.0005608237]

    expected_losses_human = pd.DataFrame(
        {
            "injuries_1": injuries_1,
            "injuries_2": injuries_2,
            "injuries_3": injuries_3,
            "injuries_4": injuries_4,
        },
        index=building_id,
    )

    assert len(returned_losses_human.columns) == len(expected_losses_human.columns)
    assert len(returned_losses_human.index) == len(expected_losses_human.index)

    for index in expected_losses_human.index:
        for col in ["injuries_1", "injuries_2", "injuries_3", "injuries_4"]:
            assert round(returned_losses_human.loc[index, col], 8) == round(
                expected_losses_human.loc[index, col], 8
            )


def test_assign_zero_human_losses():
    # Read exposure model
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_2.csv"
    )
    exposure = pd.read_csv(filepath)

    # Scale of severity of injuries
    injuries_scale = ["1", "2", "3", "4"]

    returned_zero_loss_summary = Losses.assign_zero_human_losses(exposure, injuries_scale)

    # Expected output
    building_id = ["osm_1", "tile_8", "shm_1"]

    expected_zero_loss_summary = pd.DataFrame(
        {
            "injuries_1": [0.0 for i in range(len(building_id))],
            "injuries_2": [0.0 for i in range(len(building_id))],
            "injuries_3": [0.0 for i in range(len(building_id))],
            "injuries_4": [0.0 for i in range(len(building_id))],
        },
        index=building_id,
    )

    assert len(returned_zero_loss_summary.columns) == len(expected_zero_loss_summary.columns)
    assert len(returned_zero_loss_summary.index) == len(expected_zero_loss_summary.index)

    for index in expected_zero_loss_summary.index:
        for col in ["injuries_1", "injuries_2", "injuries_3", "injuries_4"]:
            assert round(returned_zero_loss_summary.loc[index, col], 8) == round(
                expected_zero_loss_summary.loc[index, col], 8
            )


def test_define_timeline_recovery_relative():
    # Test 1
    timeline_raw = np.array([5, 5, 365, 1095, 1095])
    shortest_time = 0
    longest_time = 730

    expected_timeline = np.array([0, 5, 365, 730])

    returned_timeline = Losses.define_timeline_recovery_relative(
        timeline_raw, shortest_time, longest_time
    )

    np.testing.assert_almost_equal(returned_timeline, expected_timeline, decimal=8)

    # Test 2
    timeline_raw = np.array([5, 5, 365, 1095, 1095])
    shortest_time = 0
    longest_time = 3650

    expected_timeline = np.array([0, 5, 365, 1095, 3650])

    returned_timeline = Losses.define_timeline_recovery_relative(
        timeline_raw, shortest_time, longest_time
    )

    np.testing.assert_almost_equal(returned_timeline, expected_timeline, decimal=8)

    # Test 3
    timeline_raw = np.array([5, 5, 10, 365, 1095, 1095])
    shortest_time = 10
    longest_time = 3650

    expected_timeline = np.array([10, 365, 1095, 3650])

    returned_timeline = Losses.define_timeline_recovery_relative(
        timeline_raw, shortest_time, longest_time
    )

    np.testing.assert_almost_equal(returned_timeline, expected_timeline, decimal=8)


def test_calculate_injuries_recovery_timeline():
    # Define time of the earthquake
    datetime_earthquake = np.datetime64("2009-04-06T01:32:00")

    # Define longest time to calculate future occupants
    longest_time = 36500

    # Human losses per asset ID
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_injuries_cycle_2.csv"
    )
    losses_human_per_asset = pd.read_csv(filepath)
    losses_human_per_asset.set_index(
        losses_human_per_asset["id"], drop=True, inplace=True
    )
    losses_human_per_asset = losses_human_per_asset.drop(columns=["id"])

    # Load the recovery times dependent on health
    recovery_injuries = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "recovery_injuries.csv"),
        dtype={"injuries_scale": str, "N_discharged": int},
    )
    recovery_injuries.set_index(recovery_injuries["injuries_scale"], drop=True, inplace=True)
    recovery_injuries = recovery_injuries.drop(columns=["injuries_scale"])

    # Expected output:
    expected_injured_still_away = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "expected_injured_still_away.csv"),
    )
    expected_injured_still_away.set_index(
        expected_injured_still_away["id"], drop=True, inplace=True
    )
    expected_injured_still_away = expected_injured_still_away.drop(columns=["id"])

    returned_injured_still_away = Losses.calculate_injuries_recovery_timeline(
        losses_human_per_asset,
        recovery_injuries,
        longest_time,
        datetime_earthquake,
    )

    for col in expected_injured_still_away.columns:
        assert col in returned_injured_still_away.columns
    for index in expected_injured_still_away.index:
        assert index in returned_injured_still_away.index

    for col in ["taxonomy", "original_asset_id", "building_id"]:
        for index in expected_injured_still_away.index:
            assert (
                returned_injured_still_away.loc[index, col]
                == expected_injured_still_away.loc[index, col]
            )

    expected_dates = [
        "2009-04-06T01:32:00",
        "2009-04-08T01:32:00",
        "2009-04-16T01:32:00",
        "2109-03-13T01:32:00",
    ]
    for col in expected_dates:
        for index in expected_injured_still_away.index:
            assert (
                round(returned_injured_still_away.loc[index, col], 5)
                == round(expected_injured_still_away.loc[index, col], 5)
            )


def test_calculate_repair_recovery_timeline():
    # Define time of the earthquake
    datetime_earthquake = np.datetime64("2009-04-06T01:32:00")

    # Define longest time to calculate future occupants
    longest_time = 36500

    # Load the recovery times dependent on damage
    recovery_damage = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "recovery_damage.csv"),
        dtype={"dmg_state": str, "N_inspection": int, "N_repair":int},
    )
    recovery_damage.set_index(recovery_damage["dmg_state"], drop=True, inplace=True)
    recovery_damage = recovery_damage.drop(columns=["dmg_state"])
    recovery_damage["N_damage"] = np.maximum(
        recovery_damage["N_inspection"], recovery_damage["N_repair"]
    )

    # Expected output
    expected_occupancy_factors = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "expected_occupancy_factors.csv"),
        dtype={"dmg_state": str},
    )
    expected_occupancy_factors.set_index(
        expected_occupancy_factors["dmg_state"], drop=True, inplace=True
    )
    expected_occupancy_factors = expected_occupancy_factors.drop(columns=["dmg_state"])

    returned_occupancy_factors = Losses.calculate_repair_recovery_timeline(
        recovery_damage,
        longest_time,
        datetime_earthquake,
    )

    for col in expected_occupancy_factors.columns:
        assert col in returned_occupancy_factors.columns
    for index in expected_occupancy_factors.index:
        assert index in returned_occupancy_factors.index

    for col in expected_occupancy_factors.columns:
        for index in expected_occupancy_factors.index:
            assert (
                returned_occupancy_factors.loc[index, col]
                == expected_occupancy_factors.loc[index, col]
            )


def test_get_occupancy_factors():
    # Test 1: previous earthquakes run
    datetime_earthquake = np.datetime64("2010-04-10T00:00:00")

    aux = {
        "dmg_state": ["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"],
        "fragility": ["DS0", "DS1", "DS2", "DS3", "DS4"]
    }
    mapping_damage_states = pd.DataFrame(aux)
    mapping_damage_states.set_index(mapping_damage_states["dmg_state"], drop=True, inplace=True)
    mapping_damage_states = mapping_damage_states.drop(columns=["dmg_state"])

    main_path = os.path.join(os.path.dirname(__file__), "data")

    expected_occupancy_factors = {"DS0": 1, "DS1": 1, "DS2": 1, "DS3": 0, "DS4": 0}

    returned_occupancy_factors = Losses.get_occupancy_factors(
        datetime_earthquake, mapping_damage_states, main_path
    )

    for dmg_state in expected_occupancy_factors:
        assert returned_occupancy_factors[dmg_state] == expected_occupancy_factors[dmg_state]

    # Test 2: no previous earthquakes run
    main_path = os.path.join(os.path.dirname(__file__), "data", "intentionally_no_files")

    expected_occupancy_factors = {"DS0": 1, "DS1": 1, "DS2": 1, "DS3": 1, "DS4": 1}

    returned_occupancy_factors = Losses.get_occupancy_factors(
        datetime_earthquake, mapping_damage_states, main_path
    )

    for dmg_state in expected_occupancy_factors:
        assert returned_occupancy_factors[dmg_state] == expected_occupancy_factors[dmg_state]


def test_get_occupancy_factors_per_asset():
    exposure_taxonomies = np.array([
        "CR/LFINF+CDN+LFC:0.0/H:1/DS1",
        "CR/LFINF+CDN+LFC:0.0/H:1/DS2",
        "CR/LFINF+CDN+LFC:0.0/H:1/DS3",
        "CR/LFINF+CDN+LFC:0.0/H:1/DS4",
        "CR/LFINF+CDN+LFC:0.0/H:1/DS0",
    ])

    occupancy_factors = {"DS0": 1, "DS1": 1, "DS2": 1, "DS3": 0, "DS4": 0}

    expected_occupancy_factors_per_asset = np.array([1, 1, 0, 0, 1])

    returned_occupancy_factors_per_asset = Losses.get_occupancy_factors_per_asset(
        exposure_taxonomies, occupancy_factors
    )

    np.testing.assert_almost_equal(
        returned_occupancy_factors_per_asset, expected_occupancy_factors_per_asset, decimal=8
    )


def test_get_injured_still_away():
    # Test 1: previous earthquakes run
    datetime_earthquake = np.datetime64("2010-04-10T00:00:00")
    main_path = os.path.join(os.path.dirname(__file__), "data")
    exposure_indices = np.array(["res_%s" % (i) for i in range(1, 26)])

    expected_injured_still_away = np.array([
        0.0, 0.0, 0.0000013278, 0.0037317946, 0.0,
        0.0, 0.0, 0.0000009988, 0.0014675546, 0.0,
        0.0, 0.0, 0.0000010856, 0.0022090366, 0.0,
        0.0, 0.0, 0.0000008462, 0.0008446804, 0.0,
        0.0, 0.0, 0.0000101968, 0.0046522900, 0.0,
    ])

    returned_injured_still_away = Losses.get_injured_still_away(
        exposure_indices, datetime_earthquake, main_path
    )

    np.testing.assert_almost_equal(
        returned_injured_still_away, expected_injured_still_away, decimal=8
    )

    # Test 2: no previous earthquakes run
    main_path = os.path.join(os.path.dirname(__file__), "data", "intentionally_no_files")

    expected_injured_still_away = np.zeros([len(exposure_indices)])

    returned_injured_still_away = Losses.get_injured_still_away(
        exposure_indices, datetime_earthquake, main_path
    )

    np.testing.assert_almost_equal(
        returned_injured_still_away, expected_injured_still_away, decimal=8
    )


def test_get_time_of_day_factors_per_asset():
    exposure_occupancies = np.array(["commercial", "residential", "residential", "commercial"])
    earthquake_time_of_day = "transit"
    time_of_day_factors = {
        "residential": {
            "day": 0.242853, "night": 0.9517285, "transit": 0.532079,
        },
        "commercial": {
            "day": 0.4982155, "night": 0.0436495, "transit": 0.090751,
        }
    }

    expected_time_of_day_factors_per_asset = np.array([0.090751, 0.532079, 0.532079, 0.090751])

    returned_time_of_day_factors_per_asset = Losses.get_time_of_day_factors_per_asset(
        exposure_occupancies, earthquake_time_of_day, time_of_day_factors
    )

    np.testing.assert_almost_equal(
        returned_time_of_day_factors_per_asset,
        expected_time_of_day_factors_per_asset,
        decimal=6
    )
