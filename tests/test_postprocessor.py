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
import pandas as pd
from realtimelosstools.postprocessor import PostProcessor


def test_collect_output_losses_economic():
    path = os.path.join(os.path.dirname(__file__), "data")

    filename_pattern = "losses_economic_after_RLA_%s.csv"

    # Test case in which the list of earthquakes is empty
    list_earthquakes = []

    returned_collected_output = PostProcessor._collect_output_losses_economic(
        path, list_earthquakes, filename_pattern
    )

    assert returned_collected_output is None

    # Test case in which the list of earthquakes is not empty
    list_earthquakes = ["EQ_01", "EQ_02"]

    returned_collected_output = PostProcessor._collect_output_losses_economic(
        path, list_earthquakes, filename_pattern
    )

    expected_collected_output = pd.DataFrame(
        {
            "EQ_01": [19143.54, 95131.12, 3335.29],
            "EQ_02": [21702.05, 104658.52, 5651.97],
        },
        index=["building_1", "building_2", "building_3"],
    )
    expected_collected_output.index = expected_collected_output.index.rename("building_id")

    assert returned_collected_output.index.all() == expected_collected_output.index.all()
    assert returned_collected_output.shape == expected_collected_output.shape

    for col in expected_collected_output.columns:
        for bdg_id in expected_collected_output.index:
            assert round(returned_collected_output.loc[bdg_id, col], 2) == round(
                expected_collected_output.loc[bdg_id, col], 2
            )


def test_collect_output_losses_human():
    path = os.path.join(os.path.dirname(__file__), "data")

    injuries_scale = ["1", "2"]

    filename_pattern = "losses_human_after_RLA_%s.csv"

    # Test case in which the list of earthquakes is empty
    list_earthquakes = []

    returned_collected_output = PostProcessor._collect_output_losses_human(
        path, injuries_scale, list_earthquakes, filename_pattern
    )

    assert isinstance(returned_collected_output, dict)
    assert len(returned_collected_output.keys()) == 0

    # Test case in which the list of earthquakes is not empty
    list_earthquakes = ["EQ_01", "EQ_02"]

    returned_collected_output = PostProcessor._collect_output_losses_human(
        path, injuries_scale, list_earthquakes, filename_pattern
    )

    expected_collected_output = {
        "1": pd.DataFrame(
            {
                "EQ_01": [3.4060, 6.2026, 0.2350],
                "EQ_02": [1.4467, 4.6191, 3.8176],
            },
            index=["building_1", "building_2", "building_3"],
        ),
        "2": pd.DataFrame(
            {
                "EQ_01": [0.6752, 1.2185, 0.0455],
                "EQ_02": [0.2832, 0.9121, 0.7438],
            },
            index=["building_1", "building_2", "building_3"],
        ),
    }

    for severity in injuries_scale:
        expected_collected_output[severity].index = expected_collected_output[
            severity
        ].index.rename("building_id")

    for severity in injuries_scale:
        assert severity in returned_collected_output.keys()
        assert (
            returned_collected_output[severity].shape
            == expected_collected_output[severity].shape
        )

        for col in expected_collected_output[severity].columns:
            for bdg_id in expected_collected_output[severity].index:
                assert round(returned_collected_output[severity].loc[bdg_id, col], 4) == round(
                    expected_collected_output[severity].loc[bdg_id, col], 4
                )


def test_collect_output_damage():
    path = os.path.join(os.path.dirname(__file__), "data")

    filename_pattern = "damage_states_after_RLA_%s.csv"

    # Test case in which the list of earthquakes is empty
    list_earthquakes = []

    returned_collected_output = PostProcessor._collect_output_damage(
        path, list_earthquakes, filename_pattern
    )

    assert returned_collected_output is None

    # Test case in which the list of earthquakes is not empty
    list_earthquakes = ["EQ_01", "EQ_02"]

    returned_collected_output = PostProcessor._collect_output_damage(
        path, list_earthquakes, filename_pattern
    )

    expected_collected_output = pd.DataFrame(
        {
            "EQ_01": [
                0.4,
                0.3,
                0.2,
                0.08,
                0.02,
                10,
                5,
                7.5,
                2,
                0.5,
                1.42,
                0.568,
                2.84,
                2.13,
                0.142,
            ],
            "EQ_02": [
                0.27,
                0.2,
                0.4,
                0.1,
                0.03,
                6.75,
                10,
                5,
                2.5,
                0.75,
                0.71,
                1.25,
                2.841,
                1.916,
                0.383,
            ],
        },
    )

    building_id = [
        "building_1", "building_1", "building_1", "building_1", "building_1",
        "building_2", "building_2", "building_2", "building_2", "building_2",
        "building_3", "building_3", "building_3", "building_3", "building_3",
    ]
    damage_state = [
        "DS0", "DS1", "DS2", "DS3", "DS4",
        "DS0", "DS1", "DS2", "DS3", "DS4",
        "DS0", "DS1", "DS2", "DS3", "DS4",
    ]
    new_index = pd.MultiIndex.from_arrays([building_id, damage_state])
    expected_collected_output.index = new_index

    for bdg_id_dmg_state in expected_collected_output.index:
        assert bdg_id_dmg_state in expected_collected_output.index

    assert returned_collected_output.shape == expected_collected_output.shape

    for col in expected_collected_output.columns:
        for bdg_id_dmg_state in expected_collected_output.index:
            assert round(returned_collected_output.loc[bdg_id_dmg_state, col], 3) == round(
                expected_collected_output.loc[bdg_id_dmg_state, col], 3
            )


def test_get_incremental_from_cumulative():
    # Test case in which the list of earthquakes is not empty
    list_earthquakes = ["EQ_01", "EQ_02"]

    cumulative_collected = pd.DataFrame(
        {
            "EQ_01": [19143.54, 95131.12, 3335.29],
            "EQ_02": [21702.05, 104658.52, 5651.97],
        },
        index=["building_1", "building_2", "building_3"],
    )
    cumulative_collected.index = cumulative_collected.index.rename("building_id")

    returned_incremental_output = PostProcessor._get_incremental_from_cumulative(
        cumulative_collected, list_earthquakes
    )

    expected_incremental_output = pd.DataFrame(
        {
            "EQ_01": [19143.54, 95131.12, 3335.29],
            "EQ_02": [2558.51, 9527.4, 2316.68],
        },
        index=["building_1", "building_2", "building_3"],
    )
    expected_incremental_output.index = expected_incremental_output.index.rename("building_id")

    assert returned_incremental_output.index.all() == expected_incremental_output.index.all()
    assert returned_incremental_output.shape == expected_incremental_output.shape

    for col in expected_incremental_output.columns:
        for bdg_id in expected_incremental_output.index:
            assert round(returned_incremental_output.loc[bdg_id, col], 2) == round(
                expected_incremental_output.loc[bdg_id, col], 2
            )

    # Test case in which the list of earthquakes is empty
    list_earthquakes = []

    returned_incremental_output = PostProcessor._get_incremental_from_cumulative(
        cumulative_collected, list_earthquakes
    )

    assert returned_incremental_output is None


def test_get_cumulative_from_incremental():
    # Test case in which the list of earthquakes is not empty
    list_earthquakes = ["EQ_01", "EQ_02", "EQ_03"]

    incremental_collected = pd.DataFrame(
        {
            "EQ_01": [3.4060, 6.2026, 0.2350],
            "EQ_02": [1.4467, 4.6191, 3.8176],
            "EQ_03": [2.5522, 5.1794, 0.9971],
        },
        index=["building_1", "building_2", "building_3"],
    )

    returned_cumulative_output = PostProcessor._get_cumulative_from_incremental(
        incremental_collected, list_earthquakes
    )

    expected_cumulative_output = pd.DataFrame(
        {
            "EQ_01": [3.4060, 6.2026, 0.2350],
            "EQ_02": [4.8527, 10.8217, 4.0526],
            "EQ_03": [7.4049, 16.0011, 5.0497],
        },
        index=["building_1", "building_2", "building_3"],
    )

    assert returned_cumulative_output.index.all() == expected_cumulative_output.index.all()
    assert returned_cumulative_output.shape == expected_cumulative_output.shape

    for col in expected_cumulative_output.columns:
        for bdg_id in expected_cumulative_output.index:
            assert round(returned_cumulative_output.loc[bdg_id, col], 4) == round(
                expected_cumulative_output.loc[bdg_id, col], 4
            )

    # Test case in which the list of earthquakes is empty
    list_earthquakes = []

    returned_cumulative_output = PostProcessor._get_cumulative_from_incremental(
        incremental_collected, list_earthquakes
    )

    assert returned_cumulative_output is None


def test_get_loss_ratio():
    loss_type = "structural"  # economic losses

    absolute_losses = pd.DataFrame(
        {
            "EQ_01": [19143.54, 95131.12, 3335.29],
            "EQ_02": [21702.05, 104658.52, 5651.97],
        },
        index=["building_1", "building_2", "building_3"],
    )
    absolute_losses.index = absolute_losses.index.rename("building_id")

    exposure_costs_occupants = pd.DataFrame(
        {
            "structural": [150000.0, 250000.0, 100000.0],
        },
        index=["building_1", "building_2", "building_3"],
    )
    exposure_costs_occupants.index = exposure_costs_occupants.index.rename("building_id")

    returned_loss_ratios = PostProcessor._get_loss_ratio(absolute_losses, exposure_costs_occupants, loss_type)

    expected_loss_ratios = pd.DataFrame(
        {
            "EQ_01": [12.762360, 38.052448, 3.335290],
            "EQ_02": [14.468033, 41.863408, 5.651970],
        },
        index=["building_1", "building_2", "building_3"],
    )
    expected_loss_ratios.index = expected_loss_ratios.index.rename("building_id")

    assert returned_loss_ratios.index.all() == expected_loss_ratios.index.all()
    assert returned_loss_ratios.shape == expected_loss_ratios.shape

    for col in expected_loss_ratios.columns:
        for bdg_id in expected_loss_ratios.index:
            assert round(returned_loss_ratios.loc[bdg_id, col], 6) == round(
                expected_loss_ratios.loc[bdg_id, col], 6
            )
