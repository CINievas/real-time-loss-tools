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
    injuries_1 = [0.0277947209, 1.6384291018, 0.0096997060]
    injuries_2 = [0.0053323604, 0.3094905813, 0.0014555932]
    injuries_3 = [0.0002597158, 0.0148709595, 0.0000560824]
    injuries_4 = [0.0026484623, 0.1514596842, 0.0005608240]

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
