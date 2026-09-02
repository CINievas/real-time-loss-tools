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

import numpy as np
import pandas as pd
from realtimelosstools.oq_handler import OpenQuakeDataStoreHandler


def test_OpenQuakeDataStoreHandler_retrieve_exposure_asset_id_mapping():

    # OpenQuake output
    dt = np.dtype([
        ('id', 'S50'),
        ('ordinal', '<u4'),
        ('value-number', '<f4')
    ])
    oq_exposure = np.array([
        (b'exp_1', 0, 8.6),
        (b'exp_2', 1, 4.2),
        (b'exp_3', 2, 1.0)
    ], dtype=dt)

    returned_mapping = OpenQuakeDataStoreHandler.retrieve_exposure_asset_id_mapping(oq_exposure)

    expected_mapping = {0: "exp_1", 1: "exp_2", 2: "exp_3"}

    for key in expected_mapping:
        assert returned_mapping[key] == expected_mapping[key]


def test_OpenQuakeDataStoreHandler_backward_compatibility_damage():

    # OpenQuake output
    dt = np.dtype([
        ('structural-no_damage', '<f4'),
        ('structural-slight', '<f4'),
        ('structural-moderate', '<f4'),
        ('structural-extensive', '<f4'),
        ('structural-complete', '<f4'),
    ])
    dmg_states = np.array([
        (0.18, 0.36, 0.25, 0.14, 0.07),  # asset_id 0
        (1.18, 1.36, 1.25, 1.14, 1.07),  # asset_id 1
        (2.18, 2.36, 2.25, 2.14, 2.07),  # asset_id 2
    ], dtype=dt)
    datastore_damage_results = pd.DataFrame(
        {
            "asset_id": [0, 1, 2],
            "rlz_id": [0, 0, 0],
            "value": [dmg_states[0], dmg_states[1], dmg_states[2]]
        }
    )

    returned_damage_results_OQ = OpenQuakeDataStoreHandler.backward_compatibility_damage(
        datastore_damage_results, {0: "exp_1", 1: "exp_2", 2: "exp_3"}
    )

    expected_damage_results_OQ = pd.DataFrame(
        {
            "asset_id": [
                "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                "exp_2", "exp_2", "exp_2", "exp_2", "exp_2",
                "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
            ],
            "dmg_state": [
                "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4",
                "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4"
            ],
            "rlz": np.zeros([15]),
            "loss_type": ["structural" for i in range(15)],
            "value": [
                0.18, 0.36, 0.25, 0.14, 0.07,
                1.18, 1.36, 1.25, 1.14, 1.07,
                2.18, 2.36, 2.25, 2.14, 2.07
            ]
        }
    )

    assert returned_damage_results_OQ.shape[0] == expected_damage_results_OQ.shape[0]
    assert returned_damage_results_OQ.shape[1] == expected_damage_results_OQ.shape[1]

    for col in expected_damage_results_OQ.columns:
        assert col in returned_damage_results_OQ.columns
        for row in expected_damage_results_OQ.index:
            if col == "value":
                assert (
                    round(float(returned_damage_results_OQ.loc[row, col]), 2) ==
                    round(float(expected_damage_results_OQ.loc[row, col]), 2)
                )
            else:
                assert (
                    returned_damage_results_OQ.loc[row, col] ==
                    expected_damage_results_OQ.loc[row, col]
                )
