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
import pytest
import numpy as np
import pandas as pd
from copy import deepcopy
from realtimelosstools.oq_handler import OpenQuakeJob, OpenQuakeFragility, OpenQuakeUtils


def test_OpenQuakeJob_retrieve_exposure_asset_id_mapping():

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

    returned_mapping = OpenQuakeJob.retrieve_exposure_asset_id_mapping(oq_exposure)

    expected_mapping = {0: "exp_1", 1: "exp_2", 2: "exp_3"}

    for key in expected_mapping:
        assert returned_mapping[key] == expected_mapping[key]


def test_OpenQuakeJob_backward_compatibility_damage():

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

    returned_damage_results_OQ = OpenQuakeJob.backward_compatibility_damage(
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

    returned_damage_results_OQ_weighted = OpenQuakeJob.update_OQ_damage_w_logic_tree_weights(
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

    returned_damage_results_OQ = OpenQuakeJob.ensure_no_negative_damage_results_OQ(
        damage_results_OQ, tolerance=0.0001
    )

    assert returned_damage_results_OQ.shape == expected_damage_results_OQ.shape

    for multiindex in expected_damage_results_OQ.index:
        assert round(returned_damage_results_OQ.loc[multiindex, "value"], 5) == round(
            expected_damage_results_OQ.loc[multiindex, "value"], 5
        )

    # Test case in which the number of rows of the damage input are more than
    # 'n_rows_simplified' and, as a consequence, the simplified procedure is used
    returned_damage_results_OQ = OpenQuakeJob.ensure_no_negative_damage_results_OQ(
        damage_results_OQ, tolerance=0.0001, n_rows_simplified=100
    )

    expected_damage_results_OQ = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "data", "expected_damages_OQ_negative_simplified.csv"
        )
    )
    new_index = pd.MultiIndex.from_arrays(
        [expected_damage_results_OQ["asset_id"], expected_damage_results_OQ["dmg_state"]]
    )
    expected_damage_results_OQ.index = new_index
    expected_damage_results_OQ = expected_damage_results_OQ.drop(
        columns=["asset_id", "dmg_state"]
    )

    assert returned_damage_results_OQ.shape == expected_damage_results_OQ.shape

    for multiindex in expected_damage_results_OQ.index:
        assert round(returned_damage_results_OQ.loc[multiindex, "value"], 5) == round(
            expected_damage_results_OQ.loc[multiindex, "value"], 5
        )

    # Test case in which there is nothing to adjust (input the already adjusted case)
    returned_damage_results_OQ = OpenQuakeJob.ensure_no_negative_damage_results_OQ(
        expected_damage_results_OQ, tolerance=0.0001
    )

    for multiindex in expected_damage_results_OQ.index:
        assert round(returned_damage_results_OQ.loc[multiindex, "value"], 5) == round(
            expected_damage_results_OQ.loc[multiindex, "value"], 5
        )

    # Test case in which the negative values do not comply with the tolerance
    with pytest.raises(ValueError) as excinfo:
        OpenQuakeJob.ensure_no_negative_damage_results_OQ(
            damage_results_OQ, tolerance=0.00001
        )
    assert "ValueError" in str(excinfo.type)

def test_OpenQuakeFragility():
    # From XML path
    frag = OpenQuakeFragility()
    frag.from_xml(
        os.path.join(os.path.dirname(__file__), "data", "fragility_model_diff_IMs.xml")
    )

    expected_ims = ["AvgSA", "SA(0.3)", "SA(0.6)", "AvgSA(0.2)"]

    assert len(expected_ims) == len(frag.ims)

    for imt in expected_ims:
        assert imt in frag.ims

    assert frag.path == os.path.join(
        os.path.dirname(__file__), "data", "fragility_model_diff_IMs.xml"
    )

    # From job.ini
    frag = OpenQuakeFragility()
    frag.from_ini(
        os.path.join(os.path.dirname(__file__), "data", "integration_rla"),
        os.path.join(os.path.dirname(__file__), "data", "integration_rla", "current", "job.ini")
    )

    expected_ims = ["AvgSA"]

    assert len(expected_ims) == len(frag.ims)

    for imt in expected_ims:
        assert imt in frag.ims

    assert frag.path == os.path.join(
        os.path.dirname(__file__), "data", "integration_rla", "static", "fragility_model.xml"
    )

def test_OpenQuakeUtils():
    # TEST 1: Nuisance IMs included in damage IMs, previous calculation has run
    expected_previous_hazard_id = -1

    filepath_nuisance_xml = os.path.join(
        os.path.dirname(__file__), "data", "integration_rla", "static", "fragility_model.xml"
    )
    filepath_damage_job_ini = os.path.join(
        os.path.dirname(__file__), "data", "integration_rla", "current", "job.ini"
    )
    main_path = os.path.join(os.path.dirname(__file__), "data", "integration_rla")
    previous_calc_ran = True

    returned_previous_hazard_id = OpenQuakeUtils.can_previous_hazard_be_reused(
        filepath_nuisance_xml, filepath_damage_job_ini, main_path, previous_calc_ran
    )

    assert returned_previous_hazard_id == expected_previous_hazard_id

    # TEST 2: Nuisance IMs included in damage IMs, previous calculation has not run
    expected_previous_hazard_id = None

    filepath_nuisance_xml = os.path.join(
        os.path.dirname(__file__), "data", "integration_rla", "static", "fragility_model.xml"
    )
    filepath_damage_job_ini = os.path.join(
        os.path.dirname(__file__), "data", "integration_rla", "current", "job.ini"
    )
    main_path = os.path.join(os.path.dirname(__file__), "data", "integration_rla")
    previous_calc_ran = False

    returned_previous_hazard_id = OpenQuakeUtils.can_previous_hazard_be_reused(
        filepath_nuisance_xml, filepath_damage_job_ini, main_path, previous_calc_ran
    )

    assert returned_previous_hazard_id == expected_previous_hazard_id

    # TEST 3: Nuisance IMs not included in damage IMs, previous calculation has run
    expected_previous_hazard_id = None

    filepath_nuisance_xml = os.path.join(
        os.path.join(os.path.dirname(__file__), "data", "fragility_model_diff_IMs.xml")
    )
    filepath_damage_job_ini = os.path.join(
        os.path.dirname(__file__), "data", "integration_rla", "current", "job.ini"
    )
    main_path = os.path.join(os.path.dirname(__file__), "data", "integration_rla")
    previous_calc_ran = True

    returned_previous_hazard_id = OpenQuakeUtils.can_previous_hazard_be_reused(
        filepath_nuisance_xml, filepath_damage_job_ini, main_path, previous_calc_ran
    )

    assert returned_previous_hazard_id == expected_previous_hazard_id

    # TEST 4: Nuisance IMs not included in damage IMs, previous calculation has not run
    expected_previous_hazard_id = None

    filepath_nuisance_xml = os.path.join(
        os.path.join(os.path.dirname(__file__), "data", "fragility_model_diff_IMs.xml")
    )
    filepath_damage_job_ini = os.path.join(
        os.path.dirname(__file__), "data", "integration_rla", "current", "job.ini"
    )
    main_path = os.path.join(os.path.dirname(__file__), "data", "integration_rla")
    previous_calc_ran = False

    returned_previous_hazard_id = OpenQuakeUtils.can_previous_hazard_be_reused(
        filepath_nuisance_xml, filepath_damage_job_ini, main_path, previous_calc_ran
    )

    assert returned_previous_hazard_id == expected_previous_hazard_id


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

    returned_oq_no_damage = OpenQuakeUtils.create_OQ_existing_damage(
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

    returned_oq_no_damage = OpenQuakeUtils.create_OQ_existing_damage(
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


def test_create_OQ_no_nuisance():
    filepath = os.path.join(os.path.dirname(__file__), "data", "exposure_model.csv")
    exposure = pd.read_csv(filepath)
    exposure.index = exposure["id"]
    exposure.index = exposure.index.rename("asset_id")
    exposure = exposure.drop(columns=["id"])

    mapping_nuisance_states = pd.DataFrame(
        {
            "fragility": ["nonfelt", "CDI2", "CDI3", "CDI4", "CDI5", "CDI6"]
        },
        index=["no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4", "dmg_5"]
    )

    returned_oq_no_nuisance = OpenQuakeUtils.create_OQ_no_nuisance(
            exposure,
            mapping_nuisance_states,
            loss_type="structural"
    )

    expected_oq_no_nuisance = pd.DataFrame(
            {
                "asset_id": [
                    "exp_1", "exp_1", "exp_1", "exp_1", "exp_1", "exp_1",
                    "exp_2", "exp_2", "exp_2", "exp_2", "exp_2", "exp_2",
                    "exp_3", "exp_3", "exp_3", "exp_3", "exp_3", "exp_3",
                    "exp_4", "exp_4", "exp_4", "exp_4", "exp_4", "exp_4",
                    "exp_5", "exp_5", "exp_5", "exp_5", "exp_5", "exp_5"
                ],
                "rlz": [
                    0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ],
                "loss_type": [
                    "structural", "structural", "structural", "structural", "structural", "structural",
                    "structural", "structural", "structural", "structural", "structural", "structural",
                    "structural", "structural", "structural", "structural", "structural", "structural",
                    "structural", "structural", "structural", "structural", "structural", "structural",
                    "structural", "structural", "structural", "structural", "structural", "structural",
                ],
                "dmg_state": [
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4", "dmg_5",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4", "dmg_5",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4", "dmg_5",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4", "dmg_5",
                    "no_damage", "dmg_1", "dmg_2", "dmg_3", "dmg_4", "dmg_5"
                ],
                "value": [
                    0.7, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                    90., 0.0, 0.0, 0.0, 0.0, 0.0,
                    10., 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.,  0.0, 0.0, 0.0, 0.0, 0.0
                ],
            },
        index=range(30),
    )

    for row in expected_oq_no_nuisance.index:
        assert row in returned_oq_no_nuisance.index

        for column in expected_oq_no_nuisance.columns:
            assert column in returned_oq_no_nuisance.columns

            if column in ["asset_id", "loss_type", "dmg_state"]:
                assert (
                    returned_oq_no_nuisance.loc[row, column]
                    == expected_oq_no_nuisance.loc[row, column]
                )
            if column in ["rlz", "value"]:
                assert (
                    round(returned_oq_no_nuisance.loc[row, column], 6)
                    == round(expected_oq_no_nuisance.loc[row, column], 6)
                )
