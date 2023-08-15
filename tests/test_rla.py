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
from copy import deepcopy
from realtimelosstools.rla import RapidLossAssessment
from realtimelosstools.configuration import Configuration
from realtimelosstools.ruptures import RLA_Ruptures
from realtimelosstools.utils import Loader


def test_run_rla_01():
    """Full run of a well-defined RLA calculation (expected inputs and behaviour).
    """

    percent_tolerance = 0.015  # %

    # Copy contents of tests/data/integration_rla to temp directory
    # (the temporary directory will be used to run the test and then erased)
    source_path = os.path.join(os.path.dirname(__file__), "data", "integration_rla")
    temp_path = os.path.join(os.path.dirname(__file__), "data", "temp_integration_rla")
    shutil.copytree(
        source_path, temp_path, dirs_exist_ok=False  # If dir exists, raise error
    )

    # Read configuration file
    config_filepath = os.path.join(temp_path, "config_integration_rla.yml")
    config = Configuration(config_filepath)
    # Override the main path
    config.main_path = deepcopy(temp_path)

    # Read triggers' file
    triggers = Loader.load_triggers(
        os.path.join(config.main_path, "triggering.csv"),
        os.path.join(config.main_path, "catalogues")
    )

    # Read filename of the first RLA trigger
    cat_filename = triggers[triggers.type_analysis == "RLA"]["catalogue_filename"].to_numpy()[0]

    # Read earthquake parameters
    earthquake_df = pd.read_csv(
        os.path.join(config.main_path, "catalogues", cat_filename)
    )
    earthquake_df["datetime"] = pd.to_datetime(earthquake_df["datetime"])
    earthquake_params = earthquake_df.loc[0, :].to_dict()

    # Verify/build rupture XML files for RLA
    rla_ruptures = RLA_Ruptures(triggers, config.main_path)

    # Load the consequence models
    consequence_economic = pd.read_csv(
        os.path.join(config.main_path, "static", "consequences_economic.csv")
    )
    consequence_economic.set_index(
        consequence_economic["Taxonomy"], drop=True, inplace=True
    )
    consequence_economic = consequence_economic.drop(columns=["Taxonomy"])

    consequence_injuries = {}
    for severity in config.injuries_scale:
        consequence_injuries[severity] = pd.read_csv(
            os.path.join(
                config.main_path, "static", "consequences_injuries_severity_%s.csv" % (severity)
            )
        )
        consequence_injuries[severity].set_index(
            consequence_injuries[severity]["Taxonomy"], drop=True, inplace=True
        )
        consequence_injuries[severity] = consequence_injuries[severity].drop(
            columns=["Taxonomy"]
        )

    # Load the recovery times (used for updating occupants)
    recovery_damage = pd.read_csv(
        os.path.join(config.main_path, "static", "recovery_damage.csv"),
        dtype={"dmg_state": str, "N_inspection": int, "N_repair":int},
    )
    recovery_damage.set_index(recovery_damage["dmg_state"], drop=True, inplace=True)
    recovery_damage = recovery_damage.drop(columns=["dmg_state"])
    recovery_damage["N_damage"] = recovery_damage["N_inspection"] + recovery_damage["N_repair"]

    recovery_injuries = pd.read_csv(
        os.path.join(config.main_path, "static", "recovery_injuries.csv"),
        dtype={"injuries_scale": str, "N_discharged": int},
    )
    recovery_injuries.set_index(recovery_injuries["injuries_scale"], drop=True, inplace=True)
    recovery_injuries = recovery_injuries.drop(columns=["injuries_scale"])

    # Load the "initial" exposure model
    exposure_model_undamaged = pd.read_csv(
            os.path.join(config.main_path, "exposure_models", "exposure_model_undamaged.csv")
        )
    exposure_model_undamaged.index = exposure_model_undamaged["id"]
    exposure_model_undamaged.index = exposure_model_undamaged.index.rename("asset_id")
    exposure_model_undamaged = exposure_model_undamaged.drop(columns=["id"])

    # Copy the "initial" exposure model to the 'current' sub-directory to initialise the process
    in_filename = os.path.join(
        config.main_path, "exposure_models", "exposure_model_undamaged.csv"
    )  # origin
    out_filename = os.path.join(config.main_path, "current", "exposure_model_current.csv")
    _ = shutil.copyfile(in_filename, out_filename)

    # Read damage results from SHM
    damage_results_SHM = pd.read_csv(
        os.path.join(config.main_path, "shm", "damage_results_shm.csv")
    )
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_SHM["building_id"], damage_results_SHM["dmg_state"]]
    )
    damage_results_SHM.index = new_index
    damage_results_SHM = damage_results_SHM.drop(columns=["dmg_state"])

    # Run RapidLossAssessment.run_rla()
    returned_results = RapidLossAssessment.run_rla(
        earthquake_params,
        config.description_general,
        config.main_path,
        rla_ruptures.mapping[cat_filename],
        config.state_dependent_fragilities,
        consequence_economic,
        consequence_injuries,
        recovery_damage,
        recovery_injuries,
        config.injuries_longest_time,
        config.time_of_day_occupancy,
        config.timezone,
        exposure_model_undamaged,
        config.mapping_damage_states,
        damage_results_SHM.loc[:, earthquake_params["event_id"]],
        config.store_intermediate,
        config.store_openquake,
    )
    (
        returned_exposure_updated,
        returned_damage_states,
        returned_losses_economic,
        returned_losses_human,
        returned_injured_still_away,
        returned_occupancy_factors,
    ) = returned_results

    # Go one by one each result, load expected values and compare
    expected_results_path = os.path.join(
        os.path.dirname(__file__), "data", "integration_rla_results"
    )

    # Updated exposure model
    expected_exposure_updated = pd.read_csv(
        os.path.join(expected_results_path, "expected_exposure_updated_rla_01.csv")
    )
    new_index = pd.MultiIndex.from_arrays(
        [expected_exposure_updated["asset_id"],
         expected_exposure_updated["dmg_state"]]
    )
    expected_exposure_updated.index = new_index
    expected_exposure_updated = expected_exposure_updated.drop(
        columns=["asset_id", "dmg_state"]
    )
    asset_ids = expected_exposure_updated.index.get_level_values("asset_id").unique()
    dmg_states = expected_exposure_updated.index.get_level_values("dmg_state").unique()

    string_cols = [
        "id", "taxonomy", "occupancy", "building_id", "original_asset_id",
        "id_3", "name_3", "id_2", "name_2", "id_1", "name_1"
    ]

    for asset_id in asset_ids:
        for dmg_state in dmg_states:
            if asset_id == "exp_3" and dmg_state == "no_damage":
                continue
            for col in ["lon", "lat", "number", "census", "structural"]:
                percent_diff = abs(
                    (
                        returned_exposure_updated.loc[(asset_id,  dmg_state), col]
                        - expected_exposure_updated.loc[(asset_id,  dmg_state), col]
                    ) / expected_exposure_updated.loc[(asset_id,  dmg_state), col]
                    * 100.0
                )

                assert percent_diff <= percent_tolerance

            for col in string_cols:
                assert (
                    returned_exposure_updated.loc[(asset_id,  dmg_state), col]
                    == expected_exposure_updated.loc[(asset_id,  dmg_state), col]
                )

    # Damage states
    expected_damage_states = pd.read_csv(
        os.path.join(expected_results_path, "expected_damage_states_rla_01.csv")
    )

    percent_diff = abs(
        (
            returned_damage_states["number"].sum()
            - expected_damage_states["number"].sum()
        ) / expected_damage_states["number"].sum()
        * 100.0
    )
    assert percent_diff <= percent_tolerance

    for i in range(expected_damage_states.shape[0]):
        expected_bdg_id = expected_damage_states.loc[i, "building_id"]
        expected_dmg_state =  expected_damage_states.loc[i, "damage_state"]
        expected_number =  expected_damage_states.loc[i, "number"]

        percent_diff = abs(
            (
                returned_damage_states.loc[(expected_bdg_id, expected_dmg_state), "number"]
                - expected_number
            ) / expected_number
            * 100.0
        )
        assert percent_diff <= percent_tolerance

    # Economic losses
    expected_losses_economic = pd.read_csv(
        os.path.join(expected_results_path, "expected_losses_economic_rla_01.csv"),
        index_col=0
    )
    expected_losses_economic.index.rename("building_id", inplace=True)

    percent_diff = abs(
        (
            returned_losses_economic["loss"].sum()
            - expected_losses_economic["loss"].sum()
        ) / expected_losses_economic["loss"].sum()
        * 100.0
    )
    assert percent_diff <= percent_tolerance

    for bdg_id in expected_losses_economic.index:
        percent_diff = abs(
            (
                returned_losses_economic.loc[bdg_id, "loss"]
                - expected_losses_economic.loc[bdg_id, "loss"]
            ) / expected_losses_economic.loc[bdg_id, "loss"]
            * 100.0
        )
        assert percent_diff <= percent_tolerance

    # Human casualties
    expected_losses_human = pd.read_csv(
        os.path.join(expected_results_path, "expected_losses_human_rla_01.csv"),
        index_col=0
    )
    expected_losses_human.index.rename("building_id", inplace=True)

    for injury_level in expected_losses_human.columns:
        percent_diff = abs(
            (
                returned_losses_human[injury_level].sum()
                - expected_losses_human[injury_level].sum()
            ) / expected_losses_human[injury_level].sum()
            * 100.0
        )
        assert percent_diff <= percent_tolerance

    for injury_level in expected_losses_human.columns:
        for bdg_id in expected_losses_human.index:
            percent_diff = abs(
                (
                    returned_losses_human.loc[bdg_id, injury_level]
                    - expected_losses_human.loc[bdg_id, injury_level]
                ) / expected_losses_human.loc[bdg_id, injury_level]
                * 100.0
            )
            assert percent_diff <= percent_tolerance

    # Injuries timeline
    expected_injured_still_away = pd.read_csv(
        os.path.join(expected_results_path, "expected_injured_still_away_rla_01.csv"),
        index_col=0
    )
    expected_injured_still_away.index.rename("original_asset_id", inplace=True)

    assert len(returned_injured_still_away.columns) == len(expected_injured_still_away.columns)

    for orig_asset_id in expected_injured_still_away.index:
        for col in expected_injured_still_away.columns:
            if col != "building_id":
                percent_diff = abs(
                    (
                        returned_injured_still_away.loc[orig_asset_id, col]
                        - expected_injured_still_away.loc[orig_asset_id, col]
                    ) / expected_injured_still_away.loc[orig_asset_id, col]
                    * 100.0
                )
                assert percent_diff <= percent_tolerance

        assert (
            expected_injured_still_away.loc[orig_asset_id, "building_id"]
            == returned_injured_still_away.loc[orig_asset_id, "building_id"]
        )

    # Building usability timeline
    expected_occupancy_factors = pd.read_csv(
        os.path.join(expected_results_path, "expected_occupancy_factors_rla_01.csv"),
        index_col=0
    )
    expected_occupancy_factors.index.rename("dmg_state", inplace=True)

    assert len(returned_occupancy_factors.columns) == len(expected_occupancy_factors.columns)

    for dmg_state in expected_occupancy_factors.index:
        for col in expected_occupancy_factors.columns:
            # not using the tolerance because the values are 0 and 1 (0 gives a nan difference)
            assert (
                round(
                    expected_occupancy_factors.loc[dmg_state, col], 6
                ) == round(returned_occupancy_factors.loc[dmg_state, col], 6)
            )

    shutil.rmtree(temp_path)


def test_run_rla_02():
    """
    Run of a RLA calculation that will cause OpenQuake to raise an error with message
    "No GMFs were generated, perhaps they were all below the minimum_intensity threshold".
    This will lead the RTLT to output the existing damage (i.e., no additional damage
    due to this earthquake, except for still retrieving the SHM-damage).

    The purpose of this test is to check if OpenQuake changes the way it handles this error.

    The input files for this test are the same as for test_run_rla_01, except that the magnitude
    of the earthquake was changed to 0.1, so that it leads to no ground motion fields being
    produced by OpenQuake.
    """

    percent_tolerance = 0.015  # %

    # Copy contents of tests/data/integration_rla to temp directory
    # (the temporary directory will be used to run the test and then erased)
    source_path = os.path.join(os.path.dirname(__file__), "data", "integration_rla_no_GMFs")
    temp_path = os.path.join(
        os.path.dirname(__file__), "data", "temp_integration_rla_no_GMFs"
    )
    shutil.copytree(
        source_path, temp_path, dirs_exist_ok=False  # If dir exists, raise error
    )

    # Read configuration file
    config_filepath = os.path.join(temp_path, "config_integration_rla_no_GMFs.yml")
    config = Configuration(config_filepath)
    # Override the main path
    config.main_path = deepcopy(temp_path)

    # Read triggers' file
    triggers = Loader.load_triggers(
        os.path.join(config.main_path, "triggering.csv"),
        os.path.join(config.main_path, "catalogues")
    )

    # Read filename of the first RLA trigger
    cat_filename = triggers[triggers.type_analysis == "RLA"]["catalogue_filename"].to_numpy()[0]

    # Read earthquake parameters
    earthquake_df = pd.read_csv(
        os.path.join(config.main_path, "catalogues", cat_filename)
    )
    earthquake_df["datetime"] = pd.to_datetime(earthquake_df["datetime"])
    earthquake_params = earthquake_df.loc[0, :].to_dict()

    # Verify/build rupture XML files for RLA
    rla_ruptures = RLA_Ruptures(triggers, config.main_path)

    # Load the consequence models
    consequence_economic = pd.read_csv(
        os.path.join(config.main_path, "static", "consequences_economic.csv")
    )
    consequence_economic.set_index(
        consequence_economic["Taxonomy"], drop=True, inplace=True
    )
    consequence_economic = consequence_economic.drop(columns=["Taxonomy"])

    consequence_injuries = {}
    for severity in config.injuries_scale:
        consequence_injuries[severity] = pd.read_csv(
            os.path.join(
                config.main_path, "static", "consequences_injuries_severity_%s.csv" % (severity)
            )
        )
        consequence_injuries[severity].set_index(
            consequence_injuries[severity]["Taxonomy"], drop=True, inplace=True
        )
        consequence_injuries[severity] = consequence_injuries[severity].drop(
            columns=["Taxonomy"]
        )

    # Load the recovery times (used for updating occupants)
    recovery_damage = pd.read_csv(
        os.path.join(config.main_path, "static", "recovery_damage.csv"),
        dtype={"dmg_state": str, "N_inspection": int, "N_repair":int},
    )
    recovery_damage.set_index(recovery_damage["dmg_state"], drop=True, inplace=True)
    recovery_damage = recovery_damage.drop(columns=["dmg_state"])
    recovery_damage["N_damage"] = recovery_damage["N_inspection"] + recovery_damage["N_repair"]

    recovery_injuries = pd.read_csv(
        os.path.join(config.main_path, "static", "recovery_injuries.csv"),
        dtype={"injuries_scale": str, "N_discharged": int},
    )
    recovery_injuries.set_index(recovery_injuries["injuries_scale"], drop=True, inplace=True)
    recovery_injuries = recovery_injuries.drop(columns=["injuries_scale"])

    # Load the "initial" exposure model
    exposure_model_undamaged = pd.read_csv(
            os.path.join(config.main_path, "exposure_models", "exposure_model_undamaged.csv")
        )
    exposure_model_undamaged.index = exposure_model_undamaged["id"]
    exposure_model_undamaged.index = exposure_model_undamaged.index.rename("asset_id")
    exposure_model_undamaged = exposure_model_undamaged.drop(columns=["id"])

    # Copy the "initial" exposure model to the 'current' sub-directory to initialise the process
    in_filename = os.path.join(
        config.main_path, "exposure_models", "exposure_model_undamaged.csv"
    )  # origin
    out_filename = os.path.join(config.main_path, "current", "exposure_model_current.csv")
    _ = shutil.copyfile(in_filename, out_filename)

    # Read damage results from SHM
    damage_results_SHM = pd.read_csv(
        os.path.join(config.main_path, "shm", "damage_results_shm.csv")
    )
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_SHM["building_id"], damage_results_SHM["dmg_state"]]
    )
    damage_results_SHM.index = new_index
    damage_results_SHM = damage_results_SHM.drop(columns=["dmg_state"])

    # Run RapidLossAssessment.run_rla()
    returned_results = RapidLossAssessment.run_rla(
        earthquake_params,
        config.description_general,
        config.main_path,
        rla_ruptures.mapping[cat_filename],
        config.state_dependent_fragilities,
        consequence_economic,
        consequence_injuries,
        recovery_damage,
        recovery_injuries,
        config.injuries_longest_time,
        config.time_of_day_occupancy,
        config.timezone,
        exposure_model_undamaged,
        config.mapping_damage_states,
        damage_results_SHM.loc[:, earthquake_params["event_id"]],
        config.store_intermediate,
        config.store_openquake,
    )
    (
        returned_exposure_updated,
        returned_damage_states,
        returned_losses_economic,
        returned_losses_human,
        returned_injured_still_away,
        _,
    ) = returned_results

    # Go one by one each result, load expected values and compare
    expected_results_path = os.path.join(
        os.path.dirname(__file__), "data", "integration_rla_results"
    )

    # Updated exposure model
    expected_exposure_updated = pd.read_csv(
        os.path.join(
            expected_results_path, "expected_exposure_updated_rla_01_no_addit_damage.csv"
        )
    )
    new_index = pd.MultiIndex.from_arrays(
        [expected_exposure_updated["asset_id"],
         expected_exposure_updated["dmg_state"]]
    )
    expected_exposure_updated.index = new_index
    expected_exposure_updated = expected_exposure_updated.drop(
        columns=["asset_id", "dmg_state"]
    )
    asset_ids = expected_exposure_updated.index.get_level_values("asset_id").unique()
    dmg_states = expected_exposure_updated.index.get_level_values("dmg_state").unique()

    string_cols = [
        "id", "taxonomy", "occupancy", "building_id", "original_asset_id",
        "id_3", "name_3", "id_2", "name_2", "id_1", "name_1"
    ]

    for asset_id in asset_ids:
        for dmg_state in dmg_states:
            if asset_id == "exp_3" and dmg_state == "no_damage":
                continue
            if asset_id == "exp_1" and dmg_state != "no_damage":
                continue
            if asset_id == "exp_2" and dmg_state != "no_damage":
                continue
            for col in ["lon", "lat", "number", "census", "structural"]:
                percent_diff = abs(
                    (
                        returned_exposure_updated.loc[(asset_id,  dmg_state), col]
                        - expected_exposure_updated.loc[(asset_id,  dmg_state), col]
                    ) / expected_exposure_updated.loc[(asset_id,  dmg_state), col]
                    * 100.0
                )

                assert percent_diff <= percent_tolerance

            for col in string_cols:
                assert (
                    returned_exposure_updated.loc[(asset_id,  dmg_state), col]
                    == expected_exposure_updated.loc[(asset_id,  dmg_state), col]
                )

    # Damage states
    expected_damage_states = pd.read_csv(
        os.path.join(expected_results_path, "expected_damage_states_rla_01_no_addit_damage.csv")
    )

    percent_diff = abs(
        (
            returned_damage_states["number"].sum()
            - expected_damage_states["number"].sum()
        ) / expected_damage_states["number"].sum()
        * 100.0
    )
    assert percent_diff <= percent_tolerance

    for i in range(expected_damage_states.shape[0]):
        expected_bdg_id = expected_damage_states.loc[i, "building_id"]
        expected_dmg_state =  expected_damage_states.loc[i, "damage_state"]
        expected_number =  expected_damage_states.loc[i, "number"]

        percent_diff = abs(
            (
                returned_damage_states.loc[(expected_bdg_id, expected_dmg_state), "number"]
                - expected_number
            ) / expected_number
            * 100.0
        )
        assert percent_diff <= percent_tolerance

    # Economic losses
    expected_losses_economic = pd.read_csv(
        os.path.join(
            expected_results_path, "expected_losses_economic_rla_01_no_addit_damage.csv"
        ),
        index_col=0
    )
    expected_losses_economic.index.rename("building_id", inplace=True)

    percent_diff = abs(
        (
            returned_losses_economic["loss"].sum()
            - expected_losses_economic["loss"].sum()
        ) / expected_losses_economic["loss"].sum()
        * 100.0
    )
    assert percent_diff <= percent_tolerance

    for bdg_id in expected_losses_economic.index:
        if expected_losses_economic.loc[bdg_id, "loss"] > 1E-8:
            percent_diff = abs(
                (
                    returned_losses_economic.loc[bdg_id, "loss"]
                    - expected_losses_economic.loc[bdg_id, "loss"]
                ) / expected_losses_economic.loc[bdg_id, "loss"]
                * 100.0
            )
            assert percent_diff <= percent_tolerance
        else:
            assert (
                round(returned_losses_economic.loc[bdg_id, "loss"], 4)
                == round(expected_losses_economic.loc[bdg_id, "loss"], 4)
            )

    # Human casualties
    expected_losses_human = pd.read_csv(
        os.path.join(expected_results_path, "expected_losses_human_rla_01_no_addit_damage.csv"),
        index_col=0
    )
    expected_losses_human.index.rename("building_id", inplace=True)

    for injury_level in expected_losses_human.columns:
        percent_diff = abs(
            (
                returned_losses_human[injury_level].sum()
                - expected_losses_human[injury_level].sum()
            ) / expected_losses_human[injury_level].sum()
            * 100.0
        )
        assert percent_diff <= percent_tolerance

    for injury_level in expected_losses_human.columns:
        for bdg_id in expected_losses_human.index:
            if expected_losses_human.loc[bdg_id, injury_level] > 1E-8:
                percent_diff = abs(
                    (
                        returned_losses_human.loc[bdg_id, injury_level]
                        - expected_losses_human.loc[bdg_id, injury_level]
                    ) / expected_losses_human.loc[bdg_id, injury_level]
                    * 100.0
                )
                assert percent_diff <= percent_tolerance
        else:
            assert (
                round(returned_losses_human.loc[bdg_id, injury_level], 4)
                == round(expected_losses_human.loc[bdg_id, injury_level], 4)
            )

    # Injuries timeline
    expected_injured_still_away = pd.read_csv(
        os.path.join(
            expected_results_path, "expected_injured_still_away_rla_01_no_addit_damage.csv"
        ),
        index_col=0
    )
    expected_injured_still_away.index.rename("original_asset_id", inplace=True)

    assert len(returned_injured_still_away.columns) == len(expected_injured_still_away.columns)

    for orig_asset_id in expected_injured_still_away.index:
        for col in expected_injured_still_away.columns:
            if col != "building_id":
                if expected_injured_still_away.loc[orig_asset_id, col] > 1E-8:
                    percent_diff = abs(
                        (
                            returned_injured_still_away.loc[orig_asset_id, col]
                            - expected_injured_still_away.loc[orig_asset_id, col]
                        ) / expected_injured_still_away.loc[orig_asset_id, col]
                        * 100.0
                    )
                    assert percent_diff <= percent_tolerance
                else:
                    assert (
                        round(returned_injured_still_away.loc[orig_asset_id, col], 4)
                        == round(expected_injured_still_away.loc[orig_asset_id, col], 4)
                    )

        assert (
            expected_injured_still_away.loc[orig_asset_id, "building_id"]
            == returned_injured_still_away.loc[orig_asset_id, "building_id"]
        )

    shutil.rmtree(temp_path)


def test_run_rla_03():
    """
    Run of a RLA calculation that will cause OpenQuake to raise an error with message
    "There is no damage, perhaps the hazard is too small?". This will lead the RTLT to output
    the existing damage (i.e., no additional damage due to this earthquake, except for still
    retrieving the SHM-damage).

    The purpose of this test is to check if OpenQuake changes the way it handles this error.

    The input files for this test are the same as for test_run_rla_01, except that the fragility
    XML file was changed so that all fragilities associated with no pre-existing damage (DS0),
    which are the ones that will be used in the test, have a noDamageLimit of 5 g, which clearly
    will not be achieved by the ground motions. In this way, the ground motion fields are
    generated (otherwise this would become like test_run_rla_02) but they result in no damage
    output because the buildings are "too strong".
    """

    percent_tolerance = 0.015  # %

    # Copy contents of tests/data/integration_rla to temp directory
    # (the temporary directory will be used to run the test and then erased)
    source_path = os.path.join(os.path.dirname(__file__), "data", "integration_rla_low_hazard")
    temp_path = os.path.join(
        os.path.dirname(__file__), "data", "temp_integration_rla_low_hazard"
    )
    shutil.copytree(
        source_path, temp_path, dirs_exist_ok=False  # If dir exists, raise error
    )

    # Read configuration file
    config_filepath = os.path.join(temp_path, "config_integration_rla_low_hazard.yml")
    config = Configuration(config_filepath)
    # Override the main path
    config.main_path = deepcopy(temp_path)

    # Read triggers' file
    triggers = Loader.load_triggers(
        os.path.join(config.main_path, "triggering.csv"),
        os.path.join(config.main_path, "catalogues")
    )

    # Read filename of the first RLA trigger
    cat_filename = triggers[triggers.type_analysis == "RLA"]["catalogue_filename"].to_numpy()[0]

    # Read earthquake parameters
    earthquake_df = pd.read_csv(
        os.path.join(config.main_path, "catalogues", cat_filename)
    )
    earthquake_df["datetime"] = pd.to_datetime(earthquake_df["datetime"])
    earthquake_params = earthquake_df.loc[0, :].to_dict()

    # Verify/build rupture XML files for RLA
    rla_ruptures = RLA_Ruptures(triggers, config.main_path)

    # Load the consequence models
    consequence_economic = pd.read_csv(
        os.path.join(config.main_path, "static", "consequences_economic.csv")
    )
    consequence_economic.set_index(
        consequence_economic["Taxonomy"], drop=True, inplace=True
    )
    consequence_economic = consequence_economic.drop(columns=["Taxonomy"])

    consequence_injuries = {}
    for severity in config.injuries_scale:
        consequence_injuries[severity] = pd.read_csv(
            os.path.join(
                config.main_path, "static", "consequences_injuries_severity_%s.csv" % (severity)
            )
        )
        consequence_injuries[severity].set_index(
            consequence_injuries[severity]["Taxonomy"], drop=True, inplace=True
        )
        consequence_injuries[severity] = consequence_injuries[severity].drop(
            columns=["Taxonomy"]
        )

    # Load the recovery times (used for updating occupants)
    recovery_damage = pd.read_csv(
        os.path.join(config.main_path, "static", "recovery_damage.csv"),
        dtype={"dmg_state": str, "N_inspection": int, "N_repair":int},
    )
    recovery_damage.set_index(recovery_damage["dmg_state"], drop=True, inplace=True)
    recovery_damage = recovery_damage.drop(columns=["dmg_state"])
    recovery_damage["N_damage"] = recovery_damage["N_inspection"] + recovery_damage["N_repair"]

    recovery_injuries = pd.read_csv(
        os.path.join(config.main_path, "static", "recovery_injuries.csv"),
        dtype={"injuries_scale": str, "N_discharged": int},
    )
    recovery_injuries.set_index(recovery_injuries["injuries_scale"], drop=True, inplace=True)
    recovery_injuries = recovery_injuries.drop(columns=["injuries_scale"])

    # Load the "initial" exposure model
    exposure_model_undamaged = pd.read_csv(
            os.path.join(config.main_path, "exposure_models", "exposure_model_undamaged.csv")
        )
    exposure_model_undamaged.index = exposure_model_undamaged["id"]
    exposure_model_undamaged.index = exposure_model_undamaged.index.rename("asset_id")
    exposure_model_undamaged = exposure_model_undamaged.drop(columns=["id"])

    # Copy the "initial" exposure model to the 'current' sub-directory to initialise the process
    in_filename = os.path.join(
        config.main_path, "exposure_models", "exposure_model_undamaged.csv"
    )  # origin
    out_filename = os.path.join(config.main_path, "current", "exposure_model_current.csv")
    _ = shutil.copyfile(in_filename, out_filename)

    # Read damage results from SHM
    damage_results_SHM = pd.read_csv(
        os.path.join(config.main_path, "shm", "damage_results_shm.csv")
    )
    new_index = pd.MultiIndex.from_arrays(
        [damage_results_SHM["building_id"], damage_results_SHM["dmg_state"]]
    )
    damage_results_SHM.index = new_index
    damage_results_SHM = damage_results_SHM.drop(columns=["dmg_state"])

    # Run RapidLossAssessment.run_rla()
    returned_results = RapidLossAssessment.run_rla(
        earthquake_params,
        config.description_general,
        config.main_path,
        rla_ruptures.mapping[cat_filename],
        config.state_dependent_fragilities,
        consequence_economic,
        consequence_injuries,
        recovery_damage,
        recovery_injuries,
        config.injuries_longest_time,
        config.time_of_day_occupancy,
        config.timezone,
        exposure_model_undamaged,
        config.mapping_damage_states,
        damage_results_SHM.loc[:, earthquake_params["event_id"]],
        config.store_intermediate,
        config.store_openquake,
    )
    (
        returned_exposure_updated,
        returned_damage_states,
        returned_losses_economic,
        returned_losses_human,
        returned_injured_still_away,
        _,
    ) = returned_results

    # Go one by one each result, load expected values and compare
    expected_results_path = os.path.join(
        os.path.dirname(__file__), "data", "integration_rla_results"
    )

    # Updated exposure model
    expected_exposure_updated = pd.read_csv(
        os.path.join(
            expected_results_path, "expected_exposure_updated_rla_01_no_addit_damage.csv"
        )
    )
    new_index = pd.MultiIndex.from_arrays(
        [expected_exposure_updated["asset_id"],
         expected_exposure_updated["dmg_state"]]
    )
    expected_exposure_updated.index = new_index
    expected_exposure_updated = expected_exposure_updated.drop(
        columns=["asset_id", "dmg_state"]
    )
    asset_ids = expected_exposure_updated.index.get_level_values("asset_id").unique()
    dmg_states = expected_exposure_updated.index.get_level_values("dmg_state").unique()

    string_cols = [
        "id", "taxonomy", "occupancy", "building_id", "original_asset_id",
        "id_3", "name_3", "id_2", "name_2", "id_1", "name_1"
    ]

    for asset_id in asset_ids:
        for dmg_state in dmg_states:
            if asset_id == "exp_3" and dmg_state == "no_damage":
                continue
            if asset_id == "exp_1" and dmg_state != "no_damage":
                continue
            if asset_id == "exp_2" and dmg_state != "no_damage":
                continue
            for col in ["lon", "lat", "number", "census", "structural"]:
                percent_diff = abs(
                    (
                        returned_exposure_updated.loc[(asset_id,  dmg_state), col]
                        - expected_exposure_updated.loc[(asset_id,  dmg_state), col]
                    ) / expected_exposure_updated.loc[(asset_id,  dmg_state), col]
                    * 100.0
                )

                assert percent_diff <= percent_tolerance

            for col in string_cols:
                assert (
                    returned_exposure_updated.loc[(asset_id,  dmg_state), col]
                    == expected_exposure_updated.loc[(asset_id,  dmg_state), col]
                )

    # Damage states
    expected_damage_states = pd.read_csv(
        os.path.join(expected_results_path, "expected_damage_states_rla_01_no_addit_damage.csv")
    )

    percent_diff = abs(
        (
            returned_damage_states["number"].sum()
            - expected_damage_states["number"].sum()
        ) / expected_damage_states["number"].sum()
        * 100.0
    )
    assert percent_diff <= percent_tolerance

    for i in range(expected_damage_states.shape[0]):
        expected_bdg_id = expected_damage_states.loc[i, "building_id"]
        expected_dmg_state =  expected_damage_states.loc[i, "damage_state"]
        expected_number =  expected_damage_states.loc[i, "number"]

        percent_diff = abs(
            (
                returned_damage_states.loc[(expected_bdg_id, expected_dmg_state), "number"]
                - expected_number
            ) / expected_number
            * 100.0
        )
        assert percent_diff <= percent_tolerance

    # Economic losses
    expected_losses_economic = pd.read_csv(
        os.path.join(
            expected_results_path, "expected_losses_economic_rla_01_no_addit_damage.csv"
        ),
        index_col=0
    )
    expected_losses_economic.index.rename("building_id", inplace=True)

    percent_diff = abs(
        (
            returned_losses_economic["loss"].sum()
            - expected_losses_economic["loss"].sum()
        ) / expected_losses_economic["loss"].sum()
        * 100.0
    )
    assert percent_diff <= percent_tolerance

    for bdg_id in expected_losses_economic.index:
        if expected_losses_economic.loc[bdg_id, "loss"] > 1E-8:
            percent_diff = abs(
                (
                    returned_losses_economic.loc[bdg_id, "loss"]
                    - expected_losses_economic.loc[bdg_id, "loss"]
                ) / expected_losses_economic.loc[bdg_id, "loss"]
                * 100.0
            )
            assert percent_diff <= percent_tolerance
        else:
            assert (
                round(returned_losses_economic.loc[bdg_id, "loss"], 4)
                == round(expected_losses_economic.loc[bdg_id, "loss"], 4)
            )

    # Human casualties
    expected_losses_human = pd.read_csv(
        os.path.join(expected_results_path, "expected_losses_human_rla_01_no_addit_damage.csv"),
        index_col=0
    )
    expected_losses_human.index.rename("building_id", inplace=True)

    for injury_level in expected_losses_human.columns:
        percent_diff = abs(
            (
                returned_losses_human[injury_level].sum()
                - expected_losses_human[injury_level].sum()
            ) / expected_losses_human[injury_level].sum()
            * 100.0
        )
        assert percent_diff <= percent_tolerance

    for injury_level in expected_losses_human.columns:
        for bdg_id in expected_losses_human.index:
            if expected_losses_human.loc[bdg_id, injury_level] > 1E-8:
                percent_diff = abs(
                    (
                        returned_losses_human.loc[bdg_id, injury_level]
                        - expected_losses_human.loc[bdg_id, injury_level]
                    ) / expected_losses_human.loc[bdg_id, injury_level]
                    * 100.0
                )
                assert percent_diff <= percent_tolerance
        else:
            assert (
                round(returned_losses_human.loc[bdg_id, injury_level], 4)
                == round(expected_losses_human.loc[bdg_id, injury_level], 4)
            )

    # Injuries timeline
    expected_injured_still_away = pd.read_csv(
        os.path.join(
            expected_results_path, "expected_injured_still_away_rla_01_no_addit_damage.csv"
        ),
        index_col=0
    )
    expected_injured_still_away.index.rename("original_asset_id", inplace=True)

    assert len(returned_injured_still_away.columns) == len(expected_injured_still_away.columns)

    for orig_asset_id in expected_injured_still_away.index:
        for col in expected_injured_still_away.columns:
            if col != "building_id":
                if expected_injured_still_away.loc[orig_asset_id, col] > 1E-8:
                    percent_diff = abs(
                        (
                            returned_injured_still_away.loc[orig_asset_id, col]
                            - expected_injured_still_away.loc[orig_asset_id, col]
                        ) / expected_injured_still_away.loc[orig_asset_id, col]
                        * 100.0
                    )
                    assert percent_diff <= percent_tolerance
                else:
                    assert (
                        round(returned_injured_still_away.loc[orig_asset_id, col], 4)
                        == round(expected_injured_still_away.loc[orig_asset_id, col], 4)
                    )

        assert (
            expected_injured_still_away.loc[orig_asset_id, "building_id"]
            == returned_injured_still_away.loc[orig_asset_id, "building_id"]
        )

    shutil.rmtree(temp_path)
