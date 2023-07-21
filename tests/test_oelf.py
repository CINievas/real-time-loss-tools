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
from datetime import datetime
from copy import deepcopy
from realtimelosstools.oelf import OperationalEarthquakeLossForecasting
from realtimelosstools.configuration import Configuration
from realtimelosstools.exposure_updater import ExposureUpdater
from realtimelosstools.stochastic_rupture_generator import StochasticRuptureSet


def test_format_seismicity_forecast():
    filepath = os.path.join(os.path.dirname(__file__), "data", "oef_catalogue.csv")
    forecast = pd.read_csv(filepath)

    # Test case 1: add both event_id and depth
    returned_forecast = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast, add_event_id=True, add_depth=True
    )
    expected_forecast = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_expected_add_both.csv"),
        sep=",",
    )
    expected_forecast["datetime"] = pd.to_datetime(expected_forecast["datetime"])
    expected_forecast.set_index("EQID", drop=True, inplace=True)

    assert len(returned_forecast.index) == len(expected_forecast.index)
    assert len(returned_forecast.columns) == len(expected_forecast.columns)

    for eqid in expected_forecast.index:
        for column in expected_forecast.columns:
            if column in ["longitude", "latitude", "magnitude"]:
                assert round(returned_forecast.loc[eqid, column], 5) == round(
                    expected_forecast.loc[eqid, column], 5
                )
            elif column in ["datetime", "ses_id", "event_id"]:
                assert returned_forecast.loc[eqid, column] == expected_forecast.loc[eqid, column]
            elif column in ["depth"]:
                if np.isnan(expected_forecast.loc[eqid, column]):
                    assert np.isnan(returned_forecast.loc[eqid, column])
                else:
                    assert round(returned_forecast.loc[eqid, column], 5) == round(
                        expected_forecast.loc[eqid, column], 5
                    )

    # Test case 2: add only event_id
    returned_forecast = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast, add_event_id=True, add_depth=False
    )
    expected_forecast = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_expected_add_event_id.csv"),
        sep=",",
    )
    expected_forecast["datetime"] = pd.to_datetime(expected_forecast["datetime"])
    expected_forecast.set_index("EQID", drop=True, inplace=True)

    assert len(returned_forecast.index) == len(expected_forecast.index)
    assert len(returned_forecast.columns) == len(expected_forecast.columns)
    assert "depth" not in returned_forecast.columns

    for eqid in expected_forecast.index:
        for column in expected_forecast.columns:
            if column in ["longitude", "latitude", "magnitude"]:
                assert round(returned_forecast.loc[eqid, column], 5) == round(
                    expected_forecast.loc[eqid, column], 5
                )
            elif column in ["datetime", "ses_id", "event_id"]:
                assert returned_forecast.loc[eqid, column] == expected_forecast.loc[eqid, column]
            elif column in ["depth"]:
                if np.isnan(expected_forecast.loc[eqid, column]):
                    assert np.isnan(returned_forecast.loc[eqid, column])
                else:
                    assert round(returned_forecast.loc[eqid, column], 5) == round(
                        expected_forecast.loc[eqid, column], 5
                    )

    # Test case 3: add only depth
    returned_forecast = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast, add_event_id=False, add_depth=True
    )
    expected_forecast = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_expected_add_depth.csv"),
        sep=",",
    )
    expected_forecast["datetime"] = pd.to_datetime(expected_forecast["datetime"])

    assert len(returned_forecast.index) == len(expected_forecast.index)
    assert len(returned_forecast.columns) == len(expected_forecast.columns)
    assert "event_id" not in returned_forecast.columns

    for row in expected_forecast.index:
        for column in expected_forecast.columns:
            if column in ["longitude", "latitude", "magnitude"]:
                assert round(returned_forecast.loc[row, column], 5) == round(
                    expected_forecast.loc[row, column], 5
                )
            elif column in ["datetime", "ses_id", "event_id"]:
                assert returned_forecast.loc[row, column] == expected_forecast.loc[row, column]
            elif column in ["depth"]:
                if np.isnan(expected_forecast.loc[row, column]):
                    assert np.isnan(returned_forecast.loc[row, column])
                else:
                    assert round(returned_forecast.loc[row, column], 5) == round(
                        expected_forecast.loc[row, column], 5
                    )


def test_filter_forecast():
    filepath = os.path.join(os.path.dirname(__file__), "data", "oef_catalogue.csv")
    input_forecast = pd.read_csv(filepath)
    input_forecast["aux_id"] = [i for i in range(input_forecast.shape[0])]
    input_forecast = input_forecast.rename(columns={
        "Mag": "magnitude", "Lon": "longitude", "Lat": "latitude"
    })

    filepath = os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_filtered.csv")
    expected_filtered_cat = pd.read_csv(filepath)

    expected_kept = np.array(
        [True, False, False, True, False, False, False, False, False, False]
    )

    exposure_lons = np.array([13.400949, 13.3888, 13.400949])
    exposure_lats = np.array([42.344967, 42.344967 ,42.3358])

    magnitude_min = 4.0
    distance_max = 2.5

    returned_filtered_cat, returned_kept = OperationalEarthquakeLossForecasting.filter_forecast(
        input_forecast, exposure_lons, exposure_lats, magnitude_min, distance_max
    )

    assert returned_filtered_cat.shape[0] == expected_filtered_cat.shape[0]

    assert np.all(returned_filtered_cat.index == expected_filtered_cat["aux_id"].to_numpy())

    assert np.all(returned_kept == expected_kept)

    for aux_id in expected_filtered_cat["aux_id"].to_numpy():
        assert aux_id in returned_filtered_cat["aux_id"].to_numpy()


def test_can_there_be_occupants():
    # Read a seismicity catalogue
    filepath = os.path.join(os.path.dirname(__file__), "data", "oef_catalogue.csv")
    forecast_cat = pd.read_csv(filepath)
    forecast_cat = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast_cat, add_event_id=True, add_depth=False
    )
    # Newest date of 'forecast_cat' is 2009-04-07T01:33:02

    # Date of latest "real" earthquake
    date_latest_rla = datetime(2009, 4, 5, 1, 38)  # almost two days earlier

    # Test case in which output should be True
    shortest_recovery_span = 1  # days

    there_can_be_occupants = OperationalEarthquakeLossForecasting.can_there_be_occupants(
        forecast_cat, date_latest_rla, shortest_recovery_span
    )

    assert there_can_be_occupants is True

    # Test case in which output should be False
    shortest_recovery_span = 3  # days

    there_can_be_occupants = OperationalEarthquakeLossForecasting.can_there_be_occupants(
        forecast_cat, date_latest_rla, shortest_recovery_span
    )

    assert there_can_be_occupants is False

    # Test case in which date_latest_rla is None (i.e. no real earthquake has been run)

    there_can_be_occupants = OperationalEarthquakeLossForecasting.can_there_be_occupants(
        forecast_cat, None, shortest_recovery_span
    )

    assert there_can_be_occupants is True


def test_run_oelf_01():
    """Full run of a well-defined OELF calculation (expected inputs and behaviour).
    """

    percent_tolerance = 0.015  # %

    # Copy contents of tests/data/integration_rla to temp directory
    # (the temporary directory will be used to run the test and then erased)
    source_path = os.path.join(os.path.dirname(__file__), "data", "integration_oelf")
    temp_path = os.path.join(os.path.dirname(__file__), "data", "temp_integration_oelf")
    shutil.copytree(
        source_path, temp_path, dirs_exist_ok=False  # If dir exists, raise error
    )

    # Read configuration file
    config_filepath = os.path.join(temp_path, "config_integration_oelf.yml")
    config = Configuration(config_filepath)
    # Override the main path
    config.main_path = deepcopy(temp_path)

    # Create sub-directory to store files associated with number of occupants in time
    path_to_occupants = os.path.join(config.main_path, "current", "occupants")
    os.mkdir(path_to_occupants)

    # Read triggers' file
    triggers = pd.read_csv(os.path.join(config.main_path, "triggering.csv"))

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

    # Smallest number of days to allow people back into buildings
    shortest_recovery_span = recovery_damage["N_damage"].min()  # days

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

    # Read filename of the first OELF trigger
    cat_filename = triggers[triggers.type_analysis == "OELF"]["catalogue_filename"].to_numpy()[0]

    # Read forecast earthquake catalogue
    forecast_cat = pd.read_csv(
        os.path.join(config.main_path, "catalogues", cat_filename)
    )
    forecast_cat = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast_cat, add_event_id=True, add_depth=False
    )  # The index of 'forecast_cat' is the unique ID "[ses_id]-[event_id]"

    # Filter catalogue as per minimum magnitude and maximum distance (so as to not build
    # ruptures for earthquakes that will not be used to calculate damage)
    exposure_lons, exposure_lats = ExposureUpdater.get_unique_exposure_locations(
        exposure_model_undamaged
    )
    forecast_cat_filtered, earthquakes_to_run = (
        OperationalEarthquakeLossForecasting.filter_forecast(
            forecast_cat,
            exposure_lons,
            exposure_lats,
            config.oelf["min_magnitude"],
            config.oelf["max_distance"],
        )
    )
    forecast_cat["to_run"] = earthquakes_to_run

    # Get rid of ".txt", replace ".", "-" and ":" with "_"
    forecast_name = (
        "_".join(cat_filename.split(".")[:-1]).replace("-", "_").replace(":", "_")
    )

    # Create sub-directory to store stochastically-generated rupture XML files
    path_to_ruptures = os.path.join(config.main_path, "ruptures", "oelf", forecast_name)

    # Instantiate the rupture set generator from xml
    stoch_rup = StochasticRuptureSet.from_xml(
        os.path.join(config.main_path, "ruptures", config.oelf_source_model_filename),
        mmin=3.5,  # Minimum magnitude - for calculating total rates
        region_properties=config.oelf["rupture_region_properties"],
        rupture_generator_seed=config.oelf["rupture_generator_seed"]
    )

    # Generate the ruptures for all earthquakes in 'forecast'
    ruptures = stoch_rup.generate_ruptures(
        forecast_cat_filtered,
        path_to_ruptures,  # Ruptures will be exported to this path
        export_type='xml', # Type of file for export
    )

    # Determine if occupants need to be updated (or considered zero), based on the time
    # ellapsed since the last real (RLA) earthquake and the shortest recovery span
    # specified by the user (shortest time to allow occupants back in)
    date_latest_rla = None
    there_can_be_occupants = (
        OperationalEarthquakeLossForecasting.can_there_be_occupants(
            forecast_cat, date_latest_rla, shortest_recovery_span, (59./(3600.*24.))
        )
    )

    # Run OperationalEarthquakeLossForecasting.run_oelf()
    returned_damage_states, returned_losses_economic, returned_losses_human = (
        OperationalEarthquakeLossForecasting.run_oelf(
            forecast_cat,
            forecast_name,
            there_can_be_occupants,
            config.oelf["continuous_ses_numbering"],
            config.oelf["ses_range"],
            config.state_dependent_fragilities,
            config.description_general,
            config.main_path,
            exposure_model_undamaged,
            consequence_economic,
            consequence_injuries,
            recovery_damage,
            recovery_injuries,
            config.injuries_longest_time,
            config.time_of_day_occupancy,
            config.timezone,
            config.mapping_damage_states,
            config.store_intermediate,
            config.store_openquake,
        )
    )

    # CHECKS ASSOCIATED WITH FILES BEING WRITTEN DURING THE PROCESS

    # Check that the correct rupture XML files have been created
    ruptures_should_exist = [
        "RUP_1-1.xml", "RUP_1-2.xml", "RUP_1-3.xml",
        "RUP_3-2.xml", "RUP_3-3.xml",
        "RUP_4-1.xml", "RUP_4-3.xml"
    ]
    for rupt_filename in ruptures_should_exist:
        assert os.path.exists(os.path.join(path_to_ruptures, rupt_filename))
    # Earthquake 3-1 should be filtered out
    assert not os.path.exists(os.path.join(path_to_ruptures, "RUP_3-1.xml"))

    # Check that damage and losses have been output for all SES, including SES 2, even though it
    # does not exist in the input catalogue
    out_filenames = [
        "damage_states_after_OELF_test_oelf_01_realisation_%s.csv",
        "losses_economic_after_OELF_test_oelf_01_realisation_%s.csv",
        "losses_human_after_OELF_test_oelf_01_realisation_%s.csv",
    ]
    for ses in [1, 2, 3, 4]:
        for out_filename in out_filenames:
            assert os.path.exists(
                os.path.join(
                    config.main_path, "output", "test_oelf_01", out_filename % (str(ses))
                )
            )

    # Check that there are no damage or losses associated with SES2
    ses_damage = pd.read_csv(
        os.path.join(
            config.main_path,
            "output",
            "test_oelf_01",
            "damage_states_after_OELF_test_oelf_01_realisation_2.csv"
        )
    )
    assert ses_damage.shape[0] == 1
    assert ses_damage.loc[0, "building_id"] == "tile_1"
    assert ses_damage.loc[0, "damage_state"] == "DS0"
    assert round(ses_damage.loc[0, "number"], 4) == 12.8

    ses_econ_losses = pd.read_csv(
        os.path.join(
            config.main_path,
            "output",
            "test_oelf_01",
            "losses_economic_after_OELF_test_oelf_01_realisation_2.csv"
        )
    )
    assert ses_econ_losses.shape[0] == 1
    assert ses_econ_losses.loc[0, "building_id"] == "tile_1"
    assert round(ses_econ_losses.loc[0, "loss"], 4) == 0.0
    
    ses_human_losses = pd.read_csv(
        os.path.join(
            config.main_path,
            "output",
            "test_oelf_01",
            "losses_human_after_OELF_test_oelf_01_realisation_2.csv"
        )
    )
    assert ses_human_losses.shape[0] == 1
    assert ses_human_losses.loc[0, "building_id"] == "tile_1"
    for level in [1, 2, 3, 4]:
        assert round(ses_human_losses.loc[0, "injuries_%s" % (level)], 4) == 0.0

    # Check that occupants are being placed in buildings only when they should be
    occupants = {
        "exposure_model_after_1-1.csv": (
            "night",
            (
                exposure_model_undamaged["census"].sum()
                * config.time_of_day_occupancy["residential"]["night"]
            )
        ),
        "exposure_model_after_1-2.csv": ("night", 0.0),
        "exposure_model_after_1-3.csv": ("day", 0.0),
        "exposure_model_after_3-2.csv": (
            "transit",
            (
                exposure_model_undamaged["census"].sum()
                * config.time_of_day_occupancy["residential"]["transit"]
            )
        ),
        "exposure_model_after_3-3.csv": ("transit", 0.0),
        "exposure_model_after_4-1.csv": (
            "day",
            (
                exposure_model_undamaged["census"].sum()
                * config.time_of_day_occupancy["residential"]["day"]
            )
        ),
        "exposure_model_after_4-3.csv": ("night", 0.0),
    }

    for exposure_after in occupants:
        exposure_eq = pd.read_csv(
            os.path.join(
                config.main_path,
                "exposure_models",
                "oelf",
                "test_oelf_01",
                exposure_after
            )
        )
        occupants_eq = exposure_eq[occupants[exposure_after][0]].sum()

        if occupants[exposure_after][1] > 1E-8:
            percent_diff = abs(
                (occupants_eq - occupants[exposure_after][1]) / occupants[exposure_after][1]
                * 100.0
            )
            assert percent_diff <= percent_tolerance
        else:
            assert round(occupants_eq, 4) == round(occupants[exposure_after][1], 4)


    # CHECKS ASSOCIATED WITH THE OUTPUT VARIABLES OF THE METHOD

    # Go one by one each result, load expected values and compare
    expected_results_path = os.path.join(
        os.path.dirname(__file__), "data", "integration_oelf_results"
    )

    # Damage states (for the whole seismicity catalogue, average of all SESs)
    expected_damage_states = pd.read_csv(
        os.path.join(expected_results_path, "expected_damage_states_oelf_01.csv")
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
        os.path.join(expected_results_path, "expected_losses_economic_oelf_01.csv"),
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
        os.path.join(expected_results_path, "expected_losses_human_oelf_01.csv"),
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

    shutil.rmtree(temp_path)


def test_run_oelf_02():
    """
    Run of an OELF calculation with an earthquake that will cause OpenQuake to raise an error
    with message "No GMFs were generated, perhaps they were all below the minimum_intensity
    threshold". This will lead the RTLT to output the existing damage (i.e., no additional
    damage due to this earthquake).

    The purpose of this test is to check if OpenQuake changes the way it handles this error.

    The input files for this test are almost the same as for test_run_oelf_01, except that:
    - the minimum OELF magnitude in the configuration file is now 2.0 (so that the RTLT tries
    to run smaller earthquakes),
    - the earthquakes with magnitudes 3.2 and 3.1 (which did not cause damage in
    test_run_oelf_01 because of the minimum OELF magnitude in the configuration file being 3.5)
    were removed (so that they still do not cause damage in this other test),
    - the magnitude of the last earthquake of the first SES was changed to 2.0, so that it leads
    to no ground motion fields being produced by OpenQuake.
    """

    percent_tolerance = 0.015  # %

    # Copy contents of tests/data/integration_rla to temp directory
    # (the temporary directory will be used to run the test and then erased)
    source_path = os.path.join(os.path.dirname(__file__), "data", "integration_oelf_no_GMFs")
    temp_path = os.path.join(os.path.dirname(__file__), "data", "temp_integration_oelf_no_GMFs")
    shutil.copytree(
        source_path, temp_path, dirs_exist_ok=False  # If dir exists, raise error
    )

    # Read configuration file
    config_filepath = os.path.join(temp_path, "config_integration_oelf_no_GMFs.yml")
    config = Configuration(config_filepath)
    # Override the main path
    config.main_path = deepcopy(temp_path)

    # Create sub-directory to store files associated with number of occupants in time
    path_to_occupants = os.path.join(config.main_path, "current", "occupants")
    os.mkdir(path_to_occupants)

    # Read triggers' file
    triggers = pd.read_csv(os.path.join(config.main_path, "triggering.csv"))

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

    # Smallest number of days to allow people back into buildings
    shortest_recovery_span = recovery_damage["N_damage"].min()  # days

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

    # Read filename of the first OELF trigger
    cat_filename = triggers[triggers.type_analysis == "OELF"]["catalogue_filename"].to_numpy()[0]

    # Read forecast earthquake catalogue
    forecast_cat = pd.read_csv(
        os.path.join(config.main_path, "catalogues", cat_filename)
    )
    forecast_cat = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast_cat, add_event_id=True, add_depth=False
    )  # The index of 'forecast_cat' is the unique ID "[ses_id]-[event_id]"

    # Filter catalogue as per minimum magnitude and maximum distance (so as to not build
    # ruptures for earthquakes that will not be used to calculate damage)
    exposure_lons, exposure_lats = ExposureUpdater.get_unique_exposure_locations(
        exposure_model_undamaged
    )
    forecast_cat_filtered, earthquakes_to_run = (
        OperationalEarthquakeLossForecasting.filter_forecast(
            forecast_cat,
            exposure_lons,
            exposure_lats,
            config.oelf["min_magnitude"],
            config.oelf["max_distance"],
        )
    )
    forecast_cat["to_run"] = earthquakes_to_run

    # Get rid of ".txt", replace ".", "-" and ":" with "_"
    forecast_name = (
        "_".join(cat_filename.split(".")[:-1]).replace("-", "_").replace(":", "_")
    )

    # Create sub-directory to store stochastically-generated rupture XML files
    path_to_ruptures = os.path.join(config.main_path, "ruptures", "oelf", forecast_name)

    # Instantiate the rupture set generator from xml
    stoch_rup = StochasticRuptureSet.from_xml(
        os.path.join(config.main_path, "ruptures", config.oelf_source_model_filename),
        mmin=3.5,  # Minimum magnitude - for calculating total rates
        region_properties=config.oelf["rupture_region_properties"],
        rupture_generator_seed=config.oelf["rupture_generator_seed"]
    )

    # Generate the ruptures for all earthquakes in 'forecast'
    ruptures = stoch_rup.generate_ruptures(
        forecast_cat_filtered,
        path_to_ruptures,  # Ruptures will be exported to this path
        export_type='xml', # Type of file for export
    )

    # Determine if occupants need to be updated (or considered zero), based on the time
    # ellapsed since the last real (RLA) earthquake and the shortest recovery span
    # specified by the user (shortest time to allow occupants back in)
    date_latest_rla = None
    there_can_be_occupants = (
        OperationalEarthquakeLossForecasting.can_there_be_occupants(
            forecast_cat, date_latest_rla, shortest_recovery_span, (59./(3600.*24.))
        )
    )

    # Run OperationalEarthquakeLossForecasting.run_oelf()
    returned_damage_states, returned_losses_economic, returned_losses_human = (
        OperationalEarthquakeLossForecasting.run_oelf(
            forecast_cat,
            forecast_name,
            there_can_be_occupants,
            config.oelf["continuous_ses_numbering"],
            config.oelf["ses_range"],
            config.state_dependent_fragilities,
            config.description_general,
            config.main_path,
            exposure_model_undamaged,
            consequence_economic,
            consequence_injuries,
            recovery_damage,
            recovery_injuries,
            config.injuries_longest_time,
            config.time_of_day_occupancy,
            config.timezone,
            config.mapping_damage_states,
            config.store_intermediate,
            config.store_openquake,
        )
    )

    # CHECKS ASSOCIATED WITH THE OUTPUT VARIABLES OF THE METHOD

    # Go one by one each result, load expected values and compare
    expected_results_path = os.path.join(
        os.path.dirname(__file__), "data", "integration_oelf_results"
    )

    # Damage states (for the whole seismicity catalogue, average of all SESs)
    expected_damage_states = pd.read_csv(
        os.path.join(expected_results_path, "expected_damage_states_oelf_01_no_GMFs.csv")
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
        os.path.join(expected_results_path, "expected_losses_economic_oelf_01_no_GMFs.csv"),
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

    # Human casualties (the same as in test_01)
    expected_losses_human = pd.read_csv(
        os.path.join(expected_results_path, "expected_losses_human_oelf_01.csv"),
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

    shutil.rmtree(temp_path)


def test_run_oelf_03():
    """
    Run of an OELF calculation that will cause OpenQuake to raise an error with message "There
    is no damage, perhaps the hazard is too small?". This will lead the RTLT to output the
    existing damage (i.e., no additional damage due to this earthquake, except for still
    retrieving the SHM-damage).

    The purpose of this test is to check if OpenQuake changes the way it handles this error.
    
    The input files for this test are almost the same as for test_run_oelf_01, except that
    the fragility XML file was changed so that all fragilities associated with no pre-existing
    damage (DS0), have a noDamageLimit of 5 g, which clearly will not be achieved by the ground
    motions of any of the earthquakes in the trst. In this way, the ground motion fields are
    generated (otherwise this would become like test_run_oelf_02) but they result in no damage
    output because the buildings are "too strong".
    """

    percent_tolerance = 0.015  # %

    # Copy contents of tests/data/integration_rla to temp directory
    # (the temporary directory will be used to run the test and then erased)
    source_path = os.path.join(os.path.dirname(__file__), "data", "integration_oelf_low_hazard")
    temp_path = os.path.join(os.path.dirname(__file__), "data", "temp_integration_oelf_low_hazard")
    shutil.copytree(
        source_path, temp_path, dirs_exist_ok=False  # If dir exists, raise error
    )

    # Read configuration file
    config_filepath = os.path.join(temp_path, "config_integration_oelf_low_hazard.yml")
    config = Configuration(config_filepath)
    # Override the main path
    config.main_path = deepcopy(temp_path)

    # Create sub-directory to store files associated with number of occupants in time
    path_to_occupants = os.path.join(config.main_path, "current", "occupants")
    os.mkdir(path_to_occupants)

    # Read triggers' file
    triggers = pd.read_csv(os.path.join(config.main_path, "triggering.csv"))

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

    # Smallest number of days to allow people back into buildings
    shortest_recovery_span = recovery_damage["N_damage"].min()  # days

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

    # Read filename of the first OELF trigger
    cat_filename = triggers[triggers.type_analysis == "OELF"]["catalogue_filename"].to_numpy()[0]

    # Read forecast earthquake catalogue
    forecast_cat = pd.read_csv(
        os.path.join(config.main_path, "catalogues", cat_filename)
    )
    forecast_cat = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast_cat, add_event_id=True, add_depth=False
    )  # The index of 'forecast_cat' is the unique ID "[ses_id]-[event_id]"

    # Filter catalogue as per minimum magnitude and maximum distance (so as to not build
    # ruptures for earthquakes that will not be used to calculate damage)
    exposure_lons, exposure_lats = ExposureUpdater.get_unique_exposure_locations(
        exposure_model_undamaged
    )
    forecast_cat_filtered, earthquakes_to_run = (
        OperationalEarthquakeLossForecasting.filter_forecast(
            forecast_cat,
            exposure_lons,
            exposure_lats,
            config.oelf["min_magnitude"],
            config.oelf["max_distance"],
        )
    )
    forecast_cat["to_run"] = earthquakes_to_run

    # Get rid of ".txt", replace ".", "-" and ":" with "_"
    forecast_name = (
        "_".join(cat_filename.split(".")[:-1]).replace("-", "_").replace(":", "_")
    )

    # Create sub-directory to store stochastically-generated rupture XML files
    path_to_ruptures = os.path.join(config.main_path, "ruptures", "oelf", forecast_name)

    # Instantiate the rupture set generator from xml
    stoch_rup = StochasticRuptureSet.from_xml(
        os.path.join(config.main_path, "ruptures", config.oelf_source_model_filename),
        mmin=3.5,  # Minimum magnitude - for calculating total rates
        region_properties=config.oelf["rupture_region_properties"],
        rupture_generator_seed=config.oelf["rupture_generator_seed"]
    )

    # Generate the ruptures for all earthquakes in 'forecast'
    ruptures = stoch_rup.generate_ruptures(
        forecast_cat_filtered,
        path_to_ruptures,  # Ruptures will be exported to this path
        export_type='xml', # Type of file for export
    )

    # Determine if occupants need to be updated (or considered zero), based on the time
    # ellapsed since the last real (RLA) earthquake and the shortest recovery span
    # specified by the user (shortest time to allow occupants back in)
    date_latest_rla = None
    there_can_be_occupants = (
        OperationalEarthquakeLossForecasting.can_there_be_occupants(
            forecast_cat, date_latest_rla, shortest_recovery_span, (59./(3600.*24.))
        )
    )

    # Run OperationalEarthquakeLossForecasting.run_oelf()
    returned_damage_states, returned_losses_economic, returned_losses_human = (
        OperationalEarthquakeLossForecasting.run_oelf(
            forecast_cat,
            forecast_name,
            there_can_be_occupants,
            config.oelf["continuous_ses_numbering"],
            config.oelf["ses_range"],
            config.state_dependent_fragilities,
            config.description_general,
            config.main_path,
            exposure_model_undamaged,
            consequence_economic,
            consequence_injuries,
            recovery_damage,
            recovery_injuries,
            config.injuries_longest_time,
            config.time_of_day_occupancy,
            config.timezone,
            config.mapping_damage_states,
            config.store_intermediate,
            config.store_openquake,
        )
    )

    # CHECKS ASSOCIATED WITH THE OUTPUT VARIABLES OF THE METHOD

    # Go one by one each result, load expected values and compare
    expected_results_path = os.path.join(
        os.path.dirname(__file__), "data", "integration_oelf_results"
    )

    # Damage states (for the whole seismicity catalogue, average of all SESs)
    expected_damage_states = pd.read_csv(
        os.path.join(expected_results_path, "expected_damage_states_oelf_01_hazard_too_low.csv")
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
            expected_results_path, "expected_losses_economic_oelf_01_hazard_too_low.csv"
        ),
        index_col=0
    )
    expected_losses_economic.index.rename("building_id", inplace=True)

    for bdg_id in expected_losses_economic.index:
        assert (
            round(returned_losses_economic.loc[bdg_id, "loss"], 4)
            == round(expected_losses_economic.loc[bdg_id, "loss"], 4)
        )  # losses are zero

    # Human casualties (the same as in test_01)
    expected_losses_human = pd.read_csv(
        os.path.join(expected_results_path, "expected_losses_human_oelf_01_hazard_too_low.csv"),
        index_col=0
    )
    expected_losses_human.index.rename("building_id", inplace=True)

    for injury_level in expected_losses_human.columns:
        for bdg_id in expected_losses_human.index:
            assert (
                round(returned_losses_human.loc[bdg_id, injury_level], 4)
                == round(expected_losses_human.loc[bdg_id, injury_level], 4)
            )  # injuries are all zero

    shutil.rmtree(temp_path)
