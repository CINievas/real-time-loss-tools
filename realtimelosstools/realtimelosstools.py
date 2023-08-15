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

import logging
import sys
import os
import shutil
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from realtimelosstools.configuration import Configuration
from realtimelosstools.rla import RapidLossAssessment
from realtimelosstools.ruptures import RLA_Ruptures
from realtimelosstools.oelf import OperationalEarthquakeLossForecasting
from realtimelosstools.stochastic_rupture_generator import StochasticRuptureSet
from realtimelosstools.exposure_updater import ExposureUpdater
from realtimelosstools.losses import Losses
from realtimelosstools.postprocessor import PostProcessor
from realtimelosstools.utils import Files, Loader
from realtimelosstools.writers import Writer


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

log_summary = []


def main():
    """Run the programme."""

    # Log the start of the run
    logger.info("Real-Time Loss Tools has started")

    # Read configuration parameters
    config = Configuration("config.yml")
    if config.state_dependent_fragilities:
        state_dependent_message = "Running with state-dependent fragility models"
    else:
        state_dependent_message = "Running with state-independent fragility models"
    logger.info(state_dependent_message)

    # Log relevant summary parameters (to create log file that allows
    # for a quick check of correct input)
    log_summary.append("Real-Time Loss Tools has started")
    log_summary.append("General description: %s" % (config.description_general))
    log_summary.append(state_dependent_message)
    log_summary.append("%s is path in config file" % (config.main_path))
    log_summary.append("%s is current path" % (os.getcwd()))

    state_dep = Files.find_string_in_file(
        os.path.join(config.main_path, "current", "job.ini"), "state_dependent"
    )
    log_summary.append("State dependent: %s" %(state_dep))

    # If 'exposure_model_current.csv' already exists, code cannot run (the 'main_path' indicated
    # in the configuration file may refer to a directory from a previous run that the user may
    # not want to overwrite)
    if os.path.isfile(
        os.path.join(config.main_path, "current", "exposure_model_current.csv")
    ):
        error_message = (
            "File 'exposure_model_current.csv' already exists under %s. The indicated "
            "directory may have already been used by a previous run. The program will stop."
            % (os.path.join(config.main_path, "current"))
        )
        logger.critical(error_message)
        raise OSError(error_message)

    # Create sub-directory to store files associated with number of occupants in time
    path_to_occupants = os.path.join(config.main_path, "current", "occupants")
    if not os.path.exists(path_to_occupants):
        os.mkdir(path_to_occupants)
    else:
        error_message = (
            "The directory 'occupants' already exists under %s/current and may contain "
            "results from a previous run. The program will stop."
            % (config.main_path)
        )
        logger.critical(error_message)
        raise OSError(error_message)

    # Read input to simulate triggering (calculations after an earthquake of interest and/or
    # at specific points in time, e.g. mid-night)
    triggers = Loader.load_triggers(
        os.path.join(config.main_path, "triggering.csv"),
        os.path.join(config.main_path, "catalogues")
    )
    log_summary.append(
        "First filename in triggering.csv is '%s'" % (triggers.loc[0, "catalogue_filename"])
    )

    # Load data needed for RLA
    if "RLA" in triggers["type_analysis"].to_numpy():
        # Verify/build rupture XML files for RLA
        rla_ruptures = RLA_Ruptures(triggers, config.main_path)

        # Damage results from SHM
        damage_results_SHM = pd.read_csv(
            os.path.join(config.main_path, "shm", "damage_results_shm.csv")
        )
        new_index = pd.MultiIndex.from_arrays(
            [damage_results_SHM["building_id"], damage_results_SHM["dmg_state"]]
        )
        damage_results_SHM.index = new_index
        damage_results_SHM = damage_results_SHM.drop(columns=["dmg_state"])

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

    sum_days = recovery_damage["N_damage"].sum()
    if sum_days < 0.1:
        log_summary.append("No update of occupants in 'recovery_damage'")
    else:
        log_summary.append("With update of occupants in 'recovery_damage'")

    # Smallest number of days to allow people back into buildings
    shortest_recovery_span = recovery_damage["N_damage"].min()  # days

    recovery_injuries = pd.read_csv(
        os.path.join(config.main_path, "static", "recovery_injuries.csv"),
        dtype={"injuries_scale": str, "N_discharged": int},
    )
    recovery_injuries.set_index(recovery_injuries["injuries_scale"], drop=True, inplace=True)
    recovery_injuries = recovery_injuries.drop(columns=["injuries_scale"])

    sum_days = recovery_injuries["N_discharged"].sum()
    if sum_days < 0.1:
        log_summary.append("No update of occupants in 'recovery_injuries'")
    else:
        log_summary.append("With update of occupants in 'recovery_injuries'")

    # Load the "initial" exposure model
    exposure_model_undamaged = pd.read_csv(
            os.path.join(config.main_path, "exposure_models", "exposure_model_undamaged.csv")
        )
    exposure_model_undamaged.index = exposure_model_undamaged["id"]
    exposure_model_undamaged.index = exposure_model_undamaged.index.rename("asset_id")
    exposure_model_undamaged = exposure_model_undamaged.drop(columns=["id"])

    # Check that consequence models cover all the building classes in 'exposure_model_undamaged'
    classes_are_missing, missing_building_classes = Losses.check_consequence_models(
        {
            "economic": consequence_economic,
            "injuries": consequence_injuries,
        },
        exposure_model_undamaged
    )

    if classes_are_missing:
        error_message = (
            "The following building classes are missing from the consequence models: %s"
            % (missing_building_classes)
        )
        logger.critical(error_message)
        raise OSError(error_message)

    # Copy the "initial" exposure model to the 'current' sub-directory to initialise the process
    in_filename = os.path.join(
        config.main_path, "exposure_models", "exposure_model_undamaged.csv"
    )  # origin
    out_filename = os.path.join(config.main_path, "current", "exposure_model_current.csv")
    _ = shutil.copyfile(in_filename, out_filename)

    processed_rla = []
    processed_oelf = []

    date_latest_rla = None

    for i, cat_filename_i in enumerate(triggers["catalogue_filename"].to_numpy()):
        type_analysis_i = triggers["type_analysis"].to_numpy()[i]

        logger.info(
            "%s Running trigger %s of %s: %s with %s"
            % (np.datetime64('now'), i+1, triggers.shape[0], type_analysis_i, cat_filename_i)
        )

        if type_analysis_i == "RLA":
            cat_name = cat_filename_i.split(".")[0]  # Get rid of ".csv"
            processed_rla.append(cat_name)

            # Read earthquake parameters
            earthquake_df = pd.read_csv(
                os.path.join(config.main_path, "catalogues", cat_filename_i)
            )
            earthquake_df["datetime"] = pd.to_datetime(earthquake_df["datetime"])
            earthquake_params = earthquake_df.loc[0, :].to_dict()

            results = RapidLossAssessment.run_rla(
                earthquake_params,
                config.description_general,
                config.main_path,
                rla_ruptures.mapping[cat_filename_i],
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
                exposure_updated,
                damage_states,
                losses_economic,
                losses_human,
                injured_still_away,
                occupancy_factors,
            ) = results

            # Update 'exposure_model_current.csv'
            exposure_updated.to_csv(
                os.path.join(config.main_path, "current", "exposure_model_current.csv"),
                index=False,
            )

            # Store number of injured people away from the building in time, per asset ID
            injured_still_away.to_csv(
                os.path.join(
                    config.main_path,
                    "current",
                    "occupants",
                    "injured_still_away_after_RLA_%s.csv" % (cat_name)
                ),
                index=True,
            )

            # Store occupancy factors (0: people not allowed in, 1: people allowed in) as a
            # function of time and damage state
            occupancy_factors.to_csv(
                os.path.join(
                    config.main_path,
                    "current",
                    "occupants",
                    "occupancy_factors_after_RLA_%s.csv" % (cat_name)
                ),
                index=True,
            )

            # Store damage states per building ID
            damage_states.to_csv(
                os.path.join(
                    config.main_path,
                    "output",
                    "damage_states_after_RLA_%s.csv" % (cat_name)
                ),
                index=True,
            )

            # Store economic losses per building ID
            losses_economic.to_csv(
                os.path.join(
                    config.main_path,
                    "output",
                    "losses_economic_after_RLA_%s.csv" % (cat_name)
                ),
                index=True,
            )

            # Store human losses per building ID
            losses_human.to_csv(
                os.path.join(
                    config.main_path,
                    "output",
                    "losses_human_after_RLA_%s.csv" % (cat_name)
                ),
                index=True,
            )

            # Update 'date_latest_rla'
            date_latest_rla = (earthquake_params["datetime"]).to_pydatetime()

        elif type_analysis_i == "OELF":
            # Read forecast earthquake catalogue
            forecast_cat = pd.read_csv(
                os.path.join(config.main_path, "catalogues", cat_filename_i)
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
            logger.info(
                "%s out of %s earthquakes will be run, all other earthquakes "
                "will be assumed to cause no damage."
                % (earthquakes_to_run.sum(), forecast_cat.shape[0])
            )

            # Get rid of ".txt", replace ".", "-" and ":" with "_"
            forecast_name = (
                "_".join(cat_filename_i.split(".")[:-1]).replace("-", "_").replace(":", "_")
            )
            processed_oelf.append(forecast_name)

            # Create sub-directory to store stochastically-generated rupture XML files
            path_to_ruptures = os.path.join(config.main_path, "ruptures", "oelf", forecast_name)
            if not os.path.exists(path_to_ruptures):
                os.mkdir(path_to_ruptures)
            else:
                error_message = (
                    "The directory %s already exists under %s/ruptures/oelf and may contain "
                    "results from a previous run. The program will stop."
                    % (forecast_name, config.main_path)
                )
                logger.critical(error_message)
                raise OSError(error_message)

            # Instantiate the rupture set generator from xml
            stoch_rup = StochasticRuptureSet.from_xml(
                os.path.join(config.main_path, "ruptures", config.oelf_source_model_filename),
                mmin=4.5,  # Minimum magnitude - for calculating total rates
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
            there_can_be_occupants = (
                OperationalEarthquakeLossForecasting.can_there_be_occupants(
                    forecast_cat, date_latest_rla, shortest_recovery_span, (59./(3600.*24.))
                )
            )
            if there_can_be_occupants:
                logger.info("There might be occupants in buildings during OELF calculation.")
            else:
                logger.info(
                    "Occupants are all zero during OELF calculation "
                    "(too short time since last real earthquake)"
                )

            damage_states, losses_economic, losses_human = (
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

            # Store damage states per building ID
            damage_states.to_csv(
                os.path.join(
                    config.main_path,
                    "output",
                    "damage_states_after_OELF_%s.csv" % (forecast_name)
                ),
                index=True,
            )

            # Store economic losses per building ID
            losses_economic.to_csv(
                os.path.join(
                    config.main_path,
                    "output",
                    "losses_economic_after_OELF_%s.csv" % (forecast_name)
                ),
                index=True,
            )

            # Store human losses per building ID
            losses_human.to_csv(
                os.path.join(
                    config.main_path,
                    "output",
                    "losses_human_after_OELF_%s.csv" % (forecast_name)
                ),
                index=True,
            )

    # Post-process individual outputs
    if config.post_process["collect_csv"]:
        exposure_expected_costs_occupants = Losses.get_expected_costs_occupants(
            exposure_model_undamaged
        )
        PostProcessor.export_collected_output_damage(
            config.main_path, processed_rla, processed_oelf
        )

        PostProcessor.export_collected_output_losses_economic(
            config.main_path, processed_rla, processed_oelf, exposure_expected_costs_occupants
        )

        PostProcessor.export_collected_output_losses_human(
            config.main_path,
            config.injuries_scale,
            processed_rla,
            processed_oelf,
            exposure_expected_costs_occupants,
        )

    # Save 'log_summary' (to create log file that allows for a quick check of correct input)
    log_summary.append("Real-Time Loss Tools has finished")
    Writer.write_txt_from_list(
        log_summary, os.path.join(config.main_path, "quick_input_check.txt")
    )

    # Leave the program
    logger.info("Real-Time Loss Tools has finished")
    sys.exit()


if __name__ == "__main__":
    main()
