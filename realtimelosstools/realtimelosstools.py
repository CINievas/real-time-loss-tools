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
import pandas as pd
from realtimelosstools.configuration import Configuration
from realtimelosstools.rla import RapidLossAssessment
from realtimelosstools.oelf import OperationalEarthquakeLossForecasting
from realtimelosstools.stochastic_rupture_generator import StochasticRuptureSet
from realtimelosstools.exposure_updater import ExposureUpdater


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def main():
    """Run the programme."""

    # Log the start of the run
    logger.info("Real-Time Loss Tools has started")

    # Read configuration parameters
    config = Configuration("config.yml")

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

    # Read input to simulate triggering (calculations after an earthquake of interest and/or
    # at specific points in time, e.g. mid-night)
    triggers = pd.read_csv(os.path.join(config.main_path, "triggering.csv"))

    # Check that 'triggers' only refers to RLA or OELF, abort otherwise
    for type_analysis in triggers["type_analysis"].to_numpy():
        if type_analysis != "RLA" and type_analysis != "OELF":
            error_message = (
                "Type of analysis '%s' specified in triggering.csv is unknown. "
                "The software cannot run."
                % (type_analysis)
            )
            logger.critical(error_message)
            raise ValueError(error_message)

    # Load data needed for RLA
    if "RLA" in triggers["type_analysis"].to_numpy():
        # Source parameters
        source_parameters_RLA = pd.read_csv(
            os.path.join(config.main_path, "ruptures", "rla", "source_parameters.csv")
        )
        source_parameters_RLA.index = source_parameters_RLA["event_id"]

        # Damage results from SHM
        damage_results_SHM = pd.read_csv(
            os.path.join(config.main_path, "shm", "damage_results_shm.csv")
        )
        new_index = pd.MultiIndex.from_arrays(
            [damage_results_SHM["building_id"], damage_results_SHM["dmg_state"]]
        )
        damage_results_SHM.index = new_index
        damage_results_SHM = damage_results_SHM.drop(columns=["dmg_state"])

    # Load the "initial" exposure model
    exposure_model_undamaged = pd.read_csv(
            os.path.join(config.main_path, "exposure_models", "exposure_model_undamaged.csv")
        )
    exposure_model_undamaged.index = exposure_model_undamaged["id"]
    exposure_model_undamaged.index = exposure_model_undamaged.index.rename("asset_id")
    exposure_model_undamaged = exposure_model_undamaged.drop(columns=["id"])
    exposure_model_undamaged

    # Copy the "initial" exposure model to the 'current' sub-directory to initialise the process
    in_filename = os.path.join(
        config.main_path, "exposure_models", "exposure_model_undamaged.csv"
    )  # origin
    out_filename = os.path.join(config.main_path, "current", "exposure_model_current.csv")
    _ = shutil.copyfile(in_filename, out_filename)

    # Run RLA or OELF for each trigger
    for i, cat_filename_i in enumerate(triggers["catalogue_filename"].to_numpy()):
        type_analysis_i = triggers["type_analysis"].to_numpy()[i]

        if type_analysis_i == "RLA":
            cat_name = cat_filename_i.split(".")[0]  # Get rid of ".csv"

            # Read earthquake parameters
            earthquake_df = pd.read_csv(
                os.path.join(config.main_path, "catalogues", cat_filename_i)
            )
            earthquake_df["datetime"] = pd.to_datetime(earthquake_df["datetime"])
            earthquake_params = earthquake_df.loc[0, :].to_dict()

            exposure_updated = RapidLossAssessment.run_rla(
                earthquake_params,
                config.description_general,
                config.main_path,
                source_parameters_RLA,
                exposure_model_undamaged,
                config.mapping_damage_states,
                damage_results_SHM.loc[:, earthquake_params["event_id"]],
            )

            # Update 'exposure_model_current.csv'
            exposure_updated.to_csv(
                os.path.join(config.main_path, "current", "exposure_model_current.csv"),
                index=False,
            )

            # Get damage states per building ID
            damage_states = ExposureUpdater.summarise_damage_states_per_building_id(
                exposure_updated
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

        elif type_analysis_i == "OELF":
            # Read forecast earthquake catalogue
            forecast_cat = pd.read_csv(
                os.path.join(config.main_path, "catalogues", cat_filename_i)
            )
            forecast_cat["time_string"] = pd.to_datetime(forecast_cat["time_string"])

            forecast_name = cat_filename_i.split(".")[0]  # Get rid of ".csv"

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
            )

            # Generate the ruptures for all earthquakes in 'forecast'
            ruptures = stoch_rup.generate_ruptures(
                forecast_cat,
                path_to_ruptures,  # Ruptures will be exported to this path
                export_type='xml', # Type of file for export
            )

            damage_states = OperationalEarthquakeLossForecasting.run_oelf(
                forecast_cat,
                forecast_name,
                config.description_general,
                config.main_path,
                exposure_model_undamaged,
                config.mapping_damage_states,
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

    # Leave the program
    logger.info("Real-Time Loss Tools has finished")
    sys.exit()


if __name__ == "__main__":
    main()
