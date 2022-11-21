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
import os
import shutil
from copy import deepcopy
import numpy as np
import pandas as pd
from openquake.commands.run import main
from realtimelosstools.ruptures import Rupture
from realtimelosstools.exposure_updater import ExposureUpdater
from realtimelosstools.writers import Writer


logger = logging.getLogger()


class OperationalEarthquakeLossForecasting():
    """This class handles methods associated with carrying out an Operational Earthquake Loss
    Forecast (OELF).
    """

    @staticmethod
    def run_oelf(
        forecast_catalogue,
        forecast_name,
        description_general,
        main_path,
        original_exposure_model,
        mapping_damage_states,
        store_intermediate,
    ):
        """
        This method uses OpenQuake to run an Operational Earthquake Loss Forecast (OELF) due to
        an input 'forecast_catalogue' with name 'forecast_name'. The input 'forecast_catalogue'
        may consist of one or more seismicity forecasts, whose ID is indicated under the field
        'catalog_id' of 'forecast_catalogue'. Each seismicity forecast is run independently,
        always starting with the exposure model stored under
        main_path/current/exposure_model_current.csv, which is copied for each seismicity
        forecast
        under main_path/exposure_models/oelf/forecast_name/exposure_model_current_oelf_XX.csv,
        where "XX" is the 'catalog_id'. These "exposure_model_current_oelf_XX.csv" files are
        updated after running each earthquake of 'catalog_id', to reflec the damage states of
        the buildings, but main_path/current/exposure_model_current.csv is not updated. The
        'original_exposure_model' is used to keep track of the original asset_id (i.e. ID used
        by OpenQuake, one ID per row of the exposure CSV file). Results from all (realisations
        of) the seismicity forecasts are finally averaged and returned.

        Args:
            forecast_catalogue (Pandas DataFrame):
                DataFrame containing a seismicity forecast for a period of time of interest. It
                normally consists of several realisations of seismicity, each realisation being
                identified by a 'catalog_id'. One 'catalog_id' can contain one or more
                earthquakes, each of them identified with an 'event_id'. The values of
                'event_id' can be repeated across different values of 'catalog_id'. It is
                assumed that the XML of the ruptures of 'forecast_catalogue' exist under
                main_path/ruptures/oelf/forecast_name. The DataFrame must contain the following
                fields:
                    longitude (float): Longitude of the hypocentre.
                    latitude (float): Latitude of the hypocentre.
                    magnitude (float): Moment magnitude.
                    time_string (Numpy datetime64): Date and time.
                    depth (float): Depth of the hypocentre.
                    catalog_id (int): ID of the realisation of seismicity that the earthquake\
                    belongs to.
                    event_id (str): Unique identifier of an earthquake within a 'catalog_id'.
            forecast_name (str):
                Name of the forecast. It is used for the description of the OpenQuake job.ini
                file and to create sub-directories to store files associated with this forecast.
            description_general (str):
                General description of the run/analysis, used for the OpenQuake job.ini file.
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.
            original_exposure_model (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake for the undamaged
                structures. It comprises the following fields:
                    Index (simple):
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                    Columns:
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of asset_id.
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model. It can be the value of 'asset_id' in the undamaged version or
                            any other unique ID per row that refers to a combination of a
                            building ID and a building class with no initial damage.
                        lon (float):
                            Longitude of the asset in degrees.
                        lat (float):
                            Latitude of the asset in degrees.
                        taxonomy (str):
                            Building class.
                        number (float):
                            Number of buildings in this asset.
                        structural (float):
                            Total replacement cost of this asset (all buildings in "number").
                        night, day, transit (float):
                            Total number of occupants in this asset at different times of the
                            day.
                        occupancy (str):
                            "Res" (residential), "Com" (commercial) or "Ind" (industrial).
                        id_X, name_X (str):
                            ID and name of the administrative units to which the asset belongs.
                            "X" is the administrative level.
            mapping_damage_states (Pandas DataFrame):
                Mapping between the names of damage states as output by OpenQuake (index) and as
                labelled in the fragility model (value). E.g.:
                              fragility
                    asset_id
                    no_damage       DS0
                    dmg_1           DS1
                    dmg_2           DS2
                    dmg_3           DS3
                    dmg_4           DS4
            store_intermediate (bool):
                If True, a series of intermediate outputs will be stored. These intermediate
                outputs are:
                    - The exposure model updated after each earthquake, to be stored under
                    'main_path'/exposure_models/oelf/forecast_name/exposure_model_after_XXX.csv.
                    The "current" OELF exposure file will always be stored, irrespective of
                    'store_intermediate'.
                    - The damage results as directly output by OpenQuake, to be stored under
                    'main_path'/openquake_output/forecast_name/XXX_damages_OQ_raw.csv.
                    - The damage results from OpenQuake, adjusted so that the do not include
                    negative numbers of buildings, to be stored under 'main_path'/
                    openquake_output/forecast_name/XXX_damages_OQ.csv.

        Returns:
            damage_states_all_realisations (Pandas DataFrame):
                Pandas DataFrame with the damage states resulting from the average of all OELF
                realisations, reported per building_id. It has the following structure:
                    Index is multiple:
                        building_id (str):
                            ID of the building.
                        damage_state (str):
                            Damage states.
                    Columns:
                        number (float):
                            Probability of 'damage_state' for 'building_id', if building_id is
                            one individual building, or number of buildings of 'building_id'
                            under 'damage_state', if building_id is a group of buildings.
        """

        # Create sub-directory to store OpenQuake outputs
        if store_intermediate:
            path_to_oq_outputs = os.path.join(main_path, "openquake_output", forecast_name)
            if not os.path.exists(path_to_oq_outputs):
                os.mkdir(path_to_oq_outputs)
            else:
                error_message = (
                    "The directory %s already exists under %s/openquake_output and may contain "
                    "results from a previous run. The program will stop."
                    % (forecast_name, main_path)
                )
                logger.critical(error_message)
                raise OSError(error_message)

        # Create sub-directory to store general outputs
        path_to_outputs = os.path.join(main_path, "output", forecast_name)
        if not os.path.exists(path_to_outputs):
            os.mkdir(path_to_outputs)
        else:
            error_message = (
                "The directory %s already exists under %s/output and may contain results from "
                "a previous run. The program will stop."
                % (forecast_name, main_path)
            )
            logger.critical(error_message)
            raise OSError(error_message)

        # Create sub-directory to store updated exposure files
        path_to_exposure = os.path.join(main_path, "exposure_models", "oelf", forecast_name)
        if not os.path.exists(path_to_exposure):
            os.mkdir(path_to_exposure)
        else:
            error_message = (
                "The directory %s already exists under %s/exposure_models/oelf and may contain "
                "results from a previous run. The program will stop."
                % (forecast_name, main_path)
            )
            logger.critical(error_message)
            raise OSError(error_message)

        # Identify IDs of individual realisations of seismicity
        oef_realisation_ids = forecast_catalogue["catalog_id"].unique()

        # Initialise 'damage_states_all_realisations'
        damage_states_all_realisations = None

        for oef_realisation_id in oef_realisation_ids:  # Each of the seismicity forecasts
            # Earthquakes that belong only to this realisation of seismicity
            filter_realisation = (forecast_catalogue["catalog_id"] == oef_realisation_id)
            aux = forecast_catalogue[filter_realisation]
            # Order the earthquakes in chronological order
            events_in_realisation = aux.sort_values(by=['time_string'], ignore_index=True)

            # Initialise exposure_model_current_XX.csv, with XX = oef_realisation_id
            in_filename = os.path.join(
                main_path, "current", "exposure_model_current.csv"
            )  # origin
            current_exposure_filename = "exposure_model_current_oelf_%s.csv" % (
                oef_realisation_id
            )
            out_filename = os.path.join(path_to_exposure, current_exposure_filename)
            _ = shutil.copyfile(in_filename, out_filename)

            # Run earthquake by earthquake of this OEF realisation
            for i, eq_id in enumerate(events_in_realisation["EQID"]):  # i is index of DataFrame
                # Description
                description = "%s, %s, event ID %s" % (
                    description_general, forecast_name, eq_id
                )

                # Determine time of the day (used for number of occupants)
                local_hour = Rupture.determine_local_time_from_utc(
                    events_in_realisation.loc[i, "time_string"], "timezone"
                )
                time_of_day = Rupture.interpret_time_of_the_day(local_hour.hour)

                if time_of_day == "error":
                    error_message = (
                        "Something went wrong when determining the time of the day for "
                        "earthquake of %s with event ID %s. UTC time is %s. Local time is %s. "
                        "The program cannot run."
                        % (
                            forecast_name,
                            eq_id,
                            events_in_realisation.loc[i, "time_string"],
                            local_hour,
                        )
                    )
                    logger.critical(error_message)
                    raise OSError(error_message)

                # Identify rupture XML
                name_rupture_file = "RUP_%s-%s.xml" % (
                    events_in_realisation.loc[i, "catalog_id"],
                    events_in_realisation.loc[i, "event_id"]
                )

                # Update exposure XML
                Writer.update_exposure_xml(
                    os.path.join(main_path, "current", "exposure_model.xml"),
                    time_of_day,
                    "../exposure_models/oelf/%s/%s" % (
                        forecast_name, current_exposure_filename
                    ),
                )

                # Update job.ini with the description, time of the day and name_rupture_file
                Writer.update_job_ini(
                    os.path.join(main_path, "current", "job.ini"),
                    description,
                    time_of_day,
                    "../ruptures/oelf/%s/%s" % (forecast_name, name_rupture_file),
                )

                # Run OpenQuake
                path_to_job_ini = os.path.join(main_path, "current", "job.ini")
                calc = main([path_to_job_ini])

                # Retrieve damage states from OpenQuake output
                dstore = calc.datastore.open("r")
                damage_results_OQ = dstore.read_df("damages-rlzs")
                dstore.close()
                new_index = pd.MultiIndex.from_arrays(
                    [damage_results_OQ["asset_id"], damage_results_OQ["dmg_state"]]
                )
                damage_results_OQ.index = new_index
                damage_results_OQ = damage_results_OQ.drop(columns=["asset_id", "dmg_state"])

                # Store damage states from OpenQuake output to CSV
                # (incl. potential negative values)
                if store_intermediate:
                    damage_results_OQ.to_csv(
                        os.path.join(path_to_oq_outputs, "%s_damages_OQ_raw.csv" % (eq_id))
                    )

                # Ensure no negative number of buildings in 'damage_results_OQ'
                damage_results_OQ = ExposureUpdater.ensure_no_negative_damage_results_OQ(
                    damage_results_OQ
                )

                # Store adjusted damage states from OpenQuake output to CSV
                if store_intermediate:
                    damage_results_OQ.to_csv(
                        os.path.join(path_to_oq_outputs, "%s_damages_OQ.csv" % (eq_id))
                    )

                # Load exposure CSV (the exposure model just used to run OpenQuake)
                exposure_run = pd.read_csv(
                    os.path.join(path_to_exposure, current_exposure_filename),
                    dtype={"id_3": str, "id_2": str, "id_1": str}
                )
                exposure_run.index = exposure_run["id"]
                exposure_run.index = exposure_run.index.rename("asset_id")
                exposure_run = exposure_run.drop(columns=["id"])

                # Update exposure
                exposure_updated = ExposureUpdater.update_exposure(
                    exposure_run,
                    original_exposure_model,
                    damage_results_OQ,
                    mapping_damage_states,
                )

                # Store new exposure CSV
                if store_intermediate:
                    name_exposure_csv_file_next = "exposure_model_after_%s.csv" % (eq_id)
                    # Named after the earthquake
                    exposure_updated.to_csv(
                        os.path.join(path_to_exposure, name_exposure_csv_file_next),
                        index=False,
                    )
                # Replace current exposure for this OEF realisation
                exposure_updated.to_csv(
                    os.path.join(path_to_exposure, current_exposure_filename),
                    index=False,
                )

            # Get damage states per building ID for this OELF realisation
            damage_states = ExposureUpdater.summarise_damage_states_per_building_id(
                exposure_updated
            )
            # Store damage states per building ID
            damage_states.to_csv(
                os.path.join(
                    path_to_outputs,
                    "damage_states_after_OELF_%s_realisation_%s.csv" % (
                        forecast_name, oef_realisation_id
                    )
                ),
                index=True,
            )

            # Concatenate damage states from all realisations
            if damage_states_all_realisations is None:
                # First realisation
                damage_states_all_realisations = deepcopy(damage_states)
            else:
                damage_states_all_realisations = pd.concat(
                    [damage_states_all_realisations, damage_states]
                )

        # Average damage states per building ID for all OELF realisations
        damage_states_all_realisations = damage_states_all_realisations.groupby(
            ["building_id", "damage_state"]
        ).mean()

        return damage_states_all_realisations
