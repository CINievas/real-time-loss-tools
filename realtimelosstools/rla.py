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
import pandas as pd
from openquake.commands.run import main
from realtimelosstools.ruptures import Rupture
from realtimelosstools.exposure_updater import ExposureUpdater
from realtimelosstools.losses import Losses
from realtimelosstools.writers import Writer


logger = logging.getLogger()


class RapidLossAssessment:
    """This class handles methods associated with carrying out a Rapid Loss Assessment (RLA).
    """

    @staticmethod
    def run_rla(
        earthquake,
        description_general,
        main_path,
        source_parameters,
        consequence_injuries,
        original_exposure_model,
        mapping_damage_states,
        damage_results_SHM,
        store_intermediate,
        store_openquake,
    ):
        """
        This method uses OpenQuake to run a Rapid Loss Assessment (RLA) due to an input
        'earthquake' with specified 'source_parameters', using the current status of the
        exposure model, assumed to be a CSV file in OpenQuake format stored under
        'main_path'/current/exposure_model_current.csv, and returning its updated version, which
        is determined combining the damage results from OpenQuake and those obtained through
        Structural Health Monitoring (SHM) techniques ('damage_results_SHM'). The
        'original_exposure_model' is used to keep track of the original asset_id (i.e. ID used
        by OpenQuake, one ID per row of the exposure CSV file).

        Args:
            earthquake (dict):
                Dictionary containing the following earthquake parameters:
                    longitude (float): Longitude of the hypocentre.
                    latitude (float): Latitude of the hypocentre.
                    depth (float): Depth of the hypocentre.
                    magnitude (float): Moment magnitude.
                    datetime (Numpy datetime64): Date and time.
                    event_id (str): Unique identifier.
            description_general (str):
                General description of the run/analysis, used for the OpenQuake job.ini file.
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.
            source_parameters (Pandas DataFrame):
                DataFrame containing parameters of earthquake sources, with the following
                columns:
                    event_id (str):
                        Event ID. This must also be the index of the DataFrame.
                    Mw (float):
                        Moment magnitude of the event.
                    nucleation_lon (float):
                        Longitude of the nucleation point (hypocentre), in degrees.
                    nucleation_lat (float):
                        Latitude of the nucleation point (hypocentre), in degrees.
                    nucleation_depth (float):
                        Depth of the nucleation point (hypocentre), in km.
                    LL_lon (float):
                        Longitude (in degrees) of the lower left corner of the rupture plane.
                    LL_lat (float):
                        Latitude (in degrees) of the lower left corner of the rupture plane.
                    UR_lon (float):
                        Longitude (in degrees) of the upper right corner of the rupture plane.
                    UR_lat (float):
                        Latitude (in degrees) of the upper right corner of the rupture plane.
                    LR_lon (float):
                        Longitude (in degrees) of the lower right corner of the rupture plane.
                    LR_lat (float):
                        Latitude (in degrees) of the lower right corner of the rupture plane.
                    UL_lon (float):
                        Longitude (in degrees) of the upper left corner of the rupture plane.
                    UL_lat (float):
                        Latitude (in degrees) of the upper left corner of the rupture plane.
                    Z_top (float):
                        Depth to the top of the rupture, in km.
                    Strike (float):
                        Strike of the rupture, in degrees, measured from north.
                    Dip (float):
                        Dip of the rupture, in degrees, measured downwards from the horizontal.
                    Rake (float):
                        Rake of the rupture, in degrees.
            consequence_injuries (dict of Pandas DataFrame):
                Dictionary whose keys are the injury severity levels and whose contents are
                Pandas DataFrames with the consequence models for injuries in terms of mean
                values of loss ratio per damage state. Each row in the consequence model
                corresponds to a different building class. The structure is as follows:
                    Index:
                        Taxonomy (str): Building classes.
                    Columns:
                        One per damage state (float): They contain the mean loss ratios (as
                        percentages) for each building class and damage state.
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
            damage_results_SHM (Pandas Series):
                Pandas Series with probabilities of monitored buildings being in each damage
                state. This is output from SHM activities. It comprises the following fields:
                    Index is multiple:
                        building_id (str):
                            ID of the building.
                        dmg_state (str):
                            Damage states.
                    Values of the series (float): Probability of 'dmg_state' for 'building_id'.
            store_intermediate (bool):
                If True, a series of intermediate outputs will be stored. These intermediate
                outputs are:
                    - The exposure model updated after each earthquake, i.e. the output
                    'exposure_updated' of this model, to be stored under 'main_path'/
                    exposure_models/rla/exposure_model_after_XXX.csv.
                    - The damage results as directly output by OpenQuake, to be stored under
                    'main_path'/openquake_output/XXX_damages_OQ_raw.csv.
                    - The damage results from OpenQuake, adjusted so that the do not include
                    negative numbers of buildings, to be stored under 'main_path'/
                    openquake_output/XXX_damages_OQ.csv.
            store_openquake (bool):
                If True, OpenQuake HDF5 files will be stored and jobs will be kept in
                OpenQuake's database. If false, OpenQuake's database will be purged of the last
                job after running.

        Returns:
            exposure_updated (Pandas DataFrame):
                Pandas DataFrame with the updated exposure model that results from the rapid
                loss assessment. The rows are ordered by 'asset_id' and 'dmg_state' in ascending
                order. It comprises the following fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class) from the previous exposure model (i.e.
                            from 'previous_exposure_model').
                        dmg_state (str):
                            Damage states.
                    Columns:
                        id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class) in the new exposure model (i.e. one
                            unique value per row of 'new_exposure_model').
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
            losses_human (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        building_id (str):
                            ID of the building.
                    Columns:
                        injuries_Y (float):
                            Expected injuries of severity Y for 'building_id', considering all
                            its associated building classes and damage states (i.e., all its
                            associated asset IDs), with their respective probabilities.
        """

        # Description
        description = "%s, event ID %s" % (description_general, earthquake["event_id"])

        # Determine time of the day (used for number of occupants)
        local_hour = Rupture.determine_local_time_from_utc(
            earthquake["datetime"], "timezone"
        )
        time_of_day = Rupture.interpret_time_of_the_day(local_hour.hour)

        if time_of_day == "error":
            error_message = (
                "Something went wrong when determining the time of the day for earthquake "
                "with event ID %s. UTC time is %s. Local time is %s. The program cannot run."
                % (earthquake["event_id"], earthquake["datetime"], local_hour)
            )
            logger.critical(error_message)
            raise OSError(error_message)

        # Define rupture XML
        name_rupture_file = "rupture_%s.xml" % (earthquake["event_id"])

        (
            strike,
            dip,
            rake,
            hypocenter,
            rupture_plane,
        ) = Rupture.build_rupture_from_ITACA_parameters(
            earthquake["event_id"], source_parameters
        )

        Writer.write_rupture_xml(
            os.path.join(main_path, "ruptures", "rla", name_rupture_file),
            strike,
            dip,
            rake,
            earthquake["magnitude"],
            hypocenter,
            rupture_plane,
        )

        # Update exposure XML
        Writer.update_exposure_xml(
            os.path.join(main_path, "current", "exposure_model.xml"),
            time_of_day,
            "exposure_model_current.csv",
        )

        # Update job.ini with the description, time of the day and name of the rupture XML
        Writer.update_job_ini(
            os.path.join(main_path, "current", "job.ini"),
            description,
            time_of_day,
            "../ruptures/rla/%s" % (name_rupture_file),
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

        if not store_openquake:  # Erase the job just run from OpenQuake's database (and HDF5)
            Writer.delete_OpenQuake_last_job()

        # Store damage states from OpenQuake output to CSV (incl. potential negative values)
        if store_intermediate:
            damage_results_OQ.to_csv(
                os.path.join(
                    main_path,
                    "openquake_output",
                    "%s_damages_OQ_raw.csv"
                    % (earthquake["event_id"])
                ),
            )

        # Ensure no negative number of buildings in 'damage_results_OQ'
        damage_results_OQ = ExposureUpdater.ensure_no_negative_damage_results_OQ(
            damage_results_OQ
        )

        # Store adjusted damage states from OpenQuake output to CSV
        if store_intermediate:
            damage_results_OQ.to_csv(
                os.path.join(
                    main_path,
                    "openquake_output",
                    "%s_damages_OQ.csv"
                    % (earthquake["event_id"])
                ),
            )

        # Load exposure CSV (the exposure model just used to run OpenQuake)
        exposure_run = pd.read_csv(
            os.path.join(main_path, "current", "exposure_model_current.csv"),
            dtype={"id_3": str, "id_2": str, "id_1": str}
        )
        exposure_run.index = exposure_run["id"]
        exposure_run.index = exposure_run.index.rename("asset_id")
        exposure_run = exposure_run.drop(columns=["id"])

        # Update exposure to reflect new damage states
        # (occupants not updated yet)
        exposure_updated_damage = ExposureUpdater.update_exposure_with_damage_states(
            exposure_run,
            original_exposure_model,
            damage_results_OQ,
            mapping_damage_states,
            damage_results_SHM=damage_results_SHM,
        )

        # Calculate human losses per asset of 'exposure_updated_damage'
        losses_human_per_asset = Losses.expected_human_loss_per_asset_id(
            exposure_updated_damage, time_of_day, consequence_injuries
        )
        # Calculate human losses per building ID
        losses_human = Losses.expected_human_loss_per_building_id(losses_human_per_asset)

        # Update exposure to reflect new occupants (reflecting injuries and deaths)
        exposure_updated = ExposureUpdater.update_exposure_occupants(
            exposure_updated_damage,
            losses_human_per_asset,
        )

        # Store new exposure CSV
        if store_intermediate:
            name_exposure_csv_file_next = (
                "exposure_model_after_%s.csv" % (earthquake["event_id"])
            )
            exposure_updated.to_csv(
                os.path.join(main_path, "exposure_models", "rla", name_exposure_csv_file_next),
                index=False,
            )

        return exposure_updated, losses_human
