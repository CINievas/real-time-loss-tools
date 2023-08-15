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
from datetime import datetime
from openquake.commands.run import main
from realtimelosstools.ruptures import Rupture
from realtimelosstools.exposure_updater import ExposureUpdater
from realtimelosstools.losses import Losses
from realtimelosstools.writers import Writer
from realtimelosstools.utils import Time


logger = logging.getLogger()


class RapidLossAssessment:
    """This class handles methods associated with carrying out a Rapid Loss Assessment (RLA).
    """

    @staticmethod
    def run_rla(
        earthquake,
        description_general,
        main_path,
        rupture_xml_filename,
        state_dependent,
        consequence_economic,
        consequence_injuries,
        recovery_damage,
        recovery_injuries,
        recovery_longest_time,
        time_of_day_occupancy,
        local_timezone,
        original_exposure_model,
        mapping_damage_states,
        damage_results_SHM,
        store_intermediate,
        store_openquake,
    ):
        """
        This method uses OpenQuake to run a Rapid Loss Assessment (RLA) due to an input
        'earthquake' with rupture as per 'rupture_xml_filename', using the current status of the
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
            rupture_xml_filename (str):
                Name of the rupture XML file associated with this earthquake, assumed to be
                located under main_path/ruptures/rla.
            state_dependent (bool):
                True if state-dependent fragility models are being used to run OpenQuake, False
                if state-independent fragility models are being used instead.
            consequence_economic (Pandas DataFrame):
                Pandas DataFrame indicating the economic loss ratios per building class and
                damage state, with the following structure:
                    Index:
                        Taxonomy (str): Building classes.
                    Columns:
                        One per damage state (float): They contain the mean loss ratios (as
                        percentages) for each building class and damage state.
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
            occupancy_timeline (arr of int):
                Number of days since the last earthquake that mark different stages of recovery,
                used to calculate occupants of the 'exposure_updated' output (to be used for the
                next earthquake).
            recovery_damage (Pandas DataFrame):
                Pandas DataFrame indicating the expected number of days that it will take for
                people to be allowed back in to a building as a function of their damage state.
                As a minimum, its structure comprises the following:
                    Index:
                        dmg_state (str):
                            Damage state.
                    Columns:
                        N_damage (int):
                            Number of days (as an integer) for people to be allowed back into a
                            buildings as a function of their damage state (independently from
                            the health status of the people). If the damage states include
                            irreparable damage and/or collapse, use a very large number herein.
            recovery_injuries (Pandas DataFrame):
                Pandas DataFrame indicating the expected number of days that a person suffering
                from injuries of each level of severity will spend in hospital before being
                allowed to leave (i.e. before being medically discharged). It has the following
                structure:
                    Index:
                        injuries_scale (int or str):
                            Severity of the injury according to a scale.
                    Columns:
                        N_discharged (int):
                            Number of days (as an integer) for a person with each level of
                            injury to be allowed to return to their building. If the injuries
                            scale includes death, use a very large number herein.
            recovery_longest_time (numpy.datetime64):
                Maximum number of days since the time of the earthquake that will be used to
                calculate the number of occupants in the future.
            time_of_day_occupancy (dict):
                Factors by which the census population per building can be multiplied to obtain
                an estimate of the people in the building at a certain time of the day. It
                should contain one key per occupancy case present in the exposure model (e.g.
                "residential", "commercial", "industrial"), and each key should be subdivided
                into:
                    - "day": approx. 10 am to 6 pm;
                    - "night": approx. 10 pm to 6 am;
                    - "transit": approx. 6 am to 10 am and 6 pm to 10 pm.
            local_timezone (str):
                Local time zone in the format of the IANA Time Zone Database.
                E.g. "Europe/Rome".
            original_exposure_model (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake for the
                undamaged structures. It comprises the following fields:
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
                        census (float):
                            Total number of occupants in this asset, irrespective of the time of
                            the day.
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
            exposure_updated_damage (Pandas DataFrame):
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
                        census (float):
                            Total number of occupants in this asset  irrespective of the time of
                            the day.
                        occupancy (str):
                            "Res" (residential), "Com" (commercial) or "Ind" (industrial).
                        id_X, name_X (str):
                            ID and name of the administrative units to which the asset belongs.
                            "X" is the administrative level.
            damage_states (Pandas DataFrame):
                Pandas DataFrame with damage states reported per building_id. It has the
                following structure:
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
            losses_economic (Pandas DataFrame):
                Pandas DataFrame with economic losses reported per building_id through the
                following structure:
                    Index:
                        building_id (str):
                            ID of the building.
                    Columns:
                        loss (float):
                            Expected loss for 'building_id' considering all its associated
                            building classes and damage states (i.e., all its associated asset
                            IDs), with their respective probabilities.
            losses_human (Pandas DataFrame):
                Pandas DataFrame with human losses reported per building_id through the
                following structure:
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
        local_hour = Time.determine_local_time_from_utc(
            (earthquake["datetime"]).to_pydatetime(), local_timezone
        )
        time_of_day = Time.interpret_time_of_the_day(local_hour.hour)

        if time_of_day == "error":
            error_message = (
                "Something went wrong when determining the time of the day for earthquake "
                "with event ID %s. UTC time is %s. Local time is %s. The program cannot run."
                % (earthquake["event_id"], earthquake["datetime"], local_hour)
            )
            logger.critical(error_message)
            raise OSError(error_message)

        # Load exposure CSV (the exposure model to be used to run OpenQuake now)
        # (this exposure file contains census occupants not adjusted to reflect the damage
        # state of the building or the health status of people)
        exposure_full_occupants = pd.read_csv(
            os.path.join(main_path, "current", "exposure_model_current.csv"),
            dtype={"id_3": str, "id_2": str, "id_1": str}
        )
        exposure_full_occupants.index = exposure_full_occupants["id"]
        exposure_full_occupants.index = exposure_full_occupants.index.rename("asset_id")
        exposure_full_occupants = exposure_full_occupants.drop(columns=["id"])

        # Update exposure to reflect occupants for this earthquake
        # (reflecting injuries and deaths)
        exposure_run = ExposureUpdater.update_exposure_occupants(
            exposure_full_occupants,
            time_of_day_occupancy,
            time_of_day,
            earthquake["datetime"],
            mapping_damage_states,
            False,  # do not search for OELF earthquakes
            main_path,
        )

        # Update 'exposure_model_current.csv' (only update is associated with the column with
        # the number of occupants appropriate for this earthquake)
        exposure_run.to_csv(
            os.path.join(main_path, "current", "exposure_model_current.csv"),
            index=True,
            index_label="id",  # the index of 'exposure_run' is "asset_id", but OQ needs "id"
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
            "../ruptures/rla/%s" % (rupture_xml_filename),
        )

        # Run OpenQuake
        path_to_job_ini = os.path.join(main_path, "current", "job.ini")

        try:
            calc = main([path_to_job_ini])

            # Retrieve damage states from OpenQuake output
            dstore = calc.datastore.open("r")

            try:
                damage_results_OQ = dstore.read_df("damages-rlzs")
            except KeyError as ke:
                if (len(ke.args) == 1) and ("damages-rlzs" in ke.args):
                    # OpenQuake's "There is no damage, perhaps the hazard is too small?"
                    damage_results_OQ = ExposureUpdater.create_OQ_existing_damage(
                            exposure_run,
                            mapping_damage_states,
                            loss_type="structural"
                    )
                else:
                    error_message = (
                        "OpenQuake has not found 'damages-rlzs' for %s "
                        "and Real-Time Loss Tools has not been able to solve it"
                        % (description)
                    )
                    logger.critical(error_message)
                    raise OSError(error_message)

            dstore.close()

        except RuntimeError as run_e:
            if (len(run_e.args) == 1) and ("No GMFs were generated" in run_e.args[0]):
                # OpenQuake's "No GMFs were generated, perhaps they were all below
                # the minimum_intensity threshold"
                damage_results_OQ = ExposureUpdater.create_OQ_existing_damage(
                        exposure_run,
                        mapping_damage_states,
                        loss_type="structural"
                )
            else:
                error_message = (
                    "OpenQuake has crashed for %s "
                    "and Real-Time Loss Tools has not been able to solve it"
                    % (description)
                )
                logger.critical(error_message)
                raise OSError(error_message)

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

        # Update exposure to reflect new damage states
        # (occupants not updated yet)
        exposure_updated_damage = ExposureUpdater.update_exposure_with_damage_states(
            state_dependent,
            exposure_run,
            original_exposure_model,
            damage_results_OQ,
            mapping_damage_states,
            time_of_day,
            damage_results_SHM=damage_results_SHM,
        )

        # Get damage states per building ID
        damage_states = ExposureUpdater.summarise_damage_states_per_building_id(
            exposure_updated_damage
        )

        # Get economic losses per building ID
        losses_economic = Losses.expected_economic_loss(
            exposure_updated_damage, consequence_economic
        )

        # Calculate human losses per asset of 'exposure_updated_damage'
        losses_human_per_asset = Losses.expected_human_loss_per_original_asset_id(
            exposure_updated_damage, time_of_day, consequence_injuries
        )
        # Calculate human losses per building ID
        losses_human = Losses.expected_human_loss_per_building_id(losses_human_per_asset)

        # Calculate timeline of recovery (to define occupants for next earthquake)
        injured_still_away = Losses.calculate_injuries_recovery_timeline(
            losses_human_per_asset,
            recovery_injuries,
            recovery_longest_time,
            earthquake["datetime"],
        )
        occupancy_factors = Losses.calculate_repair_recovery_timeline(
            recovery_damage, recovery_longest_time, earthquake["datetime"]
        )

        # Store new exposure CSV
        if store_intermediate:
            name_exposure_csv_file_next = (
                "exposure_model_after_%s.csv" % (earthquake["event_id"])
            )
            exposure_updated_damage.to_csv(
                os.path.join(main_path, "exposure_models", "rla", name_exposure_csv_file_next),
                index=False,
            )

        # Get rid of time-of-the-day-specific numbers of occupants
        exposure_updated_damage = exposure_updated_damage.drop(columns=[time_of_day])

        return (
            exposure_updated_damage,
            damage_states,
            losses_economic,
            losses_human,
            injured_still_away,
            occupancy_factors,
        )
