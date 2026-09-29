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
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from realtimelosstools.ruptures import Rupture
from realtimelosstools.exposure_updater import ExposureUpdater
from realtimelosstools.losses import Losses
from realtimelosstools.writers import Writer
from realtimelosstools.utils import Time
from realtimelosstools.oq_handler import OpenQuakeJob, OpenQuakeUtils


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
        nuisance_params,
        calculate_casualties,
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
        number_cores=1
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
            nuisance_params (dict):
                Dictionary with parameters to run (or not) a nuisance calculation. Contents:
                    calculate (bool):
                        True to calculate nuisance, False otherwise.
                    xml_name (str):
                        Name of the XML file that contains the nuisance model. Only necessary if
                        'calculate' is True.
                    mapping (str):
                        Name of the CSV file mapping exposure to nuisance classes. Only
                        necessary if 'calculate' is True.
                    mapping_nuisance_states (Pandas DataFrame):
                        Mapping between the names of nuisance states as output by OpenQuake
                        (index) and as labelled in the nuisance model (value). E.g.:
                                      fragility
                            dmg_state
                            no_damage       nonfelt
                            dmg_1           CDI2
                            dmg_2           CDI3
                            dmg_3           CDI4
                            dmg_4           CDI5
                            dmg_5           CDI6
            calculate_casualties (bool):
                If True, casualties are calculated.
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
                    dmg_state
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
            number_cores (int):
                Number of cores to be used for processing whenever parallelisation is possible.
                This parameter does not apply to OpenQuake, which has its own parallelisation
                strategy.

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

        if calculate_casualties:
            # Determine time of the day (used for number of occupants)
            earthquake_df = pd.DataFrame(earthquake, index=[0]) # build aux DataFrame

            earthquake_df["local_time"] = Time.determine_local_time_from_utc(
                earthquake_df["datetime"], local_timezone
            )
            earthquake_df["time_of_day"] = Time.interpret_time_of_the_day(
                earthquake_df["local_time"]
            )

            time_of_day = earthquake_df.loc[0, "time_of_day"]

            del earthquake_df # delete aux DataFrame

        else:
            time_of_day = None

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
        if calculate_casualties:
            logger.info(
                "%s Updating exposed occupants" % (np.datetime64('now'))
            )
            # Retrieve c factors for the date+time of the earthquake to run (one key per
            # damage state, e.g. {"DS0": 1, "DS1": 1, "DS2": 0, "DS3": 0, "DS4": 0}; they will
            # all be equal to 1 if no earthquake has been run before)
            recovery_occupancy_factors = Losses.get_occupancy_factors_RLA(
                earthquake["datetime"],
                mapping_damage_states,
                main_path
            )

            exposure_run = ExposureUpdater.update_exposure_occupants(
                calculate_casualties,
                exposure_full_occupants,
                recovery_occupancy_factors,
                time_of_day_occupancy,
                time_of_day,
                earthquake["datetime"],
                False,  # do not search for OELF earthquakes
                main_path,
                number_cores
            )
        else:
            exposure_run = deepcopy(exposure_full_occupants)
            exposure_run[time_of_day] = np.zeros([exposure_run.shape[0]])

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

        # Run OpenQuake to calculate damage
        oqjob_dmg = OpenQuakeJob(
            description,
            earthquake["event_id"],
            os.path.join(main_path, "current", "job.ini"),
            store_intermediate,
            "damage",
            oq_previous_hazard_id=None,
            path_to_oq_outputs=os.path.join(main_path, "openquake_output")
        )
        damage_results_OQ = oqjob_dmg.run(exposure_run, mapping_damage_states)

        # Run OpenQuake to calculate nuisance
        nuisance_states = None
        if nuisance_params["calculate"]:
            # previous_hazard_id will either be -1 (re-use) or None (calculate again)
            previous_hazard_id = OpenQuakeUtils.can_previous_hazard_be_reused(
                os.path.join(main_path, "static", nuisance_params["xml_name"]), # path to nuisance XML
                os.path.join(main_path, "current", "job.ini"), # path to damage job.ini
                main_path,
                oqjob_dmg.oq_ran, # True if previous calculation did run
            )

            # Create job_nuisance.ini to run OpenQuake calculation
            Writer.create_job_nuisance_ini(
                os.path.join(main_path, "current", "job.ini"),
                os.path.join(main_path, "current", "job_nuisance.ini"),
                nuisance_params["xml_name"],
                nuisance_params["mapping"],
                erase_hazard_params=True if previous_hazard_id==-1 else False
            )

            # Run OpenQuake to calculate nuisance
            oqjob_nuis = OpenQuakeJob(
                "%s, nuisance" % (description),
                earthquake["event_id"],
                os.path.join(main_path, "current", "job_nuisance.ini"),
                store_intermediate,
                "nuisance",
                oq_previous_hazard_id=previous_hazard_id,
                path_to_oq_outputs=os.path.join(main_path, "openquake_output")
            )
            nuisance_results_OQ = oqjob_nuis.run(
                exposure_run, nuisance_params["mapping_nuisance_states"]
            )

            if not store_openquake:
                oqjob_nuis.delete_OpenQuake_job()

            # Get nuisance states per building ID
            logger.info(
                "%s Summarising nuisance states per building ID"  % (np.datetime64('now'))
            )
            nuisance_states = ExposureUpdater.summarise_nuisance_states_per_building_id(
                nuisance_results_OQ,
                exposure_run,
                nuisance_params["mapping_nuisance_states"],
                occupants_column=time_of_day
            )

        if not store_openquake:
            # Only after the potential re-use by the nuisance calculation
            oqjob_dmg.delete_OpenQuake_job()

        # Update exposure to reflect new damage states
        # (occupants not updated yet)
        logger.info(
            "%s Updating exposure with damage states" % (np.datetime64('now'))
        )
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
        logger.info(
            "%s Summarising damage states per building ID"  % (np.datetime64('now'))
        )
        damage_states = ExposureUpdater.summarise_damage_states_per_building_id(
            exposure_updated_damage
        )

        logger.info(
            "%s Calculating economic losses"  % (np.datetime64('now'))
        )
        # Get economic losses per building ID
        losses_economic = Losses.expected_economic_loss(
            exposure_updated_damage, consequence_economic
        )

        if calculate_casualties:
            logger.info(
                "%s Calculating human losses"  % (np.datetime64('now'))
            )
            # Calculate human losses per asset of 'exposure_updated_damage'
            losses_human_per_asset = Losses.expected_human_loss_per_original_asset_id(
                exposure_updated_damage, time_of_day, consequence_injuries
            )

            logger.debug(
                "%s Calculating human losses per building ID" % (np.datetime64('now'))
            )
            # Calculate human losses per building ID
            losses_human = Losses.expected_human_loss_per_building_id(losses_human_per_asset)

            logger.debug(
                "%s Calculating timeline of recovery due to health" % (np.datetime64('now'))
            )
            # Calculate timeline of recovery (to define occupants for next earthquake)
            injured_still_away = Losses.calculate_injuries_recovery_timeline(
                losses_human_per_asset,
                recovery_injuries,
                recovery_longest_time,
                earthquake["datetime"],
            )
            logger.debug(
                "%s Calculating timeline of recovery due to inspection/repair"
                % (np.datetime64('now'))
            )
            occupancy_factors = Losses.calculate_repair_recovery_timeline(
                recovery_damage, recovery_longest_time, earthquake["datetime"]
            )
        else:
            injured_still_away = None
            losses_human = None
            occupancy_factors = None

        # Store new exposure CSV
        if store_intermediate:
            logger.debug(
                "%s Storing intermediate-step updated exposure model" % (np.datetime64('now'))
            )
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
            nuisance_states,
            losses_economic,
            losses_human,
            injured_still_away,
            occupancy_factors,
        )
