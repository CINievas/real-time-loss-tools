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
import pytz
from copy import deepcopy
import numpy as np
import pandas as pd
from datetime import datetime
from openquake.commands.run import main
from openquake.hazardlib import geo
from realtimelosstools.ruptures import Rupture
from realtimelosstools.exposure_updater import ExposureUpdater
from realtimelosstools.losses import Losses
from realtimelosstools.writers import Writer
from realtimelosstools.utils import Time


logger = logging.getLogger()


class OperationalEarthquakeLossForecasting():
    """This class handles methods associated with carrying out an Operational Earthquake Loss
    Forecast (OELF).
    """

    @staticmethod
    def run_oelf(
        forecast_catalogue,
        forecast_name,
        there_can_be_occupants,
        forecast_continuous_ses_numbering,
        forecast_ses_range,
        state_dependent,
        description_general,
        main_path,
        original_exposure_model,
        consequence_economic,
        consequence_injuries,
        recovery_damage,
        recovery_injuries,
        recovery_longest_time,
        time_of_day_occupancy,
        local_timezone,
        mapping_damage_states,
        store_intermediate,
        store_openquake,
    ):
        """
        This method uses OpenQuake to run an Operational Earthquake Loss Forecast (OELF) due to
        an input 'forecast_catalogue' with name 'forecast_name'. The input 'forecast_catalogue'
        may consist of one or more seismicity forecasts, which we call "stochastic event sets"
        (SES) following OpenQuake and whose ID is indicated under the field 'ses_id' of
        'forecast_catalogue'. Each stochastic event set is run independently, always starting
        with the exposure model stored under main_path/current/exposure_model_current.csv, which
        is copied for each SES under main_path/exposure_models/oelf/forecast_name/
        exposure_model_current_oelf_XX.csv, where "XX" is the 'ses_id'. These
        "exposure_model_current_oelf_XX.csv" files are updated after running each earthquake of
        'ses_id', to reflect the damage states of the buildings, but main_path/current/
        exposure_model_current.csv is not updated. The 'original_exposure_model' is used to keep
        track of the original asset_id (i.e. ID used by OpenQuake, one ID per row of the
        exposure CSV file). Results from all stochastic event sets (i.e. all realisations of
        forecast seismicity) are finally averaged and returned.

        The method contemplates the possibility that not all desired stochastic event sets are
        contained in 'forecast_catalogue'. This may be the case, for example, if the seismicity
        forecast software only outputs earthquakes with a certain magnitude, which leads to some
        stochastic event sets being "empty". In order to consider the "empty" SES,
        'forecast_continuous_ses_numbering' needs to be set to True and the two integer values
        contained in 'forecast_ses_range' will be used to define the range of IDs of the SES. If
        the ID of a SES defined in this way is not found in 'forecast_catalogue', the effect is
        to consider that the SES does not produce any damage or losses.

        Args:
            forecast_catalogue (Pandas DataFrame):
                DataFrame containing a seismicity forecast for a period of time of interest. It
                normally consists of several realisations of seismicity, which we call
                "stochastic event sets" (SES), each realisation being identified by a 'ses_id'.
                One 'ses_id' can contain one or more earthquakes, each of them identified with
                an 'event_id'. The values of 'event_id' can be repeated across different values
                of 'ses_id'. It is assumed that the XML of the ruptures of 'forecast_catalogue'
                exist under main_path/ruptures/oelf/forecast_name. The DataFrame must contain
                the following fields:
                    longitude (float): Longitude of the hypocentre.
                    latitude (float): Latitude of the hypocentre.
                    magnitude (float): Moment magnitude.
                    datetime (Numpy datetime64): Date and time.
                    depth (float): Depth of the hypocentre.
                    ses_id (int): ID of the stochastic event set (SES) that the earthquake
                    belongs to.
                    event_id (str): Unique identifier of an earthquake within a 'ses_id'.
                    to_run (bool): If True, the corresponding earthquake is to be
                    used to run damage/loss calculations; if False, it will be skipped.
            forecast_name (str):
                Name of the forecast. It is used for the description of the OpenQuake job.ini
                file and to create sub-directories to store files associated with this forecast.
            there_can_be_occupants (bool):
                If True, the occupants of the buildings will be calculated considering the
                injured people still away (from all previous "real" RLA earthquakes and all
                previous OELF earthquakes of each individual stochastic event set). If False,
                the occupants of the buildings will be set to zero (set this parameter to False
                if you assume that 'forecast_catalogue' spans too short a duration to allow for
                occupants to return to buildings after previous earthquakes).
            forecast_continuous_ses_numbering (bool):
                If True, the method will assume there are as many stochastic event sets as
                indicated in 'forecast_ses_range', with an increment of 1. If False, the IDs of
                the stochastic event sets will be read from 'forecast_catalogue'.
            forecast_ses_range (list of two int):
                Start and end number of the ID of the stochastic event sets, which will be used
                to define the IDs of the stochastic event sets only if
                'forecast_continuous_ses_numbering' is True. Both start and end numbers are
                included.
            state_dependent (bool):
                True if state-dependent fragility models are being used to run OpenQuake, False
                if state-independent fragility models are being used instead.
            description_general (str):
                General description of the run/analysis, used for the OpenQuake job.ini file.
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.
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
                        night, day, transit, census (float):
                            Total number of occupants in this asset at different times of the
                            day (night, day, transit) and irrespective of the time of the day
                            (census).
                        occupancy (str):
                            "Res" (residential), "Com" (commercial) or "Ind" (industrial).
                        id_X, name_X (str):
                            ID and name of the administrative units to which the asset belongs.
                            "X" is the administrative level.
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
                    'main_path'/openquake_output/forecast_name/XXX_damages_OQ_raw.csv, for each
                    individual earthquake run.
                    - The damage results from OpenQuake, adjusted so that the do not include
                    negative numbers of buildings, to be stored under 'main_path'/
                    openquake_output/forecast_name/XXX_damages_OQ.csv, for each individual
                    earthquake run.
                    - The damage states per building ID for each full realisation of seismicity,
                    to be stored under 'main_path'/output/forecast_name/
                    damage_states_after_OELF_forecast_name_realisation_XXX.csv.
            store_openquake (bool):
                If True, OpenQuake HDF5 files will be stored and jobs will be kept in
                OpenQuake's database. If false, OpenQuake's database will be purged of the last
                job after each run.

        Returns:
            damage_states_all_ses (Pandas DataFrame):
                Pandas DataFrame with the damage states resulting from the average of all OELF
                realisations, i.e. all stochastic event sets of 'forecast_catalogue', reported
                per building_id. It has the following structure:
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
            losses_economic_all_ses (Pandas DataFrame):
                Pandas DataFrame with the economic losses resulting from the average of all OELF
                realisations, i.e. all stochastic event sets of 'forecast_catalogue', reported
                per building_id. It has the following structure:
                    Index:
                        building_id (str):
                            ID of the building.
                    Columns:
                        loss (float):
                            Expected loss for 'building_id' considering all its associated
                            building classes and damage states, with their respective
                            probabilities, as well as all stochastic event sets of seismicity.
            losses_human_all_ses (Pandas DataFrame):
                Pandas DataFrame with the human losses resulting from the average of all OELF
                realisations, i.e. all stochastic event sets of 'forecast_catalogue', reported
                per building_id. It has the following structure:
                    Index:
                        building_id (str):
                            ID of the building.
                    Columns:
                        loss (float):
                            Expected loss for 'building_id' considering all its associated
                            building classes and damage states, with their respective
                            probabilities, as well as all stochastic event sets of seismicity.
                        injuries_Y (float):
                            Expected injuries of severity Y for 'building_id', considering all
                            its associated building classes and damage states, with their
                            respective probabilities, as well as all stochastic event sets of
                            seismicity.
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

        # Create sub-directory to store general outputs per realisation of seismicity (SES)
        if store_intermediate:
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

        # IDs of individual realisations of seismicity (stochastic event sets, SES)
        ses_ids_in_catalogue = forecast_catalogue["ses_id"].unique()
        if forecast_continuous_ses_numbering:
            # Generate IDs based on input configuration
            oef_ses_ids = [i for i in range(forecast_ses_range[0], forecast_ses_range[1]+1)]
            # Check if there are SES IDs in 'forecast_catalogue' that are not in 'oef_ses_ids'
            existing_IDs_missing = False
            for oef_ses_id in enumerate(ses_ids_in_catalogue):
                if oef_ses_id not in oef_ses_ids:
                    existing_IDs_missing = True
                    break
            if existing_IDs_missing:
                logger.warning(
                    "The seismicity forecast %s contains IDs of stochastic event sets not "
                    "contemplated by the input configuration ('forecast_ses_range' parameter)."
                    % (forecast_name)
                )
        else:
            # Identify IDs in the input 'forecast_catalogue'
            oef_ses_ids = deepcopy(ses_ids_in_catalogue)

        # Initialise 'damage_states_all_ses', 'losses_economic_all_ses', 'losses_human_all_ses'
        damage_states_all_ses = None
        losses_economic_all_ses = None
        losses_human_all_ses = None

        for k, oef_ses_id in enumerate(oef_ses_ids):  # Each of the stochastic event sets

            logger.info(
                "%s Working on stochastic event set %s of %s"
                % (np.datetime64('now'), k+1, len(oef_ses_ids))
            )

            # Create sub-directory to store "injured_still_away" and "occupancy_factors" files
            # (this sub-directory and all its contents will be erased at the end of this SES)
            path_to_occupants_oelf = os.path.join(main_path, "current", "occupants", "oelf")
            if not os.path.exists(path_to_occupants_oelf):
                os.mkdir(path_to_occupants_oelf)
            else:
                error_message = (
                    "The directory %s already exists and may contain results from "
                    "a previous run. The program will stop."
                    % (path_to_occupants_oelf)
                )
                logger.critical(error_message)
                raise OSError(error_message)

            # Earthquakes that belong only to this realisation of seismicity (SES)
            filter_realisation = (forecast_catalogue["ses_id"] == oef_ses_id)
            aux = forecast_catalogue[filter_realisation]
            # Order the earthquakes of this SES in chronological order
            events_in_ses = aux.sort_values(by=['datetime'])  # can be empty

            # Initialise exposure_model_current_XX.csv, with XX = oef_ses_id
            in_filename = os.path.join(
                main_path, "current", "exposure_model_current.csv"
            )  # origin
            current_exposure_filename = "exposure_model_current_oelf_%s.csv" % (
                oef_ses_id
            )
            out_filename = os.path.join(path_to_exposure, current_exposure_filename)
            _ = shutil.copyfile(in_filename, out_filename)

            damage_states = None
            losses_human_all_events = None
            # Initialise exposure, in case there are no elements in 'events_in_ses'
            exposure_updated_damage = pd.read_csv(
                os.path.join(path_to_exposure, current_exposure_filename),
                dtype={"id_3": str, "id_2": str, "id_1": str}
            )
            exposure_updated_damage.index = exposure_updated_damage["id"]
            exposure_updated_damage.index = exposure_updated_damage.index.rename("asset_id")
            exposure_updated_damage = exposure_updated_damage.drop(columns=["id"])

            # Run earthquake by earthquake of this stochastic event set
            for eq_id in events_in_ses.index:  # EQID is index of DataFrame

                # Description
                description = "%s, %s, event ID %s" % (
                    description_general, forecast_name, eq_id
                )

                # Determine time of the day (used for number of occupants)
                local_hour = Time.determine_local_time_from_utc(
                    (events_in_ses.loc[eq_id, "datetime"]).to_pydatetime(),
                    local_timezone
                )
                time_of_day = Time.interpret_time_of_the_day(
                    local_hour.hour
                )

                if time_of_day == "error":
                    error_message = (
                        "Something went wrong when determining the time of the day for "
                        "earthquake of %s with event ID %s. UTC time is %s. Local time is %s. "
                        "The program cannot run."
                        % (
                            forecast_name,
                            eq_id,
                            events_in_ses.loc[eq_id, "datetime"],
                            local_hour,
                        )
                    )
                    logger.critical(error_message)
                    raise OSError(error_message)

                # Load exposure CSV (the exposure model to be used to run OpenQuake with this
                # earthquake). This exposure file contains census occupants not adjusted to
                # reflect the damage state of the building or the health status of people
                exposure_full_occupants = pd.read_csv(
                    os.path.join(path_to_exposure, current_exposure_filename),
                    dtype={"id_3": str, "id_2": str, "id_1": str}
                )
                exposure_full_occupants.index = exposure_full_occupants["id"]
                exposure_full_occupants.index = exposure_full_occupants.index.rename("asset_id")
                exposure_full_occupants = exposure_full_occupants.drop(columns=["id"])

                if there_can_be_occupants:
                    # Update exposure to reflect occupants for this earthquake
                    # (reflecting injuries and deaths)
                    exposure_run = ExposureUpdater.update_exposure_occupants(
                        exposure_full_occupants,
                        time_of_day_occupancy,
                        time_of_day,
                        events_in_ses.loc[eq_id, "datetime"],
                        mapping_damage_states,
                        True,  # include OELF previous earthquakes (not just RLA)
                        main_path,
                    )
                else:
                    # Assume all occupants are zero
                    exposure_run = deepcopy(exposure_full_occupants)
                    exposure_run[time_of_day] = np.zeros([exposure_run.shape[0]])

                # Update 'exposure_model_current.csv' (only update is associated with the column with
                # the number of occupants appropriate for this earthquake)
                exposure_run.to_csv(
                    os.path.join(path_to_exposure, current_exposure_filename),
                    index=True,
                    index_label="id",  # the index of 'exposure_run' is "asset_id", but OQ needs "id"
                )

                if not events_in_ses.loc[eq_id, "to_run"]:
                    # Magnitude too small or too far away from all exposure sites
                    # --> skip this earthquake and go on to the next one
                    # The 'exposure_updated_damage' is the same as the exposure so far
                    exposure_updated_damage = deepcopy(exposure_run)
                    exposure_updated_damage = exposure_updated_damage.drop(
                        columns=[time_of_day]
                    )
                    # No need to add zeros to losses_human_all_events (i.e. no injuries)
                    continue

                # Identify rupture XML
                name_rupture_file = "RUP_%s-%s.xml" % (
                    events_in_ses.loc[eq_id, "ses_id"],
                    events_in_ses.loc[eq_id, "event_id"]
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

                if not store_openquake:
                    # Erase the job just run from OpenQuake's database (and HDF5)
                    Writer.delete_OpenQuake_last_job()

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

                # Update exposure to reflect new damage states
                # (occupants not updated yet)
                exposure_updated_damage = ExposureUpdater.update_exposure_with_damage_states(
                    state_dependent,
                    exposure_run,
                    original_exposure_model,
                    damage_results_OQ,
                    mapping_damage_states,
                    time_of_day,
                )

                # Calculate human losses per asset of 'exposure_updated_damage'
                losses_human_per_asset = Losses.expected_human_loss_per_original_asset_id(
                    exposure_updated_damage, time_of_day, consequence_injuries
                )
                # Calculate human losses per building ID
                losses_human_per_building_id = Losses.expected_human_loss_per_building_id(
                    losses_human_per_asset
                )

                # Concatenate human losses from all earthquakes in the stochastic event set
                if losses_human_all_events is None:
                    # First earthquake
                    losses_human_all_events = deepcopy(losses_human_per_building_id)
                else:
                    losses_human_all_events = pd.concat(
                        [losses_human_all_events, losses_human_per_building_id]
                    )

                # Calculate timeline of recovery (to define occupants for next earthquake)
                # (only if there could be occupants for this earthquake, otherwise skip)
                if there_can_be_occupants:
                    # Calculate number of injured people still away in time
                    injured_still_away = Losses.calculate_injuries_recovery_timeline(
                        losses_human_per_asset,
                        recovery_injuries,
                        recovery_longest_time,
                        events_in_ses.loc[eq_id, "datetime"],
                    )
                    # Calculate factors regarding people being allowed to go back to buildings
                    occupancy_factors = Losses.calculate_repair_recovery_timeline(
                        recovery_damage,
                        recovery_longest_time,
                        events_in_ses.loc[eq_id, "datetime"],
                    )

                    # Store number of injured people away from the building in time, per asset ID
                    injured_still_away.to_csv(
                        os.path.join(
                            path_to_occupants_oelf,
                            "injured_still_away_after_OELF_%s.csv" % (eq_id)
                        ),
                        index=True,
                    )

                    # Store occupancy factors (0: people not allowed in, 1: people allowed in) as a
                    # function of time and damage state
                    occupancy_factors.to_csv(
                        os.path.join(
                            path_to_occupants_oelf,
                            "occupancy_factors_after_OELF_%s.csv" % (eq_id)
                        ),
                        index=True,
                    )

                # Store new exposure CSV
                if store_intermediate:
                    name_exposure_csv_file_next = "exposure_model_after_%s.csv" % (eq_id)
                    # Named after the earthquake
                    exposure_updated_damage.to_csv(
                        os.path.join(path_to_exposure, name_exposure_csv_file_next),
                        index=False,
                    )

                # Get rid of time-of-the-day-specific numbers of occupants
                exposure_updated_damage = exposure_updated_damage.drop(columns=[time_of_day])

                # Replace current exposure for this stochastic event set
                exposure_updated_damage.to_csv(
                    os.path.join(path_to_exposure, current_exposure_filename),
                    index=False,
                )

            # Get damage states per building ID for this stochastic event set (OELF realisation)
            damage_states = ExposureUpdater.summarise_damage_states_per_building_id(
                exposure_updated_damage
            )
            # Get economic losses per building ID for this stochastic event set (realisation)
            losses_economic = Losses.expected_economic_loss(
                exposure_updated_damage, consequence_economic
            )

            # Add human losses due to all events
            # (i.e. get total human losses from this stochastic event set)
            if losses_human_all_events is None:
                # None of the earthquakes of this stochastic event set were run
                # (too small or far away) --> assign zero injuries to all building IDs
                losses_human_all_events = Losses.assign_zero_human_losses(
                    original_exposure_model, list(consequence_injuries.keys())
                )
            else:
                losses_human_all_events = losses_human_all_events.groupby(["building_id"]).sum(
                    numeric_only=True
                )

            # Store damage states, economic losses and human losses per building ID
            if store_intermediate:
                damage_states.to_csv(
                    os.path.join(
                        path_to_outputs,
                        "damage_states_after_OELF_%s_realisation_%s.csv" % (
                            forecast_name, oef_ses_id
                        )
                    ),
                    index=True,
                )

                losses_economic.to_csv(
                    os.path.join(
                        path_to_outputs,
                        "losses_economic_after_OELF_%s_realisation_%s.csv" % (
                            forecast_name, oef_ses_id
                        )
                    ),
                    index=True,
                )

                losses_human_all_events.to_csv(
                    os.path.join(
                        path_to_outputs,
                        "losses_human_after_OELF_%s_realisation_%s.csv" % (
                            forecast_name, oef_ses_id
                        )
                    ),
                    index=True,
                )

            # Concatenate damage states from all stochastic event sets
            if damage_states_all_ses is None:
                # First realisation
                damage_states_all_ses = deepcopy(damage_states)
            else:
                damage_states_all_ses = pd.concat(
                    [damage_states_all_ses, damage_states]
                )

            # Concatenate economic losses from all stochastic event sets
            if losses_economic_all_ses is None:
                # First realisation
                losses_economic_all_ses = deepcopy(losses_economic)
            else:
                losses_economic_all_ses = pd.concat(
                    [losses_economic_all_ses, losses_economic]
                )

            # Concatenate human losses from all stochastic event sets
            if losses_human_all_ses is None:
                # First realisation
                losses_human_all_ses = deepcopy(losses_human_all_events)
            else:
                losses_human_all_ses = pd.concat(
                    [losses_human_all_ses, losses_human_all_events]
                )

            # Erase 'path_to_occupants_oelf' and all its contents
            # (next SES needs to start clean)
            shutil.rmtree(path_to_occupants_oelf)

        # Average damage states per building ID for all stochastic event sets
        damage_states_all_ses = damage_states_all_ses.groupby(
            ["building_id", "damage_state"]
        ).sum(numeric_only=True) / len(oef_ses_ids)

        # Average economic losses per building ID for all stochastic event sets
        losses_economic_all_ses = losses_economic_all_ses.groupby(["building_id"]).sum(
            numeric_only=True
        ) / len(oef_ses_ids)

        # Average human losses per building ID for all stochastic event sets
        losses_human_all_ses = losses_human_all_ses.groupby(["building_id"]).sum(
            numeric_only=True
        ) / len(oef_ses_ids)

        return damage_states_all_ses, losses_economic_all_ses, losses_human_all_ses

    @staticmethod
    def format_seismicity_forecast(forecast_catalogue, add_event_id=True, add_depth=False):
        """This method reformats 'forecast_catalogue' so that its fields are those required by
        the Real Time Loss Tools and its earthquakes are ordered, firstly, by stochastic event
        set (SES), and secondly in chronological order (within each SES).

        Args:
            forecast_catalogue (Pandas DataFrame):
                DataFrame containing a seismicity forecast for a period of time of interest. It
                normally consists of several realisations of seismicity, or "Stochastic Event
                Sets" (SES), each of them being identified by an SES ID. One SES ID can contain
                one or more earthquakes. The DataFrame must contain the following fields:
                    Lon or longitude (float): Longitude of the hypocentre.
                    Lat or latitude (float): Latitude of the hypocentre.
                    Mag or magnitude (float): Moment magnitude.
                    Time or datetime (Numpy datetime64): Date and time.
                    Idx.cat or catalog_id or ses_id (int): ID of the stochastic event set that
                    the earthquake belongs to.
                Optional fields are:
                    event_id (str): Unique identifier of an earthquake within a Stochastic Event
                    Set.
                    depth (float): Depth of the hypocentre.
            add_event_id (bool):
                If True, a field named "event_id" will be added to 'forecast_catalogue'. Unique IDs
                will be assigned to each earthquake within each stochastic event set (SES), in
                chronlogical order. Default: True.
            add_depth (bool):
                If True, and if "Depth" or "depth" are not already fields of 'forecast_catalogue',
                a "depth" field will be added to 'forecast_catalogue' and filled in with numpy.nan.
                Default: False.

        Returns:
            out_catalogue (Pandas DataFrame):
                Re-formatted version of 'forecast_catalogue' with the following fields:
                    longitude (float): Longitude of the hypocentre.
                    latitude (float): Latitude of the hypocentre.
                    magnitude (float): Moment magnitude.
                    datetime (Numpy datetime64): Date and time.
                    ses_id (int): ID of the stochastic event set that the
                    earthquake belongs to.
                    event_id (str): Unique identifier of an earthquake within a Stochastic Event
                    Set (only output if 'add_event_id' is True and 'forecast_catalogue' did not
                    have an 'event_id' field already).
                    depth (float): Depth of the hypocentre (only output if 'add_depth' is True).
                The index is called "EQID" and it has the format "[ses_id]-[event_id]". It is
                not created if "event_id" is missing and 'add_event_id' is False.
        """

        out_catalogue = deepcopy(forecast_catalogue)

        columns_input = forecast_catalogue.columns

        if "Lon" in columns_input:
            out_catalogue = out_catalogue.rename(columns={"Lon": "longitude"})
        if "Lat" in columns_input:
            out_catalogue = out_catalogue.rename(columns={"Lat": "latitude"})
        if "Time" in columns_input:
            out_catalogue = out_catalogue.rename(columns={"Time": "datetime"})
        if "datetime" in out_catalogue.columns:
            out_catalogue["datetime"] = pd.to_datetime(out_catalogue["datetime"])
        if "Mag" in columns_input:
            out_catalogue = out_catalogue.rename(columns={"Mag": "magnitude"})
        if "Idx.cat" in columns_input:
            out_catalogue = out_catalogue.rename(columns={"Idx.cat": "ses_id"})
        if "catalog_id" in columns_input:
            out_catalogue = out_catalogue.rename(columns={"catalog_id": "ses_id"})

        if add_depth and ("Depth" not in columns_input) and ("depth" not in columns_input):
            out_catalogue["depth"] = np.nan * np.ones_like(out_catalogue["magnitude"])

        # Order 'out_catalogue' first by SES ID and then chronologically (within each SES)
        out_catalogue = out_catalogue.sort_values(by=["ses_id", "datetime"], ignore_index=True)

        if add_event_id and "event_id" not in columns_input:
            event_ids = []

            # Identify IDs of individual realisations of seismicity
            ses_ids = out_catalogue["ses_id"].unique()

            for ses_id in ses_ids:  # Each of the seismicity forecasts
                # Earthquakes that belong only to this realisation of seismicity
                filter_realisation = (out_catalogue["ses_id"] == ses_id)
                aux = out_catalogue[filter_realisation]
                # Order the earthquakes in chronological order

                event_ids_ses = [i for i in range(1, aux.shape[0]+1)]
                event_ids = event_ids + event_ids_ses

            out_catalogue["event_id"] = event_ids

        if ("event_id" in out_catalogue.columns) and ("ses_id" in out_catalogue.columns):
            out_catalogue["EQID"] = pd.Series([
                "{:g}-{:g}".format(cat_id, ev_id)
                for (cat_id, ev_id) in zip(out_catalogue["ses_id"], out_catalogue["event_id"])])
            out_catalogue.set_index("EQID", drop=True, inplace=True)

        return out_catalogue

    @staticmethod
    def filter_forecast(
        forecast_catalogue, exposure_lons, exposure_lats, magnitude_min, distance_max
    ):
        """
        This method filters 'forecast_catalogue' to keep only earthquakes:
        (1) whose magnitude is equal to or larger than 'magnitude_min', and
        (2) whose epicentral distance to the closest exposure site (defined by 'exposure_lons'
        and 'exposure_lats') is smaller than 'distance_max'.
        The index of the output catalogue is kept as in 'forecast_catalogue'.

        Args:
            forecast_catalogue (Pandas DataFrame):
                DataFrame containing a seismicity forecast for a period of time of interest with
                at least the following fields:
                    longitude (float): Longitude of the hypocentre.
                    latitude (float): Latitude of the hypocentre.
                    magnitude (float): Moment magnitude.
            exposure_lons (float):
                Longitude of unique exposure locations.
            exposure_lats (float):
                Latitude of unique exposure locations.
            magnitude_min (float):
                Minimum earthquake magnitude.
            distance_max (float):
                Maximum epicentral distance (km).

        Returns:
            forecast_cat_filtered (Pandas DataFrame):
                Filtered version of 'forecast_catalogue', with the same column structure.
            earthquakes_kept (array of bool):
                Numpy array with length equal to the number of rows of 'forecast_catalogue',
                indicating whether each earthquake of 'forecast_catalogue' has been kept in
                'forecast_cat_filtered' (True) or not (False).
        """

        # Initialise output
        forecast_cat_filtered = deepcopy(forecast_catalogue)
        earthquakes_kept = np.array([True for i in range(forecast_cat_filtered.shape[0])])

        # Keep only earthquakes with magnitude >= magnitude_min
        magnitude_filter = (forecast_cat_filtered.magnitude >= magnitude_min)
        forecast_cat_filtered = forecast_cat_filtered[magnitude_filter]
        earthquakes_kept = magnitude_filter.to_numpy()
        logger.info(
            "%s out of %s earthquakes have magnitudes equal to or larger than the minimum %s."
            % (magnitude_filter.sum(), forecast_catalogue.shape[0], magnitude_min)
        )

        # Calculate minimum distance between each earthquake and all exposure sites
        eq_lons = forecast_cat_filtered["longitude"].to_numpy()
        eq_lats = forecast_cat_filtered["latitude"].to_numpy()
        keep = [True for i in range(forecast_cat_filtered.shape[0])]

        for i in range(forecast_cat_filtered.shape[0]):
            distances_to_all_exposure = geo.geodetic.geodetic_distance(
                eq_lons[i], eq_lats[i], exposure_lons, exposure_lats
            )
            if distances_to_all_exposure.min() > distance_max:
                keep[i] = False

        logger.info(
            "%s of those %s are located within %s km of at least one exposure asset."
            % (np.array(keep).sum(), magnitude_filter.sum(), distance_max)
        )
        forecast_cat_filtered = forecast_cat_filtered[keep]
        earthquakes_kept[earthquakes_kept == True] = np.array(keep)

        return forecast_cat_filtered, earthquakes_kept

    @staticmethod
    def can_there_be_occupants(
        forecast_catalogue, date_latest_rla, shortest_recovery_span, tolerance=0.0
    ):
        """
        This method calculates the difference between the newest date of 'forecast_catalogue'
        and 'date_latest_rla' and compares it against 'shortest_recovery_span' (+ 'tolerance').
        If the difference is smaller than 'shortest_recovery_span' (+ 'tolerance'), it returns
        False (i.e. occupants are not allowed to return to buildings before the end of the
        seismicity catalogue). If the difference is larger than 'shortest_recovery_span'
        (+ 'tolerance'), it returns True (i.e. occupants are allowed to return to buildings
        before the end of the seismicity catalogue).

        If 'date_latest_rla' it is interpreted as no "real" earthquake having been run before,
        which results in occupants being allowed in their buildings, and the method returns
        True.

        The purpose of 'tolerance' is to allow for discrepancies due to precision in the time of
        "real" earthquakes and those in the seismicity forecast.

        Args:
            forecast_catalogue (Pandas DataFrame):
                DataFrame containing a seismicity forecast for a period of time of interest. It
                must contain at least the following field:
                    datetime (Numpy datetime64): Date and time.
            date_latest_rla (datetime.datetime object):
                Date and time of the last "real" (RLA) earthquake run. Use None to indicate that
                no "real" earthquake has been run before.
            shortest_recovery_span (int):
                Shortest number of days before occupants are allowed back into buildings.
            tolerance (float):
                Tolerance for the comparison (in number of days). Default: 0.0.
        """

        if date_latest_rla is None:
            # Used to indicate no "real" earthquake has been run before
            return True

        # Newest earthquake in 'forecast_catalogue'
        #newest_date_in_forecast = pd.Timestamp(
        #    forecast_catalogue["datetime"].max()
        #).to_pydatetime()
        newest_date_in_forecast = (forecast_catalogue["datetime"].max()).to_pydatetime()

        # Difference in time
        time_diff = newest_date_in_forecast - date_latest_rla
        time_diff_days = time_diff.total_seconds()/(3600.*24.)

        if time_diff_days < (shortest_recovery_span + tolerance):
            there_can_be_occupants = False
        else:
            there_can_be_occupants = True

        return there_can_be_occupants
