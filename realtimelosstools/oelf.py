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
from openquake.hazardlib import geo
from realtimelosstools.ruptures import Rupture
from realtimelosstools.exposure_updater import ExposureUpdater
from realtimelosstools.losses import Losses
from realtimelosstools.writers import Writer
from realtimelosstools.utils import Time
from realtimelosstools.oq_handler import OpenQuakeJob


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
        forecast_model_weights,
        state_dependent,
        description_general,
        main_path,
        original_exposure_model,
        consequence_economic,
        calculate_casualties,
        consequence_injuries,
        recovery_damage,
        recovery_injuries,
        recovery_longest_time,
        time_of_day_occupancy,
        local_timezone,
        mapping_damage_states,
        store_intermediate,
        store_openquake,
        number_cores=1
    ):
        """
        This method uses OpenQuake to run an Operational Earthquake Loss Forecast (OELF) due to
        an input 'forecast_catalogue' with name 'forecast_name'. The input 'forecast_catalogue'
        may consist of one or more seismicity forecasts, which we call "stochastic event sets"
        (SES) following OpenQuake and whose ID is indicated under the field 'ses_id' of
        'forecast_catalogue'. Different SES may have been generated with different seismicity
        models, whose names, IDs and weights are specified in 'forecast_model_weights'. Each
        stochastic event set from each seismicity model is run independently, always starting
        with the exposure model stored under main_path/current/exposure_model_current.csv, which
        is copied for each SES of each seismicity model under
        main_path/exposure_models/oelf/forecast_name/exposure_model_current_oelf_MM_XX.csv,
        where "MM" is the 'model_id' and "XX" is the 'ses_id'. These
        "exposure_model_current_oelf_MM_XX.csv" files are updated after running each earthquake
        of 'model_id' and 'ses_id', to reflect the damage states of the buildings, but
        main_path/current/exposure_model_current.csv is not updated. The
        'original_exposure_model' is used to keep track of the original asset_id (i.e. ID used
        by OpenQuake, one ID per row of the exposure CSV file). Results from all stochastic
        event sets from all seismicity models (i.e. all realisations of forecast seismicity) are
        finally brought together and relevant statistics are calculated and returned.

        The method contemplates the possibility that not all desired stochastic event sets are
        contained in 'forecast_catalogue'. This may be the case, for example, if the seismicity
        forecast software only outputs earthquakes with a certain magnitude, which leads to some
        stochastic event sets being "empty". In order to consider the "empty" SES,
        'forecast_continuous_ses_numbering' needs to be set to True and the two integer values
        contained in 'forecast_ses_range' will be used to define the range of IDs of the SES. If
        the ID of a SES defined in this way is not found in 'forecast_catalogue', the effect is
        to consider that the SES does not produce any damage or losses. Either when the range of
        all SES is calculated this way ('forecast_continuous_ses_numbering'=True) or when it is
        automatically retrieved from 'forecast_catalogue' ('forecast_continuous_ses_numbering'=
        False), this method assumes that the same SES IDs apply to all seismicity forecasts.

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
                    model_id (int): ID of the seismicity model used to generate the stochastic
                        event set to which the earthquake belongs.
                    ses_id (int): ID of the stochastic event set (SES) that the earthquake
                        belongs to.
                    event_id (str): Unique identifier of an earthquake within a 'ses_id'.
                    to_run (bool): If True, the corresponding earthquake is to be
                        used to run damage/loss calculations; if False, it will be skipped.
                    there_can_be_occups (bool): True if the time difference with the previously-
                        run earthquake is such that there might be people occupying buildings
                        during this earthquake. Only relevant if 'there_can_be_occupants' and
                        'calculate_casualties' are True, not needed otherwise.
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
            forecast_model_weights (Pandas DataFrame):
                Weights of the different seismicity models used to generate the
                'forecast_catalogue'. The index is the "model_id". Its columns are:
                    model_name (str): Name of the model.
                    weight (float): Weight of the model.
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
            number_cores (int):
                Number of cores to be used for processing whenever parallelisation is possible.
                This parameter does not apply to OpenQuake, which has its own parallelisation
                strategy.

        Returns:
            All output dictionaries below have primary keys "damage", "losses_economic" and
            "losses_human" (if calculate_casualties' is True). Within "damage", there is one
            sub-key per damage state. Within "losses_economic", just one sub-key "loss". Within
            "losses_human", there is one sub-key per injury level.

            rates_exceed_mapping (Pandas DataFrame):
                Pandas DataFrame mapping the names of the columns in 'curves_by_model_portfolio'
                and 'curves_by_model_and_bdg_id' with the rates of exceedance they represent.
                    Index: names of the columns.
                    Column "rate_val": value of the rate of exceedance.
            curves_by_model_and_bdg_id (dict of dict of Pandas DataFrame):
                Dictionary with key structure as described above containing one DataFrame with
                the individual exceedance curves per model and building ID.
                    MultiIndex: (model_id, building_id)
                    Columns: Index of 'rates_exceed_mapping' (coded rates of exceedances of the
                    values contained in the DataFrame).
                (Each row is the exceedance curve for a particular building_id and model_id).
            stats_by_model_and_bdg_id (dict of dict of Pandas DataFrame):
                Dictionary with key structure as described above containing one DataFrame with
                the statistics per model and building_id.
                    MultiIndex: (model_id, building_id)
                    Columns: mean, median, p05 (5th percentile), p95 (95th percentile)
                (Each row contains the set of stats from all SES of a model, for a building_id).
            curves_by_model_portfolio (dict of dict of Pandas DataFrame):
                Dictionary with key structure as described above containing one DataFrame with
                the portfolio (i.e. all building IDs) exceedance curves per model.
                    Index: model_id
                    Columns: Index of 'rates_exceed_mapping' (coded rates of exceedances of the
                    values contained in the DataFrame).
                (Each row is the portfolio exceedance curve for a particular model_id).
            stats_by_model_portfolio (dict of dict of Pandas DataFrame):
                Dictionary with key structure as described above containing one DataFrame with
                the portfolio statistics per model.
                    Index: model_id
                    Columns: mean, median, p05 (5th percentile), p95 (95th percentile)
                (Each row contains the set of portfolio stats from all SES of a model).
            stats_by_bdg_id (dict of dict of Pandas DataFrame):
                Dictionary with key structure as described above containing one DataFrame with
                the per-building-id statistics.
                    Index: building_id
                    Columns: mean, median, p05 (5th percentile), p95 (95th percentile)
                (Each row contains the set of stats from all SES of all (weighted) models for a
                building).
            mean_median_portfolio_curves (dict of dict of Pandas DataFrame):
                Dictionary with key structure as described above containing one DataFrame with
                the weighted mean and median portfolio exceedance curves from all models
                (weights as per 'forecast_model_weights').
                    No index.
                    Columns:
                        "value": values of the type of 'value_names' (i.e., output type) for
                            which the mean/median curves have been calculated (only in the range
                            where all individual models' curves are defined).
                        "mean": weighted mean of all models.
                        "median": weighted median of all models.
            mean_damage_states (Pandas DataFrame):
                Pandas DataFrame with the damage states resulting from the average of all OELF
                realisations from all seismicity models, i.e. all stochastic event sets of
                'forecast_catalogue', reported per building_id. It has the following structure:
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
            mean_losses_economic (Pandas DataFrame):
                Pandas DataFrame with the economic losses resulting from the average of all OELF
                realisations from all seismicity models, i.e. all stochastic event sets of
                'forecast_catalogue', reported per building_id. It has the following structure:
                    Index:
                        building_id (str):
                            ID of the building.
                    Columns:
                        loss (float):
                            Expected loss for 'building_id' considering all its associated
                            building classes and damage states, with their respective
                            probabilities, as well as all stochastic event sets of seismicity.
            mean_losses_human (Pandas DataFrame):
                Pandas DataFrame with the human losses resulting from the average of all OELF
                realisations from all seismicity models, i.e. all stochastic event sets of
                'forecast_catalogue', reported per building_id. It has the following structure:
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

        # Create output sub-folders if they don't yet exist
        export_folders_to_create = [
            "exceed_curves_rates_mapping",
            "exceed_curves_by_model_and_bdg_id",
            "stats_by_model_and_bdg_id",
            "exceed_curves_by_model_portfolio",
            "stats_by_model_portfolio",
            "stats_by_bdg_id",
            "mean_median_portfolio_exceed_curves"
        ]
        for folder_name in export_folders_to_create:
            path_to_exceed_outputs = os.path.join(main_path, "output", folder_name)
            if not os.path.exists(path_to_exceed_outputs):
                os.mkdir(path_to_exceed_outputs)

        # IDs of individual realisations of seismicity (stochastic event sets, SES)
        ses_ids_in_catalogue = forecast_catalogue["ses_id"].unique()
        if forecast_continuous_ses_numbering:
            # Generate IDs based on input configuration
            oef_ses_ids = [i for i in range(forecast_ses_range[0], forecast_ses_range[1]+1)]
            # Check if there are SES IDs in 'forecast_catalogue' that are not in 'oef_ses_ids'
            existing_IDs_missing = False
            for oef_ses_id in ses_ids_in_catalogue:
                if oef_ses_id not in oef_ses_ids:
                    existing_IDs_missing = True
                    break
            if existing_IDs_missing:
                error_message = (
                    "The seismicity forecast %s contains IDs of stochastic event sets not "
                    "contemplated by the input configuration ('forecast_ses_range' parameter). "
                    "The software cannot decide what to do."
                    % (forecast_name)
                )
                logger.critical(error_message)
                raise OSError(error_message)
        else:
            # Identify IDs in the input 'forecast_catalogue'
            oef_ses_ids = deepcopy(ses_ids_in_catalogue)

        # Local time of day and classification into day, night, transit
        # (used to calculate number of occupants)
        if calculate_casualties and there_can_be_occupants:
            forecast_catalogue["local_time"] = Time.determine_local_time_from_utc(
                forecast_catalogue["datetime"],
                local_timezone
            )
            forecast_catalogue["time_of_day"] = Time.interpret_time_of_the_day(
                forecast_catalogue["local_time"]
            )
        else:
            forecast_catalogue["local_time"] = None
            forecast_catalogue["time_of_day"] = None

        # Calculate recovery occupancy factors for all earthquakes in the catalogue
        if calculate_casualties and there_can_be_occupants:
            recovery_occupancy_factors = Losses.get_occupancy_factors_all_OELF_earthquakes(
                forecast_catalogue, recovery_damage, main_path
            )
        else:
            recovery_occupancy_factors = None

        # Group 'forecast_catalogue' so that it is easier to identify each
        # (oef_model_id, oef_ses_id) subset ('catalogue_groups' is DataFrameGroupBy object)
        catalogue_groups = forecast_catalogue.groupby(["model_id", "ses_id"])

        # Initialise 'damage_states_all_ses', 'losses_economic_all_ses', 'losses_human_all_ses'
        # to gather results from all SES from all seismicity models
        damage_states_all_ses = None
        losses_economic_all_ses = None
        losses_human_all_ses = None

        precalculated_nodmg = None

        for oef_model_id in forecast_model_weights.index:  # Each of the seismicity models
            for k, oef_ses_id in enumerate(oef_ses_ids):  # Each of the stochastic event sets

                logger.info(
                    "%s Working on stochastic event set %s of %s of seismicity model ID %s"
                    % (np.datetime64('now'), k+1, len(oef_ses_ids), oef_model_id)
                )

                if calculate_casualties:
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
                # and seismicity model
                if (oef_model_id, oef_ses_id) in catalogue_groups.groups:
                    events_in_ses = catalogue_groups.get_group((oef_model_id, oef_ses_id))
                else:  # this combination of (oef_model_id, oef_ses_id) does not exist
                    # Empty DataFrame with the same structure
                    events_in_ses = forecast_catalogue.head(0)

                damage_states = None
                losses_economic = None
                losses_human_all_events = None

                if (
                    (events_in_ses.shape[0] > 0) or
                    (events_in_ses.shape[0] == 0 and precalculated_nodmg is None)
                ):
                    # Initialise exposure_model_current_XX.csv, with XX = oef_model_id_oef_ses_id
                    in_filename = os.path.join(
                        main_path, "current", "exposure_model_current.csv"
                    )  # origin
                    current_exposure_filename = "exposure_model_current_oelf_%s_%s.csv" % (
                        oef_model_id, oef_ses_id
                    )
                    out_filename = os.path.join(path_to_exposure, current_exposure_filename)
                    _ = shutil.copyfile(in_filename, out_filename)

                    # Initialise exposure, in case there are no elements in 'events_in_ses'
                    exposure_updated_damage = pd.read_csv(
                        os.path.join(path_to_exposure, current_exposure_filename),
                        dtype={"id_3": str, "id_2": str, "id_1": str}
                    )
                    exposure_updated_damage.index = exposure_updated_damage["id"]
                    exposure_updated_damage.index = exposure_updated_damage.index.rename("asset_id")
                    exposure_updated_damage = exposure_updated_damage.drop(columns=["id"])
                else:
                    # No earthquakes to run for this combination of (oef_model_id, oef_ses_id)
                    # and the no-damage outputs have already been pre-calculated
                    exposure_updated_damage = None # No need to initialise the exposure model

                # Run earthquake by earthquake of this stochastic event set
                for eq_id in events_in_ses.index:  # EQID is index of DataFrame

                    # Description
                    description = "%s, %s, event ID %s" % (
                        description_general, forecast_name, eq_id
                    )

                    # Retrieve "time_of_day"
                    time_of_day = events_in_ses.loc[eq_id, "time_of_day"]
                    # can be None if not calculate_casualties

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

                    if there_can_be_occupants and events_in_ses.loc[eq_id, "to_run"]:
                        # Update exposure to reflect occupants for this earthquake
                        # (reflecting injuries and deaths)
                        exposure_run = ExposureUpdater.update_exposure_occupants(
                            calculate_casualties,  # Nothing will change if this is False
                            exposure_full_occupants,
                            recovery_occupancy_factors[eq_id], #e.g. {"DS0": 1, ... "DS4": 0}
                            time_of_day_occupancy,
                            time_of_day,
                            events_in_ses.loc[eq_id, "datetime"],
                            True,  # include OELF previous earthquakes (not just RLA)
                            main_path,
                            number_cores
                        )
                    else:  # either there cannot be occupants or 'calculate_casualties' is False
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
                    if "model_id" in events_in_ses.columns:
                        name_rupture_file = "RUP_%s-%s-%s.xml" % (
                            events_in_ses.loc[eq_id, "model_id"],
                            events_in_ses.loc[eq_id, "ses_id"],
                            events_in_ses.loc[eq_id, "event_id"]
                        )
                    else:
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
                    oqjob = OpenQuakeJob(
                        description,
                        eq_id,
                        os.path.join(main_path, "current", "job.ini"),
                        store_intermediate,
                        store_openquake,
                        path_to_oq_outputs
                    )
                    damage_results_OQ = oqjob.run(exposure_run, mapping_damage_states)

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

                    if calculate_casualties and there_can_be_occupants:
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
                        # Calculate number of injured people still away in time
                        injured_still_away = Losses.calculate_injuries_recovery_timeline(
                            losses_human_per_asset,
                            recovery_injuries,
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

                    else: # 'calculate_casualties' is False
                        injured_still_away = None
                        losses_human_all_events = None

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

                # Calculate or retrieve pre-calculated (no damage)
                # 'damage_states', 'losses_economic', and 'losses_human_all_events'
                # (i.e. results of this (oef_model_id, oef_ses_id))
                if (
                    (events_in_ses.shape[0] > 0) or
                    ((events_in_ses.shape[0] == 0) and (precalculated_nodmg is None))
                ):
                    # Get damage states per building ID for this stochastic event set (OELF
                    # realisation) of this seismicity model
                    damage_states = ExposureUpdater.summarise_damage_states_per_building_id(
                        exposure_updated_damage
                    )
                    # Get economic losses per building ID for this stochastic event set
                    # (realisation)
                    losses_economic = Losses.expected_economic_loss(
                        exposure_updated_damage, consequence_economic
                    )

                    if calculate_casualties:
                        # Add human losses due to all events
                        # (i.e. get total human losses from this stochastic event set)
                        if losses_human_all_events is None:
                            # None of the earthquakes of this stochastic event set were run (too
                            # small or far away) --> assign zero injuries to all building IDs
                            losses_human_all_events = Losses.assign_zero_human_losses(
                                original_exposure_model, list(consequence_injuries.keys())
                            )
                        else:
                            losses_human_all_events = (
                                losses_human_all_events.groupby(["building_id"]).sum(
                                    numeric_only=True
                                )
                            )

                    if ((events_in_ses.shape[0] == 0) and (precalculated_nodmg is None)):
                        # Store this no-damage results for future model-SES with no damage
                        precalculated_nodmg = {}
                        precalculated_nodmg["damage_states"] = deepcopy(damage_states)
                        precalculated_nodmg["losses_economic"] = deepcopy(losses_economic)

                        if calculate_casualties:
                            precalculated_nodmg["losses_human_all_events"] = deepcopy(
                                losses_human_all_events
                            )

                else: # events_in_ses.shape[0] == 0 and precalculated_nodmg is NOT None
                    # Retrieve previously-calculated
                    damage_states = deepcopy(precalculated_nodmg["damage_states"])
                    losses_economic = deepcopy(precalculated_nodmg["losses_economic"])

                    if calculate_casualties:
                        losses_human_all_events = deepcopy(
                            precalculated_nodmg["losses_human_all_events"]
                        )

                # Store damage states, economic losses and human losses per building ID
                # (i.e. results of this (oef_model_id, oef_ses_id))
                if store_intermediate:
                    damage_states.to_csv(
                        os.path.join(
                            path_to_outputs,
                            "damage_states_after_OELF_%s_model_%s_realisation_%s.csv" % (
                                forecast_name, oef_model_id, oef_ses_id
                            )
                        ),
                        index=True,
                    )

                    losses_economic.to_csv(
                        os.path.join(
                            path_to_outputs,
                            "losses_economic_after_OELF_%s_model_%s_realisation_%s.csv" % (
                                forecast_name, oef_model_id, oef_ses_id
                            )
                        ),
                        index=True,
                    )

                    if calculate_casualties:
                        losses_human_all_events.to_csv(
                            os.path.join(
                                path_to_outputs,
                                "losses_human_after_OELF_%s_model_%s_realisation_%s.csv" % (
                                    forecast_name, oef_model_id, oef_ses_id
                                )
                            ),
                            index=True,
                        )

                # Concatenate damage states from all (oef_model_id, oef_ses_id) sets
                damage_states["model_id"] = [
                    oef_model_id for j in range(damage_states.shape[0])
                ]
                damage_states["ses_id"] = [
                    oef_ses_id for j in range(damage_states.shape[0])
                ]
                if damage_states_all_ses is None:
                    # First realisation
                    damage_states_all_ses = deepcopy(damage_states)
                else:
                    damage_states_all_ses = pd.concat(
                        [damage_states_all_ses, damage_states]
                    )

                # Concatenate economic losses from all (oef_model_id, oef_ses_id) sets
                losses_economic["model_id"] = [
                    oef_model_id for j in range(losses_economic.shape[0])
                ]
                losses_economic["ses_id"] = [
                    oef_ses_id for j in range(losses_economic.shape[0])
                ]
                if losses_economic_all_ses is None:
                    # First realisation
                    losses_economic_all_ses = deepcopy(losses_economic)
                else:
                    losses_economic_all_ses = pd.concat(
                        [losses_economic_all_ses, losses_economic]
                    )

                if calculate_casualties:
                    # Concatenate human losses from all (oef_model_id, oef_ses_id) sets
                    losses_human_all_events["model_id"] = [
                        oef_model_id for j in range(losses_human_all_events.shape[0])
                    ]
                    losses_human_all_events["ses_id"] = [
                        oef_ses_id for j in range(losses_human_all_events.shape[0])
                    ]
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

        # Initialise exceedance curves and stats outputs
        curves_by_model_and_bdg_id = {}
        stats_by_model_and_bdg_id = {}
        curves_by_model_portfolio = {}
        stats_by_model_portfolio = {}
        stats_by_bdg_id = {}
        mean_median_portfolio_curves = {}

        # Exceedance curves and stats: Damage states
        damage_states_all_ses_reshaped = (
            OperationalEarthquakeLossForecasting.reshape_damage_states_per_building_id(
                damage_states_all_ses
            )
        )# reshape so that each damage state is a column
        (
            rates_exceed_mapping,
            curves_by_model_and_bdg_id["damage"],
            stats_by_model_and_bdg_id["damage"],
            curves_by_model_portfolio["damage"],
            stats_by_model_portfolio["damage"],
            stats_by_bdg_id["damage"],
            mean_median_portfolio_curves["damage"]
        ) = OperationalEarthquakeLossForecasting.get_exceedance_curves_and_stats(
            damage_states_all_ses_reshaped,
            damage_states_all_ses.index.get_level_values("damage_state").unique(),
            forecast_model_weights.index,  # model IDs
            oef_ses_ids,
            forecast_model_weights
        )

        # Exceedance curves and stats: Economic losses
        (
            _,  # no need to re-retrieve 'rates_exceed_mapping', it's the same
            curves_by_model_and_bdg_id["losses_economic"],
            stats_by_model_and_bdg_id["losses_economic"],
            curves_by_model_portfolio["losses_economic"],
            stats_by_model_portfolio["losses_economic"],
            stats_by_bdg_id["losses_economic"],
            mean_median_portfolio_curves["losses_economic"]
        ) = OperationalEarthquakeLossForecasting.get_exceedance_curves_and_stats(
            losses_economic_all_ses,
            ["loss"],
            forecast_model_weights.index,  # model IDs
            oef_ses_ids,
            forecast_model_weights
        )

        # Exceedance curves and stats: Human casualties:
        if calculate_casualties:
            (
                _,  # no need to re-retrieve 'rates_exceed_mapping', it's the same
                curves_by_model_and_bdg_id["losses_human"],
                stats_by_model_and_bdg_id["losses_human"],
                curves_by_model_portfolio["losses_human"],
                stats_by_model_portfolio["losses_human"],
                stats_by_bdg_id["losses_human"],
                mean_median_portfolio_curves["losses_human"]
            ) = OperationalEarthquakeLossForecasting.get_exceedance_curves_and_stats(
                losses_human_all_ses,
                ["injuries_%s" % (severity) for severity in consequence_injuries.keys()],
                forecast_model_weights.index,  # model IDs
                oef_ses_ids,
                forecast_model_weights
            )

        # Weighted mean damage states per building ID for all stochastic event sets
        mean_damage_states = (
            OperationalEarthquakeLossForecasting.gather_mean_oelf_outputs(
                stats_by_bdg_id,
                "damage",
                damage_states_all_ses.index.get_level_values("damage_state").unique()
            )
        )

        # Weighted mean economic losses per building ID for all stochastic event sets
        mean_losses_economic = (
            OperationalEarthquakeLossForecasting.gather_mean_oelf_outputs(
                stats_by_bdg_id, "losses_economic", ["loss"]
            )
        )

        if calculate_casualties:
            # Weighted mean human losses per building ID for all stochastic event sets
            mean_losses_human = (
                OperationalEarthquakeLossForecasting.gather_mean_oelf_outputs(
                    stats_by_bdg_id,
                    "losses_human",
                    ["injuries_%s" % (severity) for severity in consequence_injuries.keys()]
                )
            )
        else:
            mean_losses_human = None

        return (
            rates_exceed_mapping,
            curves_by_model_and_bdg_id,
            stats_by_model_and_bdg_id,
            curves_by_model_portfolio,
            stats_by_model_portfolio,
            stats_by_bdg_id,
            mean_median_portfolio_curves,
            mean_damage_states,
            mean_losses_economic,
            mean_losses_human
        )

    @staticmethod
    def get_exceedance_curves_and_stats(
        all_ses_output, value_names, oef_model_ids, oef_ses_ids, model_weights
    ):
        """
        This method calculates:
        - exceedance curves by model_id and building_id (each curve is built from all stochastic
        event sets of the model);
        - statistics (mean, median, 5th and 95th percentile) of all stochastic event sets by
        model_id and building_id;
        - portfolio (i.e., all building IDs summed up) exceedance curves by model_id (each curve
        is built from all stochastic event sets of the model);
        - portfolio statistics (mean, median, 5th and 95th percentile) of all stochastic event
        sets by model_id;
        - weighted mean and weighted median portfolio exceedance curves (mean and median of all
        models), calculated in the range (min_val, max_val), with min_val being the maximum
        across all models of the minimum output value, and max_val being the minimum across all
        models of the maximum output value (i.e., mean/median only calculated in the range where
        all curves are defined); the number of points in the curve is the number of stochastic
        event sets, i.e., len(oef_ses_ids), but always larger than 10.

        Args:
            all_ses_output (Pandas DataFrame):
                DataFrame with the output values from all SES and all models, with the following
                structure:
                    Index: building_id
                    Columns:
                        - At least those indicated by 'value_names'
                        - "ses_id": ID of the stochastic event set (SES)
                        - "model_id": ID of the seismicity model
                    Each row of this DataFrame contains an output value for a particular
                    building_id, SES ID and seismicity model ID.
            value_names (list of str):
                Names of the columns of 'all_ses_output' that contain values of relevance (e.g.
                "loss", "injuries_X").
            oef_model_ids (list of int):
                IDs of the seismicity models.
            oef_ses_ids (list of int):
                IDs of the stochastic event sets.
            model_weights (Pandas DataFrame):
                Weights of the different seismicity models used to generate the
                'forecast_catalogue'. The index is the "model_id". Its columns are:
                    model_name (str): Name of the model.
                    weight (float): Weight of the model.

        Returns:
            rates_exceed_mapping (Pandas DataFrame):
                Pandas DataFrame mapping the names of the columns in 'curves_by_model_portfolio'
                and 'curves_by_model_and_bdg_id' with the rates of exceedance they represent.
                    Index: names of the columns.
                    Column "rate_val": value of the rate of exceedance.
            curves_by_model_and_bdg_id (dict of Pandas DataFrame):
                Dictionary with one key per element of 'value_names' (i.e., output type),
                containing one DataFrame with the individual exceedance curves per model and
                building ID.
                    MultiIndex: (model_id, building_id)
                    Columns: Index of 'rates_exceed_mapping' (coded rates of exceedances of the
                    values contained in the DataFrame).
                (Each row is the exceedance curve for a particular building_id and model_id).
            stats_by_model_and_bdg_id (dict of Pandas DataFrame):
                Dictionary with one key per element of 'value_names' (i.e., output type),
                containing one DataFrame with the statistics per model and building_id.
                    MultiIndex: (model_id, building_id)
                    Columns: mean, median, p05 (5th percentile), p95 (95th percentile)
                (Each row contains the set of stats from all SES of a model, for a building_id).
            curves_by_model_portfolio (dict of Pandas DataFrame):
                Dictionary with one key per element of 'value_names' (i.e., output type),
                containing one DataFrame with the portfolio (i.e. all building IDs) exceedance
                curves per model.
                    Index: model_id
                    Columns: Index of 'rates_exceed_mapping' (coded rates of exceedances of the
                    values contained in the DataFrame).
                (Each row is the portfolio exceedance curve for a particular model_id).
            stats_by_model_portfolio (dict of Pandas DataFrame):
                Dictionary with one key per element of 'value_names' (i.e., output type),
                containing one DataFrame with the portfolio statistics per model.
                    Index: model_id
                    Columns: mean, median, p05 (5th percentile), p95 (95th percentile)
                (Each row contains the set of portfolio stats from all SES of a model).
            stats_by_bdg_id (dict of Pandas DataFrame):
                Dictionary with one key per element of 'value_names' (i.e., output type),
                containing one DataFrame with the per-building-id statistics.
                    Index: building_id
                    Columns: mean, median, p05 (5th percentile), p95 (95th percentile)
                (Each row contains the set of stats from all SES of all (weighted) models for a
                building).
            mean_median_portfolio_curves (dict of Pandas DataFrame):
                Dictionary with one key per element of 'value_names' (i.e., output type),
                containing one DataFrame with the weighted mean and median portfolio exceedance
                curves from all models (weights as per 'model_weights').
                    No index.
                    Columns:
                        "value": values of the type of 'value_names' (i.e., output type) for
                            which the mean/median curves have been calculated (only in the range
                            where all individual models' curves are defined).
                        "mean": weighted mean of all models.
                        "median": weighted median of all models.
        """

        number_ses = float(len(oef_ses_ids))

        rates_exceed = np.arange(number_ses, 0, -1)/number_ses
        rates_exceed_names = ["r%s" % (i+1) for i in range(int(number_ses))]
        rates_exceed_mapping = pd.DataFrame({"rate_val": rates_exceed}, index=rates_exceed_names)

        curves_by_model_and_bdg_id = {}
        stats_by_model_and_bdg_id = {}
        curves_by_model_portfolio = {}
        stats_by_model_portfolio = {}
        stats_by_bdg_id = {}
        mean_median_portfolio_curves = {}

        for val_col in value_names:
            # Curve per model and building ID
            # MultiIndex("model_id", "building_id"), 1 col per rate of exceedance (from 1 to 1/N)
            # Each row is one curve, the column names are the rates of exceedance
            aux = all_ses_output.groupby(["model_id", "building_id"])[val_col].apply(np.sort)
            curves_by_model_and_bdg_id[val_col] = pd.DataFrame(
                aux.tolist(),
                index=aux.index,
                columns=rates_exceed_names
            )
            stats_by_model_and_bdg_id[val_col] = curves_by_model_and_bdg_id[val_col].apply(
                lambda row: pd.Series({
                    "mean": row.mean(),
                    "median": row.median(),
                    "p05": np.percentile(row, 5),
                    "p95": np.percentile(row, 95),
                }),
                axis=1
            )

            # Curve per model
            # Index: model_id, 1 col per rate of exceedance (from 1 to 1/N)
            # Each row is one curve, the column names are the rates of exceedance
            curves_by_model_portfolio_aux = curves_by_model_and_bdg_id[val_col].groupby(
                ["model_id"]
            ).agg(
                "sum", numeric_only=True
            )
            curves_by_model_portfolio[val_col] = curves_by_model_portfolio_aux
            stats_by_model_portfolio[val_col] = curves_by_model_portfolio_aux.apply(
                lambda row: pd.Series({
                    "mean": row.mean(),
                    "median": row.median(),
                    "p05": np.percentile(row, 5),
                    "p95": np.percentile(row, 95),
                }),
                axis=1
            )

            # Mean and median curves of all models
            min_val = curves_by_model_portfolio_aux[curves_by_model_portfolio_aux.columns[0]].max()
            max_val = curves_by_model_portfolio_aux[curves_by_model_portfolio_aux.columns[-1]].min()
            target_vals = np.linspace(min_val, max_val, num=max(int(number_ses), 10))
            # Interpolate the curve of each model to the target vals along the horizontal axis
            interp_df = curves_by_model_portfolio_aux.apply(
                lambda row: np.interp(
                    target_vals,
                    row.to_numpy(),
                    rates_exceed
                ),
                axis=1,
                result_type="expand"
            )
            interp_df.columns = target_vals
            # Calculate weighted means
            weighted_means = np.average(
                interp_df.to_numpy(),
                axis=0,  # per column
                weights=model_weights.loc[interp_df.index, "weight"].to_numpy()
            )
            # Calculate weighted medians
            weighted_medians = np.quantile(
                interp_df.to_numpy(),
                0.5,
                axis=0,  # per column
                weights=model_weights.loc[interp_df.index, "weight"].to_numpy(),
                method="inverted_cdf"
            )
            # Gather weighted means and medians in one output DataFrame
            mean_median_portfolio_curves[val_col] = pd.DataFrame(
                {
                    "value": interp_df.columns.to_numpy(),
                    "mean": weighted_means,
                    "median": weighted_medians
                }
            )

            # Stats by building_id (all models together, weighted)
            ses_weight = 1.0 / number_ses
            weight_from_models = model_weights.loc[
                all_ses_output.loc[:, "model_id"], "weight"
            ].to_numpy()
            all_ses_output["weight"] = weight_from_models * ses_weight

            # Create numpy arrays with one column per building_id and one row per value/weight
            values_per_building_id_df = all_ses_output.pivot_table(
                index=["ses_id", "model_id"],
                columns="building_id",
                values=val_col
            )
            building_ids_per_col = values_per_building_id_df.columns
            values_per_building_id = values_per_building_id_df.values

            weights_per_building_id_df = all_ses_output.pivot_table(
                index=["ses_id", "model_id"],
                columns="building_id",
                values="weight"
            )
            building_ids_per_col_aux = weights_per_building_id_df.columns
            weights_per_building_id = weights_per_building_id_df.values

            if not building_ids_per_col.equals(building_ids_per_col_aux):
                error_message = (
                    "'get_exceedance_curves_and_stats': the order of 'building_id' is not the "
                    "same when grouping to calculate by-building-id stats"
                )
                logger.critical(error_message)
                raise ValueError(error_message)

            if not (
                np.all(weights_per_building_id.sum(axis=0)>0.99999) and
                np.all(weights_per_building_id.sum(axis=0)<1.00001)
            ):
                error_message = (
                    "'get_exceedance_curves_and_stats': the sum of weights per building is not 1"
                )
                logger.critical(error_message)
                raise ValueError(error_message)

            del building_ids_per_col_aux

            weighted_means = np.average(
                values_per_building_id,
                axis=0,  # per column
                weights=weights_per_building_id
            )  # each col is a building_id as per 'building_ids_per_col'
            weighted_percentiles = np.quantile(
                values_per_building_id,
                [0.05, 0.5, 0.95], # percentiles
                axis=0,  # per column
                weights=weights_per_building_id,
                method="inverted_cdf"
            )  # each row is one of the percentiles, each col a building_id

            # Gather weighted stats per-building-id in one output DataFrame
            stats_by_bdg_id[val_col] = pd.DataFrame(
                {
                    "building_id": building_ids_per_col,
                    "mean": weighted_means,
                    "median": weighted_percentiles[1, :],
                    "p05": weighted_percentiles[0, :],
                    "p95": weighted_percentiles[2, :],
                }
            )
            stats_by_bdg_id[val_col].set_index(
                stats_by_bdg_id[val_col]["building_id"], inplace=True
            )
            stats_by_bdg_id[val_col].drop(columns=["building_id"], inplace=True)

        return (
            rates_exceed_mapping,
            curves_by_model_and_bdg_id,
            stats_by_model_and_bdg_id,
            curves_by_model_portfolio,
            stats_by_model_portfolio,
            stats_by_bdg_id,
            mean_median_portfolio_curves
        )

    @staticmethod
    def reshape_damage_states_per_building_id(damage_summary_with_ids):
        """
        This method reshapes the 'damage_summary_with_ids' so that each damage state is a
        column. The input 'damage_summary_with_ids' is almost the same as the ouput of method
        ExposureUpdater.summarise_damage_states_per_building_id(), plus two columns: one for the
        ID of the stochastic event set, another one for the ID of the seismicity model.

        Args:
            damage_summary_with_ids (Pandas DataFrame):
                Pandas DataFrame with the following structure:
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
                        ses_id (int):
                            ID of the stochastic event set (SES).
                        model_id (int):
                            ID of the seismicity model.

        Returns:
            damage_summary_reshaped (Pandas DataFrame):
                Pandas DataFrame with the exact same contents as the input
                'damage_summary_with_ids' but the following structure:
                    Index:
                        building_id (str):
                            ID of the building.
                    Columns:
                        ses_id (int):
                            ID of the stochastic event set (SES).
                        model_id (int):
                            ID of the seismicity model.
                        One column per possible value of 'damage_state', with the contents of
                        'number'.
        """
        dmg_by_cols_df = (
            damage_summary_with_ids
            .reset_index()
            .pivot_table(
                index=["building_id", "ses_id", "model_id"],
                columns="damage_state",
                values="number"
            )
            .reset_index()
        ).fillna(0.0)
        dmg_by_cols_df.rename_axis(columns=None, inplace=True)
        dmg_by_cols_df.set_index(dmg_by_cols_df["building_id"], inplace=True)
        dmg_by_cols_df.drop(columns=["building_id"], inplace=True)

        return dmg_by_cols_df

    @staticmethod
    def gather_mean_oelf_outputs(stats_by_bdg_id, out_type, value_names):
        """
        This method gathers the mean results of type 'out_type' and sub-types as listed in
        'value_names' into one Pandas DataFrame with the following structure:

        - If 'out_type' is "losses_economic" or "losses_human", the index is "building_id" (as
        in the input DataFrames in 'stats_by_bdg_id') and each column contains the mean values
        of 'value_names' (i.e., one column per sub-type listed in 'value_names').
        - If 'out_type' is "damage", the index is multiple ["building_id", "damage_state"] and
        there is only one "number" column.

        Args:
            stats_by_bdg_id (dict of dict of Pandas DataFrame):
                Dictionary with structure[out_type][value_name] containing one DataFrame with
                per-building-id statistics.
                    Index: building_id
                    Columns: mean, median, p05 (5th percentile), p95 (95th percentile)
                (Each row contains the set of stats from all SES of all (weighted) models for a
                building).
            out_type (str):
                Type of output as first-level key of 'stats_by_bdg_id'.
            value_names (list of str):
                Output-subtype as second-level key of 'stats_by_bdg_id[out_type]'. For example,
                if "out_type" is "damage", the subtypes may be the damage states
                ["DS0", "DS1", ...].

        Returns:
            mean_oelf_outputs (Pandas DataFrame): As described above.
        """
        mean_oelf_outputs = None

        for val_col in value_names:
            aux_df = stats_by_bdg_id[out_type][val_col].copy()
            aux_df.drop(columns=["median", "p05", "p95"], inplace=True)
            aux_df.rename(columns={"mean": val_col}, inplace=True)

            if mean_oelf_outputs is None:
                mean_oelf_outputs = deepcopy(aux_df)
            else:
                mean_oelf_outputs = pd.concat((mean_oelf_outputs, aux_df), axis=1)

        if out_type == "damage":
            mean_oelf_outputs = (
                mean_oelf_outputs
                  .stack()
                  .rename_axis(["building_id", "damage_state"])
                  .reset_index(name="number")
            )
            mean_oelf_outputs.set_index(["building_id", "damage_state"], inplace=True)

        return mean_oelf_outputs

    @staticmethod
    def format_seismicity_forecast(
        forecast_catalogue, add_event_id=True, add_model_id=True
    ):
        """This method reformats 'forecast_catalogue' so that its fields are those required by
        the Real Time Loss Tools and its earthquakes are ordered, firstly, by seismicity model
        ID (if available), then by ID of the stochastic event set (SES) and, finally, in
        chronological order (within each SES of each model ID).

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
                    model_id (int): ID of the seismicity model used to generate the earthquake.
                    depth (float): Depth of the hypocentre (km).
                    strike (float): Strike of the rupture. Only used to generate point sources
                        (not planar ruptures).
                    dip (float): Dip of the rupture. Only used to generate point sources (not
                        planar ruptures).
                    rake (float): Slip direction of the fault plane. Only used to generate point
                        sources (not planar ruptures).
            add_event_id (bool):
                If True, a field named "event_id" will be added to 'forecast_catalogue'. Unique IDs
                will be assigned to each earthquake within each stochastic event set (SES), in
                chronlogical order. Default: True.
            add_model_id (bool):
                If True, a field named "model_id" will be added to 'forecast_catalogue'. All
                "model_id" values will be 1, as it will be assumed that all earthquakes have
                been generated with the same seismicity model. Default: True.

        Returns:
            out_catalogue (Pandas DataFrame):
                Re-formatted version of 'forecast_catalogue' with the following fields:
                    longitude (float): Longitude of the hypocentre.
                    latitude (float): Latitude of the hypocentre.
                    magnitude (float): Moment magnitude.
                    datetime (Numpy datetime64): Date and time.
                    model_id (int): ID of the seismicity model used to generate the stochastic
                        event set to which the earthquake belongs (only output if 'add_model_id'
                        is True or if 'forecast_catalogue' had a 'model_id' field already).
                    ses_id (int): ID of the stochastic event set that the earthquake belongs to.
                    event_id (str): Unique identifier of an earthquake within a Stochastic Event
                        Set (only output if 'add_event_id' is True or if 'forecast_catalogue'
                        had an 'event_id' field already).
                    depth (float): Depth of the hypocentre (only if present in the input).
                If "event_id" exists or 'add_event_id' is true, "out_catalogue" is assigned an
                index called "EQID". If "model_id" exists or 'add_model_id' is True, the format
                of the index is "[model_id]-[ses_id]-[event_id]". If "model_id" does not exist
                and 'add_model_id' is False, the format of the index is "[ses_id]-[event_id]".
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

        if add_model_id and "model_id" not in columns_input:
            # All earthquakes assumed to stem from the same seismicity model,
            # hereafter labelled "1"
            out_catalogue["model_id"] = np.ones(out_catalogue.shape[0], dtype=int)

        # Order 'out_catalogue' first by model ID (if available), then by SES ID and then
        # chronologically (within each SES of each model ID)
        if "model_id" in out_catalogue.columns:
            out_catalogue = out_catalogue.sort_values(
                by=["model_id", "ses_id", "datetime"], ignore_index=True
            )
        else:
            out_catalogue = out_catalogue.sort_values(
                by=["ses_id", "datetime"], ignore_index=True
            )

        if add_event_id and "event_id" not in columns_input:
            event_ids = []  # to collect all event IDs from all models and SESs

            # Identify IDs of individual realisations of seismicity (numbering of
            # events occurs within each unique combi of model, if available, and SES)
            if "model_id" in out_catalogue.columns:
                unique_model_ses_pairs = (
                    out_catalogue[["model_id", "ses_id"]]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
            else:
                unique_model_ses_pairs = (
                    out_catalogue[["ses_id"]].drop_duplicates().reset_index(drop=True)
                )

            # Create filter per unique identified pair of model-SES
            # (to select slices of "out_catalogue")
            cols_to_consider = list(unique_model_ses_pairs.columns)
            filters = [
                out_catalogue[cols_to_consider].eq(row).all(axis=1)
                for _, row in unique_model_ses_pairs.iterrows()
            ]  # list with one element per row of "unique_model_ses_pairs"

            for unique_pair_filter in filters:
                # Earthquakes that belong only to this realisation of seismicity & model
                aux = out_catalogue[unique_pair_filter]

                # Create event IDs for this realisation of seismicity & model
                event_ids_ses = [i for i in range(1, aux.shape[0]+1)]
                event_ids = event_ids + event_ids_ses  # concatenate

            out_catalogue["event_id"] = event_ids

        if ("event_id" in out_catalogue.columns) and ("ses_id" in out_catalogue.columns):
            if "model_id" in out_catalogue.columns:
                out_catalogue["EQID"] = pd.Series([
                    "{:g}-{:g}-{:g}".format(mod_id, cat_id, ev_id)
                    for (mod_id, cat_id, ev_id) in zip(
                        out_catalogue["model_id"],
                        out_catalogue["ses_id"],
                        out_catalogue["event_id"]
                    )
                ])
            else:
                out_catalogue["EQID"] = pd.Series([
                    "{:g}-{:g}".format(cat_id, ev_id)
                    for (cat_id, ev_id) in zip(
                        out_catalogue["ses_id"], out_catalogue["event_id"]
                    )
                ])

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

        If 'date_latest_rla' is None it is interpreted as no "real" earthquake having been run
        before, which results in occupants being allowed in their buildings, and the method
        returns True.

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
        newest_date_in_forecast = (forecast_catalogue["datetime"].max()).to_pydatetime()

        # Difference in time
        time_diff = newest_date_in_forecast - date_latest_rla
        time_diff_days = time_diff.total_seconds()/(3600.*24.)

        if time_diff_days < (shortest_recovery_span + tolerance):
            there_can_be_occupants = False
        else:
            there_can_be_occupants = True

        return there_can_be_occupants

    @staticmethod
    def can_there_be_occupants_by_earthquake(
        forecast_catalogue, date_latest_rla, shortest_recovery_span, tolerance=0.0
    ):
        """
        This method calculates the time difference between successive earthquakes and compares
        it against 'shortest_recovery_span' to determine whether occupants might be able to
        return to their buildings in between each pair of successive events.

        The definition of "successive earthquakes" accounts for the following considerations:
        - 'forecast_catalogue' is expected to contain several stochastic event sets generated
        with different seismicity models;
        - within each stochastic event set of each model, earthquakes are expected to be in
        chronological order;
        - each stochastic event set has its own "first" earthquake;
        - for each of these "first" earthquakes, the "previous" earthquake is the last-run RLA
        earthquake, whose date is 'date_latest_rla';
        - for the rest of the earthquakes, the "previous" earthquake is the one right before it;
        - only earthquakes for which forecast_catalogue["to_run"]==True are considered, but the
        output contains all earthquake IDs;
        - the "first" earthquake of a stochastic event set may be a "second" earthquake at first
        sight, if the "first" earthquake has forecast_catalogue["to_run"]==False (because if an
        earthquake is not run it is assumed that it does not cause further damage and thus the
        inspection/repair timeline is not triggered by it).

        If 'date_latest_rla' is None it is interpreted as no "real" earthquake having been run
        before, which results in occupants being allowed in their buildings.

        The purpose of 'tolerance' is to allow for discrepancies due to precision in the time of
        "real" earthquakes and those in the seismicity forecast.

        Args:
            forecast_catalogue (Pandas DataFrame):
                DataFrame containing a seismicity forecast for a period of time of interest. It
                must contain at least the following field:
                    datetime (Numpy datetime64): Date and time.
                    to_run (bool)
            date_latest_rla (datetime.datetime object):
                Date and time of the last "real" (RLA) earthquake run. Use None to indicate that
                no "real" earthquake has been run before.
            shortest_recovery_span (int):
                Shortest number of days before occupants are allowed back into buildings.
            tolerance (float):
                Tolerance for the comparison (in number of days). Default: 0.0.

        Returns:
            by_eq_there_can_be_occupants (Pandas Series of bool):
                Series with same index as 'forecast_catalogue' with:
                    - True, if the time difference with respect to the previous earthquake is
                    such that occupants might return for at least some damage states,
                    - False, if the time difference with respect to the previous earthquake is
                    such that no occupants can be expected in any building.
        """

        # Discard earthquakes that do not cause damage (i.e. too small or too far away)
        forecast_catalogue_to_run = forecast_catalogue[forecast_catalogue.to_run]

        catalogue_groups = forecast_catalogue_to_run.groupby(["model_id", "ses_id"])

        # Difference between successive earthquakes of each ("model_id", "ses_id") set
        dt_diff = (
            catalogue_groups["datetime"].diff()  # difference between successive rows
            .dt.total_seconds() / (3600. * 24.)  # convert to days
        )  # first rows will be NaN because there is no previous earthquake

        if date_latest_rla is not None:
            # First-row replacement: datetime - date_latest_rla
            first_row_diff = (
                (forecast_catalogue_to_run["datetime"] - date_latest_rla)
                .dt.total_seconds() / (3600 * 24)
            )  # computed for all earthquakes, but we take only the first rows later

        else:  # date_latest_rla is None, no previous RLA earthquake has been run
            first_row_diff = pd.Series(99999., index=forecast_catalogue_to_run.index)

        # Fill NaNs (the first rows of each set)
        dt_diff_days_all_eqs = dt_diff.fillna(first_row_diff)  # Pandas Series

        # True when the time difference is larger than the shortest recovery time
        threshold = shortest_recovery_span + tolerance
        aux_by_eq_there_can_be_occupants = (dt_diff_days_all_eqs > threshold)

        # Initialise output (earthquakes for which "to_run" is False were excluded
        # up to here but need to exist in the final output)
        by_eq_there_can_be_occupants = pd.Series(
            [False for i in range(forecast_catalogue.shape[0])],
            index=forecast_catalogue.index
        )
        by_eq_there_can_be_occupants[
            aux_by_eq_there_can_be_occupants.index
        ] = aux_by_eq_there_can_be_occupants

        return by_eq_there_can_be_occupants
