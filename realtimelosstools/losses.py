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
import glob
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from realtimelosstools.utils import MultilinearStepFunction


logger = logging.getLogger()


class Losses:
    """This class handles methods associated with calculating economic and human losses.
    """

    @staticmethod
    def expected_economic_loss(exposure, consequence_model):
        """
        This method returns the expected economic loss per building_id, where building_id refers
        to an individual building or a group of buildings.

        Args:
            exposure (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake. It
                comprises at least the following fields:
                    id (str):
                        ID of the asset (i.e. specific combination of building_id and a
                        particular building class and damage state).
                    building_id (str):
                        ID of the building. One building_id can be associated with different
                        values of asset_id.
                    original_asset_id (str):
                        ID of the asset in the initial undamaged version of the exposure model.
                    lon (float):
                        Longitude of the asset in degrees.
                    lat (float):
                        Latitude of the asset in degrees.
                    taxonomy (str):
                        Building class.
                    number (float):
                        Number of buildings in this asset (or probability of this particular
                        combination of building class and damage state for this building_id).
                    structural (float):
                        Total replacement costs of this asset (all buildings in "number").
                    occupancy (str):
                        "Res" (residential), "Com" (commercial) or "Ind" (industrial).
            consequence_model (Pandas DataFrame):
                Pandas DataFrame with the consequence model for economic losses in terms of mean
                values of loss ratio per damage state. Each row in the 'consequence_model'
                corresponds to a different building class. The structure is as follows:
                    Index:
                        Taxonomy (str): Building classes.
                    Columns:
                        One per damage state (float): They contain the mean loss ratios (as
                        percentages) for each building class and damage state.
        Returns:
            loss_summary (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        building_id (str):
                            ID of the building.
                    Columns:
                        loss (float):
                            Expected loss for 'building_id' considering all its associated
                            building classes and damage states, with their respective
                            probabilities.
        """

        # Initialise output
        loss_summary = deepcopy(exposure)
        loss_summary = loss_summary.drop(
            columns=["id", "lon", "lat", "occupancy", "original_asset_id"]
        )

        # Create separate columns for building class and damage state
        taxonomy = loss_summary["taxonomy"].to_numpy()
        loss_summary["damage_state"] = [
            taxonomy[i].split("/")[-1] for i in range(loss_summary.shape[0])
        ]
        loss_summary["building_class"] = [
            "/".join(taxonomy[i].split("/")[:-1]) for i in range(loss_summary.shape[0])
        ]

        # Join the 'loss_summary' with the 'consequence_model'
        loss_summary = loss_summary.join(consequence_model, on="building_class")

        # Calculate the losses
        losses = np.zeros([loss_summary.shape[0]])
        for i, row in enumerate(loss_summary.index):
            loss_ratio = loss_summary.loc[row, loss_summary.loc[row, "damage_state"]] / 100.0
            losses[i] = loss_ratio * loss_summary.loc[row, "structural"]
        loss_summary["loss"] = losses

        loss_summary = loss_summary.groupby(["building_id"]).sum(numeric_only=True)

        loss_summary = loss_summary[["loss"]]

        return loss_summary

    @staticmethod
    def expected_human_loss_per_asset_id(exposure, time_of_day, consequence_model):
        """
        This method returns the expected human loss per ID of the asset in 'exposure', i.e. one
        by one the rows of 'exposure', as per the damage states therein specified and the human
        loss ratios dicated by 'consequence_model' for a specific 'time_of_day'.

        Args:
            exposure (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake. It
                comprises at least the following fields:
                    id (str):
                        ID of the asset (i.e. specific combination of building_id and a
                        particular building class and damage state).
                    building_id (str):
                        ID of the building. One building_id can be associated with different
                        values of asset_id.
                    original_asset_id (str):
                        ID of the asset in the initial undamaged version of the exposure model.
                    lon (float):
                        Longitude of the asset in degrees.
                    lat (float):
                        Latitude of the asset in degrees.
                    taxonomy (str):
                        Building class.
                    number (float):
                        Number of buildings in this asset (or probability of this particular
                        combination of building class and damage state for this building_id).
                    night, day, transit, census (float):
                        Total number of occupants in this asset at different times of the
                        day (night, day, transit) and irrespective of the time of the day
                        (census).
                    occupancy (str):
                        "Res" (residential), "Com" (commercial) or "Ind" (industrial).
            time_of_day (str):
                Time of the day at which the earthquake occurs: "day", "night" or "transit".
            consequence_model (dict of Pandas DataFrame):
                Dictionary whose keys are the injury severity levels and whose contents are
                Pandas DataFrames with the consequence models for injuries in terms of mean
                values of loss ratio per damage state. Each row in the consequence model
                corresponds to a different building class. The structure is as follows:
                    Index:
                        Taxonomy (str): Building classes.
                    Columns:
                        One per damage state (float): They contain the mean loss ratios (as
                        percentages) for each building class and damage state.
        Returns:
            losses_per_asset (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        id (str):
                            ID of the asset as in the 'id' column of input 'exposure'.
                    Columns:
                        taxonomy (str):
                            Building class.
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of original_asset_id and id.
                        injuries_X (float):
                            Expected injuries of severity X for 'id'.
        """

        # Initialise output
        losses_per_asset = deepcopy(exposure)

        # Create separate columns for building class and damage state
        taxonomy = losses_per_asset["taxonomy"].to_numpy()
        losses_per_asset["building_class"] = [
            "/".join(taxonomy[i].split("/")[:-1]) for i in range(losses_per_asset.shape[0])
        ]
        losses_per_asset["damage_state"] = [
            taxonomy[i].split("/")[-1] for i in range(losses_per_asset.shape[0])
        ]

        injuries_columns = []

        for severity in consequence_model:
            losses_per_asset_aux = deepcopy(losses_per_asset)

            # Join the 'losses_per_asset_aux' with the consequence model
            losses_per_asset_aux = losses_per_asset_aux.join(
                consequence_model[severity], on="building_class"
            )

            # Calculate the losses
            losses = np.zeros([losses_per_asset_aux.shape[0]])
            for i, row in enumerate(losses_per_asset_aux.index):
                loss_ratio = (
                    losses_per_asset_aux.loc[row, losses_per_asset_aux.loc[row, "damage_state"]]
                    / 100.0
                )
                losses[i] = loss_ratio * losses_per_asset_aux.loc[row, time_of_day]

            losses_per_asset["injuries_%s" % (severity)] = losses
            injuries_columns.append("injuries_%s" % (severity))

        losses_per_asset.set_index(
            losses_per_asset["id"], drop=True, inplace=True
        )

        losses_per_asset = losses_per_asset[
            ["taxonomy", "original_asset_id", "building_id", *injuries_columns]
        ]

        return losses_per_asset

    @staticmethod
    def expected_human_loss_per_building_id(human_losses_per_asset):
        """
        This method returns the expected human loss per building ID, starting from expected
        human losses per asset (the output of method "expected_human_loss_per_asset_id").

        Args:
            human_losses_per_asset (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        id (str):
                            ID of the asset as in the 'id' column of input 'exposure'.
                    Columns:
                        taxonomy (str):
                            Building class.
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of original_asset_id and id.
                        injuries_X (float):
                            Expected injuries of severity X for '_id'.
        Returns:
            loss_summary (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        building_id (str):
                            ID of the building.
                    Columns:
                        injuries_X (float):
                            Expected injuries of severity X for 'building_id', considering all
                            its associated building classes and damage states, with their
                            respective probabilities.
        """

        loss_summary = human_losses_per_asset.groupby(["building_id"]).sum(numeric_only=True)

        return loss_summary

    @staticmethod
    def assign_zero_human_losses(exposure, severity_levels):
        """
        This method assigns zero human losses to each value of 'building_id' for each injury
        severity level of 'severity_levels'.

        Args:
            exposure (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake. It
                comprises at least the following field:
                    building_id (str):
                        ID of the building.
            severity_levels (list of str):
                List with the scale of severity of injuries. E.g. ["1","2","3","4"].
        Returns:
            zero_loss_summary (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        building_id (str):
                            ID of the building.
                    Columns:
                        injuries_X (float):
                            Expected (zero) injuries of severity X for 'building_id'.
        """

        building_ids = exposure["building_id"].unique()

        zero_loss_summary = pd.DataFrame(index=building_ids)
        zero_loss_summary.index.name = "building_id"

        for severity in severity_levels:
            zero_loss_summary["injuries_%s" % (severity)] = np.zeros([zero_loss_summary.shape[0]])

        return zero_loss_summary

    @staticmethod
    def define_timeline_recovery_relative(timeline_raw, shortest_time, longest_time):
        """
        This method defines the points in time to be used for the updating of the occupants of
        buildings, starting from a 'timeline_raw' and making sure 'shortest_time' and
        'longest_time' are included. All three parameters are assumed to refer to number of days
        since the last earthquake (i.e. they are relative values). All three parameters are
        assumed to be integers and will be transformed to integers if they are not.

        Args:
            timeline_raw (arr of int):
                Number of days since the last earthquake that mark different stages of
                inspection or repair for different damage states and/or recovery of patients in
                hospital.
            shortest_time (int):
                Lower bound for the output.
            longest_time (int):
                Upper bound for the output.

        Returns:
            timeline (arr of int):
                Number of days since the last earthquake that mark different stages of recovery,
                defined as described above.

        Examples:
            1) timeline_raw = np.array([5, 5, 365, 1095, 1095])
               shortest_time = 0
               longest_time = 730

               Result: timeline = np.array([0, 5, 365, 730])

            2) timeline_raw = np.array([5, 5, 365, 1095, 1095])
               shortest_time = 0
               longest_time = 3650

               Result: timeline = np.array([0, 5, 365, 1095, 3650])

            3) timeline_raw = np.array([5, 5, 10, 365, 1095, 1095])
               shortest_time = 10
               longest_time = 3650

               Result: timeline = np.array([10, 365, 1095, 3650])
        """

        timeline = np.unique(timeline_raw.astype(int))
        timeline = timeline[timeline >= shortest_time]
        timeline = timeline[timeline <= longest_time]
        if shortest_time not in timeline:
            timeline = np.append(timeline, shortest_time)
        if longest_time not in timeline:
            timeline = np.append(timeline, longest_time)
        timeline.sort()

        return timeline

    @staticmethod
    def calculate_injuries_recovery_timeline(
        losses_human_per_asset,
        recovery_injuries,
        longest_time,
        datetime_earthquake,
    ):
        """
        This method calculates the number of people still not able to return to their building
        at each point in time defined by the time of the earthquake in UTC
        ('datetime_earthquake') and time horizons defined relative to the earthquake in terms of
        numbers of days in 'recovery_injuries'.

        Args:
            losses_human_per_asset (Pandas DataFrame):
                Pandas DataFrame indicating number of injured people (with different severity
                levels) and the following structure:
                    Index:
                        id (str):
                            ID of the asset.
                    Columns:
                        taxonomy (str):
                            Building class.
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of original_asset_id and id.
                        injuries_X (float):
                            Expected injuries of severity X for 'id'.
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
            longest_time (int):
                Maximum number of days since 'datetime_earthquake' that will be used
                (irrespective of the largest number of days in 'recovery_injuries').
            datetime_earthquake (numpy.datetime64):
                UTC date and time of the earthquake.

        Returns:
            injured_still_away (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        id (str):
                            ID of the asset.
                    Columns:
                        taxonomy (str):
                            Building class.
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of original_asset_id and id.
                        dates in UTC (float):
                            The names of the columns are UTC dates and times (as str) at which
                            the number of injured people still unable to return to their
                            buildings has been calculated. The content of the columns is the
                            number of injured people.
        """

        # Define timeline in UTC
        timeline_injuries_relative = Losses.define_timeline_recovery_relative(
            recovery_injuries["N_discharged"].to_numpy(), 0, longest_time
        )
        timeline_injuries_absolute = (
            datetime_earthquake
            + np.array([np.timedelta64(t, "D") for t in timeline_injuries_relative])
        )

        # Define f_severity: factor to account for whether people are able to go back into the
        # building based on their health status and irrespective of the damage state of the
        # building and the inspection times. One factor per injury severity (index) and time
        # threshold (column).
        f_severity = deepcopy(recovery_injuries)

        for i, time_threshold in enumerate(timeline_injuries_relative):
            # Initialise with zeros
            f_severity_aux = np.zeros([f_severity.shape[0]], dtype=int)
            # Turn to ones the cases where the time threshold is < the N_discharge number of days
            # (factors of 1 will cause the injured people to be removed from the occupancy)
            which_one = np.where(time_threshold < f_severity["N_discharged"].to_numpy())[0]
            f_severity_aux[which_one] = 1

            # Add column for this time threshold
            col_name = str(timeline_injuries_absolute[i].astype("datetime64[s]"))
            f_severity[col_name] = f_severity_aux

        # Calculate injured people still away (e.g. in hospital or dead)
        # per asset ID and time threshold
        injured_still_away = pd.DataFrame(
            {
                "taxonomy": losses_human_per_asset["taxonomy"],
                "original_asset_id": losses_human_per_asset["original_asset_id"],
                "building_id": losses_human_per_asset["building_id"],
            },
            index=losses_human_per_asset.index  # index is "id"
        )

        for time_threshold in timeline_injuries_absolute:
            time_threshold_str = str(time_threshold.astype("datetime64[s]"))
            injured_still_away_aux = np.zeros([losses_human_per_asset.shape[0]])
            for i, asset_id in enumerate(losses_human_per_asset.index):
                remove = 0.0
                for severity in f_severity.index:  # injury severity level
                    remove += (
                        f_severity.loc[severity, time_threshold_str]
                        * losses_human_per_asset.loc[asset_id, "injuries_%s" % (severity)]
                    )
                injured_still_away_aux[i] = remove

            injured_still_away[time_threshold_str] = injured_still_away_aux

        return injured_still_away

    @staticmethod
    def calculate_repair_recovery_timeline(
        recovery_damage,
        longest_time,
        datetime_earthquake,
    ):
        """
        This method calculates binary factors indicating whether occupants are allowed (1) back
        into their buildings or not (0), as a function of the damage state of the building and
        a specific point in UTC time. The latter is defined as a function of
        'datetime_earthquake' and the number of days relative to the date and time of the
        earthquake specified in 'recovery_damage'.

        Args:
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
            longest_time (int):
                Maximum number of days since 'datetime_earthquake' that will be used
                (irrespective of the largest number of days in 'recovery_damage').
            datetime_earthquake (numpy.datetime64):
                UTC date and time of the earthquake.

        Returns:
            occupancy_factors (Pandas DataFrame):
                Pandas DataFrame indicating whether any occupants are allowed in a building with
                a certain damage state at a specific point in time, with the following
                structure:
                    Index:
                        dmg_state (str):
                            Damage state.
                    Columns:
                        dates in UTC (int):
                            The names of the columns are UTC dates and times (as str) at which
                            the occupancy factors have been calculated, while the contents of
                            the columns are the factors themselves: 0 if nobody is allowed in
                            the building yet, 1 if occupants are allowed back in.
        """

        # Define timeline in UTC
        timeline_damage_relative = Losses.define_timeline_recovery_relative(
            recovery_damage["N_damage"].to_numpy(), 0, longest_time
        )
        timeline_damage_absolute = (
            datetime_earthquake
            + np.array([np.timedelta64(t, "D") for t in timeline_damage_relative])
        )

        # Define damage_factors: factor to account for whether people are allowed to go back
        # into the building, irrespective of their health status (e.g. based on inspection times
        # and damage state). One factor per damage state (index) and time threshold (column).
        occupancy_factors = pd.DataFrame(
            index=recovery_damage.index  # index is "dmg_state"
        )

        for i, time_threshold in enumerate(timeline_damage_relative):
            # Initialise with zeros
            occupancy_factors_aux = np.zeros([recovery_damage.shape[0]], dtype=int)
            # Turn to ones the cases where the time threshold is >= the N_damage number of days
            which_one = np.where(time_threshold >= recovery_damage["N_damage"].to_numpy())[0]
            occupancy_factors_aux[which_one] = 1

            # Add column for this time threshold
            col_name = str(timeline_damage_absolute[i].astype("datetime64[s]"))
            occupancy_factors[col_name] = occupancy_factors_aux

        return occupancy_factors

    @staticmethod
    def get_occupancy_factors(datetime_earthquake, mapping_damage_states, main_path):
        """
        This method goes through the repair recovery timelines of all earthquakes that have
        occurred before 'datetime_earthquake' (this is done by searching for all the occupancy
        factors files present in 'main_path'/current/occupants), and determines whether people
        will be allowed back in (1) or not (0) to buildings with the damage states indicated in
        'mapping_damage_states'.

        If there are no occupancy factors files in 'main_path'/current/occupants, all returned
        factors will be equal to 1.

        Args:
            datetime_earthquake (numpy.datetime64):
                UTC date and time of the earthquake.
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
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.

        Returns:
            occupancy_factors (dict of int: 0 or 1):
                Dictionary whose keys are the elements of the 'fragility' column of
                'mapping_damage_states', i.e. each damage state, and whose contents are 1 or 0,
                indicating that by the date of 'datetime_earthquake', people will be allowed
                back in (1) or not (0) to buildings with the damage state (key).
                Example output: {"DS0": 1, "DS1": 1, "DS2": 0, "DS3": 0, "DS4": 0}.
        """

        # Retrieve filenames of occupancy factors due to previous earthquakes
        path_to_factors = os.path.join(main_path, "current", "occupants")
        filenames = glob.glob(
            os.path.join(path_to_factors, "occupancy_factors_after_RLA_*.csv")
        )  # each file corresponds to one past earthquake

        # Read factors and collect them in a dictionary
        occupancy_factors = {}
        for dmg_state in mapping_damage_states.index:
            # Initiate as 1 because the retrieved factors (0 or 1) will be multiplied
            occupancy_factors[mapping_damage_states.loc[dmg_state, "fragility"]] = 1

        for filename in filenames:
            factors = pd.read_csv(os.path.join(path_to_factors, filename))
            factors.set_index(factors["dmg_state"], drop=True, inplace=True)
            factors = factors.drop(columns=["dmg_state"])

            for dmg_state in factors.index:
                occup_function = MultilinearStepFunction(
                    factors.columns.to_numpy().astype(np.datetime64),  # thresholds
                    factors.loc[dmg_state, :].to_numpy()  # values
                )
                occupancy_factors[dmg_state] = (
                    occupancy_factors[dmg_state]
                    * occup_function.evaluate_as_datetime(datetime_earthquake)
                )

        return occupancy_factors

    @staticmethod
    def get_occupancy_factors_per_asset(exposure_taxonomies, occupancy_factors):
        """
        This method retrieves from 'occupancy_factors' the factor associated with the damage
        state of each of the taxonomy strings of 'exposure_taxonomies'.

        Args:
            exposure_taxonomies (arr of str):
                Array of taxonomy strings as per the GEM Building Taxonomy v3.0, including the
                damage state as the last attribute (e.g. "CR/LFINF+CDN+LFC:0.0/H:1/DS1").
            occupancy_factors (dict of int: 0 or 1):
                Dictionary whose keys are damage states and whose contents are 1 (people are
                allowed back in) or 0 (people are not allowed back in) to buildings with the
                damage state (key).
                Example: {"DS0": 1, "DS1": 1, "DS2": 0, "DS3": 0, "DS4": 0}.

        Returns:
            occupancy_factors_per_asset (arr of int):
                Array of occupancy factors for each element of 'exposure_taxonomies'.
        """

        damage_states = [
            exposure_taxonomies[i].split("/")[-1] for i in range(len(exposure_taxonomies))
        ]

        occupancy_factors_per_asset = np.array([
            occupancy_factors[dmg_state] for dmg_state in damage_states
        ])

        return occupancy_factors_per_asset

    @staticmethod
    def get_injured_still_away(exposure_indices, datetime_earthquake, main_path):
        """
        This method goes through the injuries recovery timelines of all earthquakes that have
        occurred before 'datetime_earthquake' (this is done by searching for all the
        "still-away-injured" files present in 'main_path'/current/occupants), and determines
        the total number of injured people still away from their buildings by the time of
        'datetime_earthquake'. The output follows the order of the asset IDs listed in
        'exposure_indices'.

        If there are no "still-away-injured" files in 'main_path'/current/occupants, all
        returned values of injured people still away will be 0.

        Args:
            exposure_indices (arr of str):
                Array of asset IDs from the OpenQuake exposure CSV files. These should exist in
                the "still-away-injured" files present in 'main_path'/current/occupants.
            datetime_earthquake (numpy.datetime64):
                UTC date and time of the earthquake.
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.

        Returns:
            injured_still_away (arr of float):
                Array of total number of injured people still away from each of the buildings
                whose asset ID is listed in 'exposure_indices'.
        """

        # Retrieve filenames of injured people still away due to previous earthquakes
        path_to_injured = os.path.join(main_path, "current", "occupants")
        filenames = glob.glob(
            os.path.join(path_to_injured, "injured_still_away_after_RLA_*.csv")
        )  # each file corresponds to one past earthquake

        # Initialise the number of injured people still away (injured due to each earthquake
        # will be added)
        injured_still_away = np.zeros([len(exposure_indices)])

        for filename in filenames:
            injured = pd.read_csv(os.path.join(path_to_injured, filename))
            injured.set_index(injured["id"], drop=True, inplace=True)
            injured.index = injured.index.rename("asset_id")
            injured = injured.drop(columns=["id"])

            # Recover dates of the timeline from names of columns
            date_thresholds = list(injured.columns)
            date_thresholds.remove("taxonomy")
            date_thresholds.remove("original_asset_id")
            date_thresholds.remove("building_id")
            date_thresholds = np.array(date_thresholds, dtype=np.datetime64)

            for i, asset_id in enumerate(exposure_indices):
                injuries_function = MultilinearStepFunction(
                    date_thresholds,
                    injured.loc[asset_id, date_thresholds.astype(str)].to_numpy()  # values
                )
                injured_still_away[i] += injuries_function.evaluate_as_datetime(
                    datetime_earthquake
                )

        return injured_still_away

    @staticmethod
    def get_time_of_day_factors_per_asset(
        exposure_occupancies, earthquake_time_of_day, time_of_day_factors
    ):
        """
        This method returns the time-of-day factors from 'time_of_day_factors' for each
        occupancy case in 'exposure_occupancies', for the time of the day indicated by
        'earthquake_time_of_day'.

        Args:
            exposure_occupancies (arr of str):
                Array of occupancy cases from the OpenQuake exposure CSV files (e.g.
                "residential", "commercial", "industrial"). These should be keys of
                'time_of_day_factors'.
            earthquake_time_of_day (str):
                Time of the day of the earthquake, i.e. "day", "night" or "transit".
            time_of_day_factors (dict):
                Factors by which the census population per building can be multiplied to obtain
                an estimate of the people in the building at a certain time of the day. It
                should contain one key per occupancy case present in the exposure model (e.g.
                "residential", "commercial", "industrial"), and each key should be subdivided
                into:
                    - "day": approx. 10 am to 6 pm;
                    - "night": approx. 10 pm to 6 am;
                    - "transit": approx. 6 am to 10 am and 6 pm to 10 pm.

        Returns:
            time_of_day_factors_per_asset (arr of float):
                Array of factors from 'time_of_day_factors' corresponding to each element of
                'exposure_occupancies'.
        """

        time_of_day_factors_per_asset = np.array(
            [
                time_of_day_factors[occupancy][earthquake_time_of_day]
                for occupancy in exposure_occupancies
            ]
        )

        return time_of_day_factors_per_asset
