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
    def check_consequence_models(consequence_models, exposure_model):
        """
        This method verifies that all of the building classes listed in the "taxonomy" column of
        'exposure_model' are listed as well in each of the 'consequence_models'.

        Args:
            consequence_models (dict):
                Dictionary with the following keys and contents:
                - economic (Pandas DataFrame):
                    Pandas DataFrame with the consequence model for economic losses in terms of
                    mean values of loss ratio per damage state. Each row in the
                    'consequence_model' corresponds to a different building class. The structure
                    is as follows:
                        Index:
                            Taxonomy (str): Building classes.
                        Columns:
                            One per damage state (float): They contain the mean loss ratios (as
                            percentages) for each building class and damage state.
                - injuries (dict of Pandas DataFrame):
                    Dictionary whose keys are the injury severity levels and whose contents are
                    Pandas DataFrames with the consequence models for injuries in terms of mean
                    values of loss ratio per damage state. Each row in the consequence model
                    corresponds to a different building class. The structure is as follows:
                        Index:
                            Taxonomy (str): Building classes.
                        Columns:
                            One per damage state (float): They contain the mean loss ratios (as
                            percentages) for each building class and damage state.
            exposure_model (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake. It must
                contain at least a "taxonomy" column with the strings that define the building
                classes.

        Returns:
            classes_are_missing (bool):
                True if there are building classes listed in the "taxonomy" column of
                'exposure_model' that are not listed in each of the 'consequence_models'.
            missing_building_classes (dict):
                Dictionary with the same structure as 'consequence_models'. The contents are
                lists of building classes listed in the "taxonomy" column of 'exposure_model'
                that are not listed in each of the 'consequence_models' (i.e. the list of
                building classes that are missing).
        """

        # Unique building classes in the exposure model, without the damage state
        exposure_bdg_classes = np.array(
            [
                "/".join(exposure_model["taxonomy"].to_numpy()[i].split("/")[:-1])
                for i in range(exposure_model.shape[0])
            ]
        )
        exposure_bdg_classes = np.unique(exposure_bdg_classes)

        # Initialise output
        missing_building_classes = {}
        classes_are_missing = False

        for loss_type in consequence_models:  # economic, injuries
            if isinstance(consequence_models[loss_type], dict):
                missing_building_classes[loss_type] = {}
                for level in consequence_models[loss_type]:  # injuries: level of severity
                    missing_building_classes[loss_type][level] = (
                        Losses.identify_missing_building_classes(
                            consequence_models[loss_type][level].index.to_numpy(),
                            exposure_bdg_classes
                        )
                    )
                    if len(missing_building_classes[loss_type][level]) > 0:
                        classes_are_missing = True
            else:  # economic
                missing_building_classes[loss_type] = (
                    Losses.identify_missing_building_classes(
                        consequence_models[loss_type].index.to_numpy(),
                        exposure_bdg_classes
                    )
                )
                if len(missing_building_classes[loss_type]) > 0:
                    classes_are_missing = True

        return classes_are_missing, missing_building_classes

    @staticmethod
    def identify_missing_building_classes(available_strings, target_strings):
        """
        This method checks if all the strings in 'target_strings' are present in
        'available_strings' strings too (but not the other way around), and returns a list with
        the missing strings (empty list if nothing is missing).

        Args:
            available_strings (arr of str)
            target_strings (arr of str)

        Returns:
            missing_strings (list of str):
                List of strings from 'target_strings' not present in 'available_strings'.
        """

        missing_strings = []

        for a_string in target_strings:
            if a_string not in available_strings:
                missing_strings.append(a_string)

        return missing_strings

    @staticmethod
    def _get_loss_ratios(consequence_model, building_classes, damage_states):
        """
        This method returns the loss ratio (as stated in 'consequence_model') for each
        combination of building class in 'building_classes' and damage state in 'damage_states'.
        The loss ratio is returned in the 0-1 range (not in percentage, which is what
        'consequence_model' is expected to contain).

        Args:
            consequence_model (Pandas DataFrame):
                Pandas DataFrame with the consequence model for economic losses in terms of mean
                values of loss ratio per damage state. Each row in the 'consequence_model'
                corresponds to a different building class. The structure is as follows:
                    Index:
                        Taxonomy (str): Building classes.
                    Columns:
                        One per damage state (float): They contain the mean loss ratios (as
                        percentages) for each building class and damage state.
            building_classes (Numpy array of str):
                Array of target building classes, expected to exist in 'consequence_model'.
            damage_states (Numpy array of str):
                Array of target damage states, expected to exist in 'consequence_model'.
        """

        # Retrieve the indices and col numbers within 'consequence_model' of the elements in
        # 'building_classes' and 'damage_states'
        # ("-1" if class/state is missing from 'consequence_model')
        row_idx = consequence_model.index.get_indexer(building_classes)
        col_idx = consequence_model.columns.get_indexer(damage_states)

        if (row_idx < 0).any():
            missing = np.unique(np.asarray(building_classes)[row_idx < 0])
            raise KeyError("Building classes not in consequence model: %s" % (missing,))

        if (col_idx < 0).any():
            missing = np.unique(np.asarray(damage_states)[col_idx < 0])
            raise KeyError("Damage states not in consequence model: %s" % (missing,))

        losspercentages = consequence_model.to_numpy()[row_idx, col_idx]
        ratios = losspercentages / 100.0

        return ratios

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
            columns=["lon", "lat", "occupancy", "original_asset_id"]
        )
        if "id" in loss_summary.columns:
            loss_summary = loss_summary.drop(columns=["id"])

        # Create separate columns for building class and damage state
        logger.debug(
            "%s Method 'Losses.expected_economic_loss': "
            "separating building class from damage state"
            % (np.datetime64('now'))
        )
        taxonomy = loss_summary["taxonomy"].to_numpy()
        loss_summary["damage_state"] = [
            taxonomy[i].split("/")[-1] for i in range(loss_summary.shape[0])
        ]
        loss_summary["building_class"] = [
            "/".join(taxonomy[i].split("/")[:-1]) for i in range(loss_summary.shape[0])
        ]

        loss_ratios = Losses._get_loss_ratios(
            consequence_model,
            loss_summary["building_class"].to_numpy(),
            loss_summary["damage_state"].to_numpy(),
        )  # array of loss ratios in range 0-1
        loss_summary["loss"] = loss_ratios * loss_summary["structural"]

        loss_summary = loss_summary.groupby(["building_id"]).agg({"loss": "sum"})

        return loss_summary

    @staticmethod
    def expected_human_loss_per_original_asset_id(exposure, time_of_day, consequence_model):
        """
        This method returns the expected human loss per 'original_asset_id' of the asset in
        'exposure', as per the damage states therein specified and the human loss ratios dicated
        by 'consequence_model' for a specific 'time_of_day'.

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
            losses_per_orig_asset (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                    Columns:
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of original_asset_id.
                        injuries_X (float):
                            Expected injuries of severity X for 'original_asset_id'.
        """

        # Initialise output
        losses_per_orig_asset = deepcopy(exposure)

        # Create separate columns for building class and damage state
        logger.debug(
            "%s Method 'Losses.expected_human_loss_per_original_asset_id': "
            "separating building class from damage state"
            % (np.datetime64('now'))
        )
        taxonomy = losses_per_orig_asset["taxonomy"].to_numpy()
        losses_per_orig_asset["building_class"] = [
            "/".join(taxonomy[i].split("/")[:-1]) for i in range(losses_per_orig_asset.shape[0])
        ]
        losses_per_orig_asset["damage_state"] = [
            taxonomy[i].split("/")[-1] for i in range(losses_per_orig_asset.shape[0])
        ]

        how_to_group = {"building_id": "first"}

        building_classes = losses_per_orig_asset["building_class"].to_numpy()
        damage_states = losses_per_orig_asset["damage_state"].to_numpy()
        occupants = losses_per_orig_asset[time_of_day].to_numpy()

        for severity in consequence_model:
            logger.debug(
                "%s Method 'Losses.expected_human_loss_per_original_asset_id': "
                "calculating injuries of severity %s"
                % (np.datetime64('now'), severity)
            )

            loss_ratios = Losses._get_loss_ratios(
                consequence_model[severity], building_classes, damage_states
            )
            losses_per_orig_asset["injuries_%s" % (severity)] = loss_ratios * occupants

            how_to_group["injuries_%s" % (severity)] = "sum"

        logger.debug(
            "%s Method 'Losses.expected_human_loss_per_original_asset_id': "
            "grouping by original_asset_id"
            % (np.datetime64('now'))
        )
        losses_per_orig_asset = losses_per_orig_asset.groupby(
            ["original_asset_id"]  # original_asset_id becomes index
        ).agg(how_to_group)  # sum injuries, keep building_id

        return losses_per_orig_asset

    @staticmethod
    def expected_human_loss_per_building_id(human_losses_per_original_asset_id):
        """
        This method returns the expected human loss per building ID, starting from expected
        human losses per original asset ID (the output of method
        "expected_human_loss_per_original_asset_id").

        Args:
            human_losses_per_original_asset_id (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                    Columns:
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of original_asset_id.
                        injuries_X (float):
                            Expected injuries of severity X for 'original_asset_id'.
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

        loss_summary = human_losses_per_original_asset_id.groupby(
            ["building_id"]
        ).sum(numeric_only=True)

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
        losses_human_per_orig_asset_id,
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
            losses_human_per_orig_asset_id (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                    Columns:
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of original_asset_id.
                        injuries_X (float):
                            Expected injuries of severity X for 'original_asset_id'.
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
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                    Columns:
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of original_asset_id.
                        dates in UTC (float):
                            The names of the columns are UTC dates and times (as str) at which
                            the number of injured people still unable to return to their
                            buildings has been calculated. The content of the columns is the
                            number of injured people.
        """

        # Define timeline in UTC
        logger.debug(
            "%s Method 'Losses.calculate_injuries_recovery_timeline': "
            "defining timeline in UTC"
            % (np.datetime64('now'))
        )
        timeline_injuries_relative = Losses.define_timeline_recovery_relative(
            recovery_injuries["N_discharged"].to_numpy(), 0, longest_time
        )
        timeline_injuries_absolute = (
            datetime_earthquake
            + np.array([np.timedelta64(t, "D") for t in timeline_injuries_relative])
        )

        # Names of the columns of the output (absolute time thresholds, as str)
        col_names = [str(t.astype("datetime64[s]")) for t in timeline_injuries_absolute]

        # Define f_severity: factor to account for whether people are able to go back into the
        # building based on their health status and irrespective of the damage state of the
        # building and the inspection times. One factor per injury severity (row) and time
        # threshold (column).

        # Turn to ones the cases where the time threshold is < the N_discharge number of days
        # (factors of 1 will cause the injured people to be removed from the occupancy).
        # (Zero otherwise)
        f_severity = (
            np.asarray(timeline_injuries_relative)[np.newaxis, :]
            < recovery_injuries["N_discharged"].to_numpy()[:, np.newaxis]
        ).astype(float)  # dim: n_sev × n_times

        # Expected injuries per original_asset_id (rows; i.e. one row per original_asset_id) and
        # severity (columns), with the columns in the same order as the rows of 'f_severity'
        injuries = losses_human_per_orig_asset_id[
            ["injuries_%s" % (severity) for severity in recovery_injuries.index]
        ].to_numpy()  # dim: n_assets × n_sev

        # Calculate injured people still away (e.g. in hospital or dead)
        # per asset ID and time threshold (sums up all severities for that asset)
        injured_still_away = pd.concat(
            [
                losses_human_per_orig_asset_id[["building_id"]],
                pd.DataFrame(
                    injuries @ f_severity,  # matrix product [n_assets × n_sev] @ [n_sev × n_times]
                    index=losses_human_per_orig_asset_id.index,  # "original_asset_id"
                    columns=col_names,
                ),
            ],
            axis=1,
        )

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
        logger.debug(
            "%s Method 'Losses.calculate_repair_recovery_timeline': "
            "defining timeline in UTC"
            % (np.datetime64('now'))
        )
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
            logger.debug(
                "%s Method 'Losses.calculate_repair_recovery_timeline': "
                "defining damage_factors for time threshold %s of %s"
                % (np.datetime64('now'), i+1, len(timeline_damage_relative))
            )
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
    def get_occupancy_factors_RLA(
        datetime_earthquake, mapping_damage_states, main_path
    ):
        """
        This method retrieves the repair recovery timeline of the last earthquake run with the
        RLA algorithm, and determines whether people will be allowed back in (1) or not (0) to
        buildings with the damage states indicated in 'mapping_damage_states' by
        'datetime_earthquake'.

        If no RLA earthquake has been run before, all returned factors will be equal to 1.

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

        # Retrieve occupancy factors due to last RLA earthquake, if it exists
        factors_timelines = Losses.read_occupancy_factors_timelines_last_RLA(main_path)

        occupancy_factors = {}

        if not (factors_timelines is None):

            for dmg_state in factors_timelines.index:
                occup_function = MultilinearStepFunction(
                    factors_timelines.columns.to_numpy().astype(np.datetime64),  # thresholds
                    factors_timelines.loc[dmg_state, :].to_numpy()  # values
                )
                occupancy_factors[dmg_state] = occup_function.evaluate(datetime_earthquake)

        else:  # No previous RLA earthquake has been run, all factors are 1
            for dmg_state in mapping_damage_states.index:
                occupancy_factors[mapping_damage_states.loc[dmg_state, "fragility"]] = 1

        return occupancy_factors

    @staticmethod
    def read_occupancy_factors_timelines_last_RLA(main_path):
        """
        This method retrieves the repair recovery timeline of the last earthquake run with the
        RLA algorithm. It returns None if no RLA earthquake has been previously run.
        """

        # Retrieve occupancy factors due to last RLA earthquake, if it exists
        path_to_factors = os.path.join(main_path, "current", "occupants")
        filepath = os.path.join(path_to_factors, "occupancy_factors_after_last_RLA.csv")

        if os.path.exists(filepath):
            factors_timelines = pd.read_csv(filepath)
            factors_timelines.set_index(factors_timelines["dmg_state"], drop=True, inplace=True)
            factors_timelines = factors_timelines.drop(columns=["dmg_state"])
        else:  # No previous RLA earthquake has been run\
            factors_timelines = None

        return factors_timelines

    @staticmethod
    def get_occupancy_factors_all_OELF_earthquakes(
        forecast_catalogue, recovery_damage, main_path
    ):
        """
        This method calculates the recovery occupancy factors (0-1) of all earthquakes of
        'forecast_catalogue' for which there can be occupants (and outputs zeroes for the
        earthquakes for which occupants cannot be expected) and forecast_catalogue["to_run"] is
        True. The output is a dictionary whose keys are "eq_id" of 'forecast_catalogue', and
        each of whose elements is a dicctionary of the kind {"DS0": 1, "DS1": 1, "DS2": 0,
        "DS3": 0, "DS4": 0}.

        For the calculation of the factors, the method splits 'forecast_catalogue' in two groups:
        - the "first" earthquakes of each (oef_model_id, oef_ses_id) subset, for which the
        "previous" earthquake is the last-run RLA earthquake (whose timeline is retrieved from
        the stored CSV file);
        - all the rest of the earthquakes for which the "previous" earthquake is the precedent
        row of 'forecast_catalogue' within its own (oef_model_id, oef_ses_id) subset.

        Args:
            forecast_catalogue (Pandas DataFrame):
                DataFrame containing a seismicity forecast for a period of time of interest. It
                normally consists of several realisations of seismicity, which we call
                "stochastic event sets" (SES), each realisation being identified by a 'ses_id'.
                One 'ses_id' can contain one or more earthquakes, each of them identified with
                an 'event_id'. The values of 'event_id' can be repeated across different values
                of 'ses_id'. The index is the "EQID" consisting of model_id, ses_id and
                event_id. The DataFrame must contain the following fields:
                    datetime (Numpy datetime64): Date and time.
                    model_id (int): ID of the seismicity model used to generate the SES.
                    ses_id (int): ID of the stochastic event set (SES) that the earthquake
                        belongs to.
                    event_id (str): Unique identifier of an earthquake within a 'ses_id'.
                    to_run (bool): If True, the corresponding earthquake is to be
                        used to run damage/loss calculations and thus occupancy factors are
                        calculated; if False, it will be skipped.
                    there_can_be_occups (bool): True if the time difference with the previously-
                        run earthquake is such that there might be people occupying buildings
                        during this earthquake. Only relevant if 'there_can_be_occupants' and
                        'calculate_casualties' are True, not needed otherwise.
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
            main_path (str):
                Path to the main running directory, assumed to have the needed structure. Needed
                to check for the last-run RLA earthquake.

        Returns:
            recovery_occupancy_factors (dict):
                Dictionary with keys = eq_id, and each element being a dictionary with the 0-1
                occupancy factors for each earthquake of 'forecast_catalogue' (e.g., {"DS0": 1,
                "DS1": 1, "DS2": 0, "DS3": 0, "DS4": 0}) for which "to_run" is True. For
                earthquakes for which no exposure occupants are expected all values are 0.
                Earthquakes for which forecast_catalogue["to_run"] is False are skipped.
        """

        # Discard earthquakes that do not cause damage (i.e. too small or too far away)
        forecast_catalogue_to_run = forecast_catalogue[forecast_catalogue.to_run]

        # Group 'forecast_catalogue' so that it is easier to identify each
        # (oef_model_id, oef_ses_id) subset ('catalogue_groups' is DataFrameGroupBy object)
        catalogue_groups = forecast_catalogue_to_run.groupby(["model_id", "ses_id"])

        # Initialise aux_occup_factors full of 0 for all elements of forecast_catalogue_to_run
        # (these zeroes will remain so for earthquakes for which "there_can_be_occups" is False)
        dtypes_per_ds = {}
        for dmg_state in recovery_damage.index:
            dtypes_per_ds[dmg_state] = "int32"

        aux_occup_factors = pd.DataFrame(
            index=forecast_catalogue_to_run.index,
            data=np.zeros((forecast_catalogue_to_run.shape[0], recovery_damage.shape[0]), dtype="int32"),
            columns=recovery_damage.index
        ).astype(dtypes_per_ds)

        # FIRST SUB-SET OF EARTHQUAKES TO PROCESS
        # First earthquakes of each (oef_model_id, oef_ses_id) subset
        # (i.e. the ones for which the previous earthquake is the last-run RLA one)
        first_row_mask = catalogue_groups.cumcount().eq(0)  # 1st earthquakes
        cat_firsts = forecast_catalogue_to_run[
            first_row_mask & forecast_catalogue_to_run["there_can_be_occups"]
        ]  # 1st earthquakes for which there can be occupants

        if cat_firsts.shape[0] > 0:
            # Retrieve occupancy factors due to last RLA earthquake, if it exists
            factors_timelines = Losses.read_occupancy_factors_timelines_last_RLA(main_path)

            if factors_timelines is None:
                # No RLA EQ has been run before, all occupancy factors are 1
                aux_occup_factors.loc[cat_firsts.index, :] = 1

            else:
                threshold_dates = factors_timelines.columns.to_numpy().astype(np.datetime64)

                for dmg_state in factors_timelines.index:
                    occup_function = MultilinearStepFunction(
                        threshold_dates,  # thresholds
                        factors_timelines.loc[dmg_state, :].to_numpy()  # values
                    )
                    aux_occup_factors.loc[cat_firsts.index, dmg_state] = (
                        occup_function.evaluate(cat_firsts["datetime"]).astype("int32")
                    )

        # SECOND SUB-SET OF EARTHQUAKES TO PROCESS
        # All non-first earthquakes for which there can be occupants
        nonfirst_row_mask = catalogue_groups.cumcount().gt(0)  # all earthquakes except first ones
        cat_nonfirsts = forecast_catalogue_to_run[
            nonfirst_row_mask & forecast_catalogue_to_run["there_can_be_occups"]
        ]  # non-1st earthquakes for which there can be occupants

        # Shift the datetime column by 1 within each group
        # (original index, now giving datetime of the previous row within the same group)
        prev_datetimes = catalogue_groups["datetime"].shift(1)
        # Select only the row-1 datetimes corresponding to non-first rows
        prev_datetimes_of_nonfirsts = prev_datetimes[
            nonfirst_row_mask & forecast_catalogue_to_run["there_can_be_occups"]
        ]

        # Time difference between earthquakes and their previous ones (as Pandas Series)
        diff = cat_nonfirsts["datetime"] - prev_datetimes_of_nonfirsts
        diff_days = diff.astype("int64") / 86400_000_000_000 # in fractional days (Pandas Series)

        # Compare the difference in days against the time it takes to inspect+repair
        damage_days = recovery_damage["N_damage"]
        comparison_df = (diff_days.values[:, None] > damage_days.values[None, :]).astype("int32")
        comparison_df = pd.DataFrame(
            comparison_df,
            index=diff_days.index,
            columns=damage_days.index,
        )  # output is 0s and 1s

        # Bring in the results of the second subset to the overall output
        aux_occup_factors.loc[comparison_df.index, comparison_df.columns] = comparison_df.loc[
            : , :
        ]

        # Convert DataFrame into dictionary with keys=eq_ids and vals={"DS0": 1, ...}
        recovery_occupancy_factors = aux_occup_factors[
            recovery_damage.index
        ].to_dict(orient="index")

        return recovery_occupancy_factors

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

        damage_states = [s.rsplit("/", 1)[-1] for s in exposure_taxonomies]

        occupancy_factors_per_asset = np.fromiter(
            (occupancy_factors[ds] for ds in damage_states),
            dtype=int,
            count=len(damage_states)
        )

        return occupancy_factors_per_asset

    @staticmethod
    def read_injured_still_away_RLA(main_path):
        """
        This method reads the injuries recovery timelines (number of injured people still away
        from their buildings in time) of all "real" (RLA) earthquakes run so far, stored under
        'main_path'/current/occupants as "injured_still_away_after_RLA_*.csv" files (one per
        earthquake).

        Args:
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.

        Returns:
            injured_still_away_rla (dict of Pandas DataFrame):
                Dictionary whose keys are the names of the files (one per RLA earthquake) and
                whose elements are Pandas DataFrames with the following structure:
                    Index:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                    Columns:
                        building_id (str):
                            ID of the building.
                        dates in UTC (float):
                            The names of the columns are UTC dates and times (as str) at which
                            the number of injured people still unable to return to their
                            buildings has been calculated. The content of the columns is the
                            number of injured people.
                The dictionary is empty if no RLA earthquake has been run before.
        """

        path_to_injured = os.path.join(main_path, "current", "occupants")
        filenames = glob.glob(
            os.path.join(path_to_injured, "injured_still_away_after_RLA_*.csv")
        )  # each file corresponds to one past earthquake

        injured_still_away_rla = {}

        for filename in filenames:
            injured = pd.read_csv(os.path.join(path_to_injured, filename))
            injured.set_index(injured["original_asset_id"], drop=True, inplace=True)
            injured = injured.drop(columns=["original_asset_id"])
            injured_still_away_rla[os.path.basename(filename)] = injured

        return injured_still_away_rla

    @staticmethod
    def get_injured_still_away(
        exposure_orig_asset_ids,
        datetime_earthquake,
        main_path,
        injured_still_away_oelf=None,
        injured_still_away_rla=None,
    ):
        """
        This method goes through the injuries recovery timelines of all earthquakes that have
        occurred before 'datetime_earthquake' and determines the total number of injured people
        still away from their buildings by the time of 'datetime_earthquake'. The output
        follows the order of the original asset IDs listed in 'exposure_orig_asset_ids'.

        The timelines of "real" (RLA) earthquakes are retrieved from the
        "injured_still_away_after_RLA_*.csv" files present in 'main_path'/current/occupants,
        unless they are provided via 'injured_still_away_rla'. The timelines of OELF
        earthquakes (earthquakes of the stochastic event set being run) are never stored in
        files, they are provided via 'injured_still_away_oelf'.

        If there are no timelines to consider, all returned values of injured people still away
        will be 0.

        Args:
            exposure_orig_asset_ids (arr of str):
                Array of original asset IDs from the OpenQuake exposure CSV files (i.e. the
                asset IDs of the undamaged exposure model, stored as the original_asset_id
                column). These should exist in all the timelines considered.
            datetime_earthquake (numpy.datetime64):
                UTC date and time of the earthquake.
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.
            injured_still_away_oelf (dict of Pandas DataFrame):
                Injuries recovery timelines of the previous OELF earthquakes of the stochastic
                event set being run. The keys are the IDs of the earthquakes and the elements
                are the Pandas DataFrames output by
                Losses.calculate_injuries_recovery_timeline(). Use None (or an empty dictionary)
                to not consider any OELF earthquakes (e.g. when running RLA).
            injured_still_away_rla (dict of Pandas DataFrame):
                Injuries recovery timelines of the RLA earthquakes, as output by
                Losses.read_injured_still_away_RLA(). If None, they are read from the CSV
                files stored under 'main_path'/current/occupants. Pass them to avoid re-reading
                the same files for every earthquake (e.g. when running OELF).

        Returns:
            injured_still_away (arr of float):
                Array of total number of injured people still away from each of the buildings
                whose original asset ID is listed in 'exposure_orig_asset_ids'.
        """

        if injured_still_away_rla is None:
            injured_still_away_rla = Losses.read_injured_still_away_RLA(main_path)

        timelines = list(injured_still_away_rla.values())
        if injured_still_away_oelf is not None:
            timelines = timelines + list(injured_still_away_oelf.values())

        # Initialise the number of injured people still away (injured due to each earthquake
        # will be added)
        injured_still_away = np.zeros([len(exposure_orig_asset_ids)])

        for injured in timelines:
            # 'injured' has as columns: "building_id" and the dates of the timeline

            # Recover dates of the timeline from names of columns
            date_thresholds = list(injured.columns)
            date_thresholds.remove("building_id")
            date_thresholds = np.array(date_thresholds, dtype=np.datetime64)

            for i, asset_id in enumerate(exposure_orig_asset_ids):
                injuries_function = MultilinearStepFunction(
                    date_thresholds,
                    injured.loc[asset_id, date_thresholds.astype(str)].to_numpy()  # values
                )
                injured_still_away[i] += injuries_function.evaluate(
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

    @staticmethod
    def get_expected_costs_occupants(exposure_model, calculate_casualties):
        """
        This method calculates the expected total replacement cost and total number of census
        occupants (i.e. occupants irrespective of the time of the day, only if
        'calculate_casualties'is True) per building_id of the input 'exposure_model', whose
        structure is that on the OpenQuake exposure CSV format.

        Args:
            exposure_model (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake for the
                undamaged structures. It comprises at least the following fields:
                    Index (simple):
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                    Columns:
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of asset_id.
                        number (float):
                            Number of buildings in this asset.
                        structural (float):
                            Total replacement cost of this asset (all buildings in "number").
                        census (float):
                            Total number of occupants in this asset (all buildings in "number"),
                            irrespective of the time of the day.
            calculate_casualties (bool):
                If True, the output will include census occupants. If False, the output will
                only include replacement costs.

        Returns:
            expected_costs_occupants (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index (simple):
                        building_id (str):
                            ID of the building. One building_id can be associated with different
                            values of asset_id of the input 'exposure_model'.
                    Columns:
                        structural (float):
                            Total replacement cost of this 'building_id'.
                        census (float):
                            Total number of occupants in this 'building_id'.
        """

        if calculate_casualties:
            grouping_dic = {"structural": "sum", "census": "sum"}
        else:
            grouping_dic = {"structural": "sum"}

        expected_costs_occupants = exposure_model.groupby(["building_id"]).agg(grouping_dic)

        return expected_costs_occupants
