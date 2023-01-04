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

import os
import logging
from copy import deepcopy
import pandas as pd


logger = logging.getLogger()


class PostProcessor:
    """This class handles methods associated with post-processing the output.
    """

    @staticmethod
    def export_collected_output_damage(main_path, list_rla, list_oelf):
        """
        This method calls 'PostProcessor._collect_output_damage()' to open all individual damage
        output files asociated with each earthquake in 'list_rla' and each earthquake forecast
        listed in 'list_oelf', and exports all results to:
            - 'main_path'/output/all_damage_states_after_RLA_cumulative.csv: Cumulative damage
            after each earthquake for which a rapid loss assessment (RLA) was run ("cumulative"
            is used herein to indicate that the output damage results from all earthquakes run
            until that point in time).
            - 'main_path'/output/all_damage_states_after_RLA_incremental.csv: Incremental damage
            after each earthquake for which a rapid loss assessment (RLA) was run ("incremental"
            is used herein to indicate the contribution of each earthquake to the damage
            results; incremental damage can be negative, as the number of non-damaged buildings
            decreases when successive earthquakes inflict damage on previously-undamaged
            structures).
            - 'main_path'/output/all_damage_states_after_OELF_cumulative.csv: Incremental damage
            after each earthquake forecast for which an operational earthquake loss forecasting
            (OELF) calculation was run. In the case of OELF, "cumulative" means considering all
            "real" (RLA) earthquakes run so far plus the earthquake forecast, not all earthquake
            forecasts after one another.

        No file is exported when the list of earthquakes is empty.

        Args:
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.
            list_rla (list of str, can be empty):
                List of names of the RLA earthquakes that have been processed.
            list_oelf (list of str, can be empty):
                List of names of the OELF earthquakes that have been processed.
        """

        # RLA damage output
        if len(list_rla) > 0:
            collected_rla = PostProcessor._collect_output_damage(
                os.path.join(main_path, "output"), list_rla, "damage_states_after_RLA_%s.csv"
            )
            collected_rla.to_csv(
                os.path.join(main_path, "output", "all_damage_states_after_RLA_cumulative.csv"),
                index=True,
            )
            collected_rla_incremental = PostProcessor._get_incremental_from_cumulative(
                collected_rla, list_rla
            )
            collected_rla_incremental.to_csv(
                os.path.join(main_path, "output", "all_damage_states_after_RLA_incremental.csv"),
                index=True,
            )

        # OELF damage output
        if len(list_oelf) > 0:
            collected_oelf = PostProcessor._collect_output_damage(
                os.path.join(main_path, "output"), list_oelf, "damage_states_after_OELF_%s.csv"
            )
            collected_oelf.to_csv(
                os.path.join(main_path, "output", "all_damage_states_after_OELF_cumulative.csv"),
                index=True,
            )

    @staticmethod
    def _collect_output_damage(path, list_earthquakes, filename_pattern):
        """
        This method opens all individual damage output files asociated with each earthquake or
        earthquake forecast listed in 'list_earthquakes' and groups all results into one output
        DataFrame. The inidividual output files are sought within 'path' and are assumed to be
        named following 'filename_pattern'.

        If 'list_earthquakes' is empty, the output is None.

        E.g.:
            path = "sth"
            list_earthquakes = ["EQ_01", "EQ_02"]
            filename_pattern = "damage_states_after_RLA_%s.csv"

            This method will collect results from files "sth/damage_states_after_RLA_EQ_01.csv"
            and "sth/damage_states_after_RLA_EQ_02.csv".

        Args:
            path (str):
                Path where individual output files will be sought.
            list_earthquakes (list of str):
                List of earthquake names to be sought.
            filename_pattern (str):
                Pattern of the filenames to be sought. E.g. "damage_states_after_RLA_%s.csv"
                (see example above).

        Returns:
            collected_output (Pandas DataFrame):
                Pandas DataFrame with damage results from all earthquakes or earthquake
                forecasts listed in 'list_earthquakes', with the following structure:
                    Index (multiple):
                        building_id (str): ID of the building.
                        damage_state (str): Damage state of the building.
                    Columns named as per each of the elements in 'list_earthquakes' (float):
                        Probability of 'building_id' resulting in 'damage_state' (if
                        'building_id' refers to an individual building) or number of buildings
                        of 'building_id' expected to result in 'damage_state'(if 'building_id'
                        refers to a group of buildings).
        """

        collected_output = None  # initialise, for case 'filenames' is empty

        filenames = [
            filename_pattern % (cat_name) for cat_name in list_earthquakes
        ]

        for i, filename in enumerate(filenames):
            indiv_output = pd.read_csv(os.path.join(path, filename))
            new_index = pd.MultiIndex.from_arrays(
                [indiv_output["building_id"], indiv_output["damage_state"]]
            )
            indiv_output.index = new_index
            indiv_output = indiv_output.drop(columns=["building_id", "damage_state"])
            indiv_output = indiv_output.rename(columns={"number": list_earthquakes[i]})

            if i == 0:
                collected_output = deepcopy(indiv_output)
            else:
                collected_output = pd.concat(
                    [collected_output, indiv_output],
                    axis=1
                )
                # NaNs may arise from damage states missing in the output --> transform to zero
                collected_output = collected_output.fillna(0)

        return collected_output

    @staticmethod
    def export_collected_output_losses_economic(main_path, list_rla, list_oelf):
        """
        This method calls 'PostProcessor._collect_output_losses_economic()' to open all
        individual output files of economic losses asociated with each earthquake in 'list_rla'
        and each earthquake forecast listed in 'list_oelf', and exports all results to:
            - 'main_path'/output/all_losses_economic_after_RLA_cumulative.csv: Cumulative
            economic losses after each earthquake for which a rapid loss assessment (RLA) was
            run ("cumulative" is used herein to indicate that the output economic loss results
            from all earthquakes run until that point in time).
            - 'main_path'/output/all_losses_economic_after_RLA_incremental.csv: Incremental
            economic losses after each earthquake for which a rapid loss assessment (RLA) was
            run ("incremental" is used herein to indicate the contribution of each earthquake to
            the economic loss results; incremental economic losses can only be equal to or
            larger than zero).
            - 'main_path'/output/all_losses_economic_after_OELF_cumulative.csv: Incremental
            economic losses after each earthquake forecast for which an operational earthquake
            loss forecasting (OELF) calculation was run. In the case of OELF, "cumulative" means
            considering all "real" (RLA) earthquakes run so far plus the earthquake forecast,
            not all earthquake forecasts after one another.

        No file is exported when the list of earthquakes is empty.

        Args:
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.
            list_rla (list of str, can be empty):
                List of names of the RLA earthquakes that have been processed.
            list_oelf (list of str, can be empty):
                List of names of the OELF earthquakes that have been processed.
        """

        # RLA economic losses output
        if len(list_rla) > 0:
            collected_rla = PostProcessor._collect_output_losses_economic(
                os.path.join(main_path, "output"),
                list_rla,
                "losses_economic_after_RLA_%s.csv",
            )
            collected_rla.to_csv(
                os.path.join(
                    main_path,
                    "output",
                    "all_losses_economic_after_RLA_cumulative.csv",
                ),
                index=True,
            )
            collected_rla_incremental = PostProcessor._get_incremental_from_cumulative(
                collected_rla, list_rla
            )
            collected_rla_incremental.to_csv(
                os.path.join(
                    main_path,
                    "output",
                    "all_losses_economic_after_RLA_incremental.csv",
                ),
                index=True,
            )

        # OELF economic losses output
        if len(list_oelf) > 0:
            collected_oelf = PostProcessor._collect_output_losses_economic(
                os.path.join(main_path, "output"),
                list_oelf,
                "losses_economic_after_OELF_%s.csv",
            )
            collected_oelf.to_csv(
                os.path.join(
                    main_path,"output", "all_losses_economic_after_OELF_cumulative.csv"
                ),
                index=True,
            )

    @staticmethod
    def _collect_output_losses_economic(path, list_earthquakes, filename_pattern):
        """
        This method opens all individual output files of economic losses asociated with each
        earthquake or earthquake forecast listed in 'list_earthquakes' and groups all results
        into one output DataFrame. The inidividual output files are sought within 'path' and are
        assumed to be named following 'filename_pattern'.

        If 'list_earthquakes' is empty, the output is None.

        E.g.:
            path = "sth"
            list_earthquakes = ["EQ_01", "EQ_02"]
            filename_pattern = "losses_economic_after_RLA_%s.csv"

            This method will collect results from files
            "sth/losses_economic_after_RLA_EQ_01.csv" and
            "sth/losses_economic_after_RLA_EQ_02.csv".

        Args:
            path (str):
                Path where individual output files will be sought.
            list_earthquakes (list of str):
                List of earthquake names to be sought.
            filename_pattern (str):
                Pattern of the filenames to be sought. E.g. "losses_economic_after_RLA_%s.csv"
                (see example above).

        Returns:
            collected_output (Pandas DataFrame):
                Pandas DataFrame with economic loss results from all earthquakes or earthquake
                forecasts listed in 'list_earthquakes', with the following structure:
                    Index:
                        building_id (str): ID of the building.
                    Columns named as per each of the elements in 'list_earthquakes' (float):
                        Economic loss associated with 'building_id' after the earthquake
                        associated with the column name.
        """

        collected_output = None  # initialise, for case 'filenames' is empty

        filenames = [
            filename_pattern % (cat_name) for cat_name in list_earthquakes
        ]

        for i, filename in enumerate(filenames):
            indiv_output = pd.read_csv(os.path.join(path, filename))
            indiv_output.set_index(
                indiv_output["building_id"], drop=True, inplace=True
            )
            indiv_output = indiv_output.drop(columns=["building_id"])
            indiv_output = indiv_output.rename(columns={"loss": list_earthquakes[i]})

            if i == 0:
                collected_output = deepcopy(indiv_output)
            else:
                collected_output = pd.concat(
                    [collected_output, indiv_output],
                    axis=1
                )
                # Transform NaNs to zero
                collected_output = collected_output.fillna(0)

        return collected_output

    @staticmethod
    def export_collected_output_losses_human(main_path, injuries_scale, list_rla, list_oelf):
        """
        This method calls 'PostProcessor._collect_output_losses_human()' to open all
        individual output files of human losses asociated with each earthquake in 'list_rla'
        and each earthquake forecast listed in 'list_oelf', and exports all results of each case
        (RLA, OELF) and severity (as per 'injuries_scale') to:
            - 'main_path'/output/all_losses_human_severity_X_after_RLA_incremental.csv:
            Incremental human losses after each earthquake for which a rapid loss assessment
            (RLA) was run ("incremental" is used herein to indicate the contribution of each
            earthquake to the human loss results; incremental human losses can only be equal to
            or larger than zero).
            - 'main_path'/output/all_losses_human_severity_X_after_RLA_cumulative.csv:
            Cumulative human losses after each earthquake for which a rapid loss assessment
            (RLA) was run ("cumulative" is used herein to indicate that the output human loss
            results from all earthquakes run until that point in time).
            - 'main_path'/output/all_losses_human_severity_X_after_OELF_incremental.csv:
            Incremental human losses after each earthquake forecast for which an operational
            earthquake loss forecasting (OELF) calculation was run. In the case of OELF,
            "incremental" means with respect to the last "real" (RLA) earthquakes run, not
            between earthquake forecasts.

        No file is exported when the list of earthquakes is empty.

        Args:
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.
            injuries_scale (list of str):
                Scale of severity of injuries. E.g., ["1","2","3","4"].
            list_rla (list of str, can be empty):
                List of names of the RLA earthquakes that have been processed.
            list_oelf (list of str, can be empty):
                List of names of the OELF earthquakes that have been processed.
        """

        # RLA human losses output
        if len(list_rla) > 0:
            collected_rla = PostProcessor._collect_output_losses_human(
                os.path.join(main_path, "output"),
                injuries_scale,
                list_rla,
                "losses_human_after_RLA_%s.csv",
            )
            for severity in collected_rla:
                collected_rla[severity].to_csv(
                    os.path.join(
                        main_path,
                        "output",
                        "all_losses_human_severity_%s_after_RLA_incremental.csv" % (severity),
                    ),
                    index=True,
                )
                collected_rla_cumulative = PostProcessor._get_cumulative_from_incremental(
                    collected_rla[severity], list_rla
                )
                collected_rla_cumulative.to_csv(
                    os.path.join(
                        main_path,
                        "output",
                        "all_losses_human_severity_%s_after_RLA_cumulative.csv" % (severity),
                    ),
                    index=True,
                )

        # OELF human losses output
        if len(list_oelf) > 0:
            collected_oelf = PostProcessor._collect_output_losses_human(
                os.path.join(main_path, "output"),
                injuries_scale,
                list_oelf,
                "losses_human_after_OELF_%s.csv",
            )
            for severity in collected_rla:
                collected_oelf[severity].to_csv(
                    os.path.join(
                        main_path,
                        "output",
                        "all_losses_human_severity_%s_after_OELF_incremental.csv" % (severity),
                    ),
                    index=True,
                )

    @staticmethod
    def _collect_output_losses_human(path, injuries_scale, list_earthquakes, filename_pattern):
        """
        This method opens all individual output files of human losses asociated with each
        earthquake or earthquake forecast listed in 'list_earthquakes' and groups all results
        into one output dictionary with one DataFrame per severity level in 'injuries_scale'.
        The inidividual output files are sought within 'path' and are
        assumed to be named following 'filename_pattern'.

        If 'list_earthquakes' is empty, the output is an empty dictionary.

        E.g.:
            path = "sth"
            list_earthquakes = ["EQ_01", "EQ_02"]
            filename_pattern = "losses_human_after_RLA_%s.csv"

            This method will collect results from files "sth/losses_human_after_RLA_EQ_01.csv"
            and "sth/losses_human_after_RLA_EQ_02.csv". Each of these files will contain columns
            as per 'injuries_scale'.

        Args:
            path (str):
                Path where individual output files will be sought.
            injuries_scale (list of str):
                Scale of severity of injuries. E.g., ["1","2","3","4"].
            list_earthquakes (list of str):
                List of earthquake names to be sought.
            filename_pattern (str):
                Pattern of the filenames to be sought. E.g. "losses_economic_after_RLA_%s.csv"
                (see example above).

        Returns:
            collected_output (dict of Pandas DataFrame):
                Dictionary with one key per severity listed in 'injuries_scale'. Each key
                contains a Pandas DataFrame with human loss results from all earthquakes or
                earthquake forecasts listed in 'list_earthquakes', with the following structure:
                    Index:
                        building_id (str): ID of the building.
                    Columns named as per each of the elements in 'list_earthquakes' (float):
                        Human loss associated with 'building_id' and the earthquake associated
                        with the column name.
        """

        collected_output = {}  # initialise, for case 'filenames' is empty
        if len(list_earthquakes) > 0:
            for severity in injuries_scale:
                collected_output[severity] = None

        filenames = [
            filename_pattern % (cat_name) for cat_name in list_earthquakes
        ]

        for i, filename in enumerate(filenames):
            indiv_output = pd.read_csv(os.path.join(path, filename))
            indiv_output.set_index(
                indiv_output["building_id"], drop=True, inplace=True
            )
            indiv_output = indiv_output.drop(columns=["building_id"])

            for severity in injuries_scale:
                severity_name = "injuries_%s" % (severity)

                # Keep only the column associated with this severity level
                indiv_output_severity = indiv_output[[severity_name]]
                # Rename the column to the name of the earthquake
                indiv_output_severity = indiv_output_severity.rename(
                    columns={severity_name: list_earthquakes[i]}
                )

                if i == 0:
                    collected_output[severity] = deepcopy(indiv_output_severity)
                else:
                    collected_output[severity] = pd.concat(
                        [collected_output[severity], indiv_output_severity],
                        axis=1
                    )
                    # Transform NaNs to zero
                    collected_output[severity] = collected_output[severity].fillna(0)

        return collected_output

    @staticmethod
    def _get_incremental_from_cumulative(cumulative, list_earthquakes):
        """
        This method calculates incremental change in the columns of 'cumulative', "walked
        through" in the order indicated by 'list_earthquakes'.

        Args:
            cumulative (Pandas DataFrame):
                DataFrame whose columns are enumerated in 'list_earthquakes'. No other
                conditions apply.
            list_earthquakes (list of str):
                List of earthquake names, ordered chronologically. They need to be the columns
                of 'cumulative'.

        Returns:
            incremental (Pandas DataFrame):
                DataFrame with same structure as 'cumulative', but whose contents are the
                difference between columns (in the order of 'list_earthquakes').
        """

        if len(list_earthquakes) < 1:
            return None

        incremental = deepcopy(cumulative)

        for i, earthquake_i in enumerate(list_earthquakes):
            if i == 0:
                continue

            incremental[earthquake_i] = (
                cumulative[earthquake_i] - cumulative[list_earthquakes[i-1]]
            )

        return incremental

    @staticmethod
    def _get_cumulative_from_incremental(incremental, list_earthquakes):
        """
        This method calculates cumulative values in the columns of 'incremental', "walked
        through" in the order indicated by 'list_earthquakes'.

        Args:
            incremental (Pandas DataFrame):
                DataFrame whose columns are enumerated in 'list_earthquakes'. No other
                conditions apply.
            list_earthquakes (list of str):
                List of earthquake names, ordered chronologically. They need to be the columns
                of 'incremental'.

        Returns:
            cumulative (Pandas DataFrame):
                DataFrame with same structure as 'incremental', but whose contents are the
                cumulative addition of columns in the order of 'list_earthquakes'.
        """

        if len(list_earthquakes) < 1:
            return None

        cumulative = deepcopy(incremental)

        for i, earthquake_i in enumerate(list_earthquakes):
            if i == 0:
                continue

            cumulative[earthquake_i] = (
                incremental[earthquake_i] + cumulative[list_earthquakes[i-1]]
            )

        return cumulative
