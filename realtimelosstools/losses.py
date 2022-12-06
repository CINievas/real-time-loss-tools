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
from copy import deepcopy
import numpy as np
import pandas as pd


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
                    night, day, transit (float):
                        Total number of occupants in this asset at different times of the day.
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
