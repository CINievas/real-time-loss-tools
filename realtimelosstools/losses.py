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


logger = logging.getLogger()


class EconomicLosses:
    """This class handles methods associated with calculating economic losses.
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
                        One per damage state (float): They contain the mean loss ratios for
                        each building class and damage state.
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
            loss_ratio = loss_summary.loc[row, loss_summary.loc[row, "damage_state"]]
            losses[i] = loss_ratio * loss_summary.loc[row, "structural"]
        loss_summary["loss"] = losses

        loss_summary = loss_summary.groupby(["building_id"]).sum(numeric_only=True)

        loss_summary = loss_summary[["loss"]]

        return loss_summary
