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


class ExposureUpdater:
    """This class handles methods associated with the updating of the exposure model to include
    damage states.
    """

    @staticmethod
    def create_mapping_asset_id_building_id(exposure):
        """
        This method retrieves the connection between asset_id and building_id from 'exposure'.

        The difference between "asset_id" and "building_id" is that "building_id" represents a
        physical entity, which can be either one "real" building or an aggregation of buildings
        at a location (e.g. the centre of a tile), while "asset_id" is the ID of each individual
        row in the exposure CSV input for OpenQuake. An "asset_id" corresponds to a particular
        combination of a "building_id" and a building class, because a "real" building can be
        associated with different building classes with different probabilities (i.e.
        probabilities of them being the "correct" class), in the same way that an aggregation
        of buildings at a location can be disaggregated into different classes.

        Args:
            exposure (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake. It
                comprises the following fields:
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

        Returns:
            id_asset_building_mapping (Pandas DataFrame):
                Pandas DataFrame with the mapping between asset_id (index of the DataFrame) and
                building_id (column) for the buildings in damage_results_SHM.
        """

        if not exposure.index.name == "asset_id":
            error_message = (
                "Method 'create_mapping_asset_id_building_id' cannot run because the index of "
                "the input 'exposure' is different from 'asset_id'. Index name is %s."
                % (exposure.index.name)
            )
            logger.critical(error_message)
            raise OSError(error_message)

        # Create mapping between asset_id and building_id
        id_asset_building_mapping = pd.DataFrame(
            {"building_id": exposure["building_id"]},
            index=exposure.index,  # asset_id
        )

        return id_asset_building_mapping

    @staticmethod
    def merge_damage_results_OQ_SHM(
        damage_results_OQ, damage_results_SHM, id_asset_building_mapping
    ):
        """
        This method merges the damage probability results from OpenQuake (OQ) and Structural
        Health Monitoring (SHM). Whenever SHM results are available, these are adopted;
        otherwise the OpenQuake results are selected. The mapping 'id_asset_building_mapping' is
        needed because 'damage_results_OQ' is defined in terms of the asset_id, while
        'damage_results_SHM' is defined in terms of the building_id.

        The difference between "asset_id" and "building_id" is that "building_id" represents a
        physical entity, which can be either one "real" building or an aggregation of buildings
        at a location (e.g. the centre of a tile), while "asset_id" is the ID of each individual
        row in the exposure CSV input for OpenQuake. An "asset_id" corresponds to a particular
        combination of a "building_id" and a building class, because a "real" building can be
        associated with different building classes with different probabilities (i.e.
        probabilities of them being the "correct" class), in the same way that an aggregation
        of buildings at a location can be disaggregated into different classes.

        If the same "building_id" is associated with several "asset_id" in
        'id_asset_building_mapping' and this "building_id" has available SHM results in
        'damage_results_SHM', the results from 'damage_results_SHM' are divided by the number of
        associated "asset_id" when adopting the SHM values for this "building_id". The need for
        this workaround arises from the fact that SHM will always treat a "building_id" as one
        building while the different damage states that this building may be in can easily lead
        to this building being represented by different asset IDs (e.g. "ClassA/DS0",
        "ClassA/DS1" and so on).

        Args:
            damage_results_OQ (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state. This is output from running OpenQuake. It comprises the following
                fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'asset_id'.
                        (Columns "loss_type" and "rlz", which are part of OpenQuake's output,
                        are not used).
            damage_results_SHM (Pandas DataFrame):
                Pandas DataFrame with probabilities of monitored buildings being in each damage
                state. This is output from SHM activities. It comprises the following fields:
                    Index is multiple:
                        building_id (str):
                            ID of the building.
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'building_id'.
            id_asset_building_mapping (Pandas DataFrame):
                Pandas DataFrame with the mapping between 'asset_id' (index of the DataFrame)
                and 'building_id' (column) for the buildings in 'damage_results_SHM'.

        Returns:
            damage_results_merged (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state. These are taken from SHM when they exist and from OpenQuake
                otherwise. It comprises the following fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'asset_id'.
        """

        # Start from the OQ results
        damage_results_merged = deepcopy(damage_results_OQ)

        # Get list of damage states
        unique_damage_states = damage_results_SHM.index.get_level_values("dmg_state").unique()

        # Go one by one each asset_id
        for asset_id in id_asset_building_mapping.index:
            building_id = id_asset_building_mapping.loc[asset_id, "building_id"]
            # If this building_id has SHM results, take them
            if building_id in damage_results_SHM.index.get_level_values(0):
                how_many_asset_ids = id_asset_building_mapping[
                    id_asset_building_mapping.building_id == building_id
                ].shape[0]
                for damage_state in unique_damage_states:
                    damage_results_merged.loc[(asset_id, damage_state), "value"] = (
                        damage_results_SHM.loc[(building_id, damage_state), "value"]
                        / how_many_asset_ids
                    )

        return damage_results_merged

    @staticmethod
    def update_exposure(
        previous_exposure_model,
        original_exposure_model,
        damage_results_OQ,
        damage_results_SHM,
        mapping_damage_states,
    ):
        """
        This method creates the exposure model for the next earthquake in the sequence, starting
        from the exposure model for the previous earthquake ('previous_exposure_model') and its
        associated damage results ('damage_results_OQ' from OpenQuake and 'damage_results_SHM'
        from Structural Health Monitoring).

        Args:
            previous_exposure_model (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake whose
                damage results are contained in 'damage_results_OQ'. It comprises the following
                fields:
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
            original_exposure_model (Pandas DataFrame):
                Pandas DataFrame representation of the undamaged (or starting-point) exposure
                model. The format and fields is the same as for 'previous_exposure_model'.
            damage_results_OQ (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state. This is output from running OpenQuake. It comprises the following
                fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'asset_id'.
                        (Columns "loss_type" and "rlz", which are part of OpenQuake's output,
                        are not used).
            damage_results_SHM (Pandas DataFrame):
                Pandas DataFrame with probabilities of monitored buildings being in each damage
                state. This is output from SHM activities. It comprises the following fields:
                    Index is multiple:
                        building_id (str):
                            ID of the building.
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'building_id'.
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

        Returns:
            new_exposure_model (Pandas DataFrame):
                Pandas DataFrame with the updated exposure model. The rows are ordered by
                'asset_id' and 'dmg_state' in ascending order. It comprises the following
                fields:
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
        """

        # Create mapping between asset_id (used by OQ) and building_id (used by SHM)
        id_asset_building_mapping = ExposureUpdater.create_mapping_asset_id_building_id(
            previous_exposure_model
        )

        # Replace probabilities in damage_results_OQ by probabilities from damage_results_SHM
        # for buildings that are monitored
        damage_results_merged = ExposureUpdater.merge_damage_results_OQ_SHM(
            damage_results_OQ, damage_results_SHM, id_asset_building_mapping
        )

        # Create new exposure model
        new_exposure_model = damage_results_merged.join(previous_exposure_model)

        # Re-calculate costs and people
        for col_name in ["structural", "day", "night", "transit"]:
            new_exposure_model[col_name] = (
                new_exposure_model["value"].to_numpy() / new_exposure_model["number"].to_numpy()
            ) * new_exposure_model[col_name].to_numpy()

        # Replace the contents of "number" with the contents of "value"
        new_exposure_model["number"] = new_exposure_model["value"]

        # Eliminate columns "value", "rlz", "loss_type"
        new_exposure_model = new_exposure_model.drop(columns=["value", "rlz", "loss_type"])

        # Re-write the taxonomy strings
        new_taxonomies = [
            "%s/%s"
            % (
                "/".join(new_exposure_model["taxonomy"].to_numpy()[j].split("/")[:-1]),
                mapping_damage_states.loc[
                    new_exposure_model.index.get_level_values("dmg_state")[j], "fragility"
                ],
            )
            for j in range(new_exposure_model.shape[0])
        ]
        new_exposure_model["taxonomy"] = new_taxonomies

        # Group same damage states for same original_asset_id (e.g. for the same building and
        # class, two instances of "ClassA/DS1" should be grouped in the same row)
        columns_by_original_asset_id = [  # These values depend only on the original_asset_id
            "lon",
            "lat",
            "id_1",
            "id_2",
            "id_3",
            "name_1",
            "name_2",
            "name_3",
            "occupancy",
            "building_id",
        ]
        new_exposure_model = new_exposure_model.drop(columns=columns_by_original_asset_id)
        new_exposure_model = new_exposure_model.groupby(
            ["original_asset_id", "taxonomy"]
        ).sum()  # sum number of buildings, people, costs for rows that need to be grouped
        # Re-assign values of columns that only depend on original_asset_id (retrieve from the
        # original exposure model)
        for col in columns_by_original_asset_id:
            aux_cols_content = []
            for multiindex in new_exposure_model.index:
                original_asset_id = multiindex[0]
                aux_cols_content.append(original_exposure_model.loc[original_asset_id, col])
            new_exposure_model[col] = aux_cols_content

        # Eliminate rows for which the number of buildings is zero (defined as < 1E-10)
        filter_keep_nonzeros = new_exposure_model["number"] > 1e-10
        new_exposure_model = new_exposure_model[filter_keep_nonzeros]

        # Re-arrange index (up to now it is MultiIndex on (original_asset_id, taxonomy), need to
        # make it based again on ("asset_id", "dmg_state"), "asset_id" being that of the
        # previous exposure model; also need to recover contents of "taxonomy" as a column)
        new_exposure_model["taxonomy"] = new_exposure_model.index.get_level_values("taxonomy")
        new_exposure_model["dmg_state"] = [
            mapping_damage_states[
                mapping_damage_states.fragility == taxonomy_val.split("/")[-1]
            ].index[0]
            for taxonomy_val in new_exposure_model["taxonomy"]
        ]
        original_asset_id = new_exposure_model.index.get_level_values("original_asset_id")
        new_index = pd.MultiIndex.from_arrays(
            [original_asset_id, new_exposure_model["dmg_state"]]
        )
        new_exposure_model.index = new_index
        new_exposure_model.index = new_exposure_model.index.rename(["asset_id", "dmg_state"])
        new_exposure_model = new_exposure_model.drop(columns=["dmg_state"])
        new_exposure_model["original_asset_id"] = original_asset_id

        # Order the DataFrame by asset_id and dmg_state (ascending order)
        new_exposure_model = new_exposure_model.sort_values(
            by=[("asset_id"), ("dmg_state")], ascending=True
        )

        # Create new asset_id for the next calculation
        new_exposure_model["id"] = [
            "res_%s" % (j) for j in range(1, new_exposure_model.shape[0] + 1)
        ]

        # Re-order columns
        new_exposure_model = new_exposure_model.reindex(
            columns=["id", *original_exposure_model.columns]
        )

        return new_exposure_model


    @staticmethod
    def ensure_no_negative_damage_results_OQ(damage_results_OQ, tolerance=0.0001):
        """
        This method ensures that there are no negative numbers of buildings in the column
        "value" of 'damage_results_OQ', by setting them to zero and adjusting the number of
        buildings of the other damage grades so that the total of each 'asset_id' adds up to the
        original value. If the ratio of any of the negative values to the total number of
        buildings associated with that 'asset_id' is larger (in absolute value) than
        'tolerance', a ValueError is raised.

        Args:
            damage_results_OQ (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state. This is output from running OpenQuake. It comprises the following
                fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'asset_id'.
                        (Columns "loss_type" and "rlz", which are part of OpenQuake's output,
                        are not used).
            tolerance (float):
                Default: 0.0001 (= 0.01%).

        Returns:
            damage_results_OQ_adjusted (Pandas DataFrame):
                Same structure as 'damage_results_OQ' in the input, but adjusted so that there
                are no negative values in the column "value".
        """

        if np.all(damage_results_OQ.loc[:, "value"] >= 0):  # Nothing to be done
            return damage_results_OQ

        damage_results_OQ_adjusted = deepcopy(damage_results_OQ)
        
        for asset_id in damage_results_OQ_adjusted.index.get_level_values("asset_id"):
            damage_results_OQ_asset = damage_results_OQ_adjusted.loc[asset_id, "value"]

            if np.any(damage_results_OQ_asset < 0):  # There are negative values
                total_bdgs = damage_results_OQ_asset.sum()

                if abs(damage_results_OQ_asset.min()) / total_bdgs > tolerance:
                    error_message = (
                        "There are negative values in the damage results from OpenQuake "
                        "that exceed the %s tolerance. The program cannot continue running"
                        % (tolerance)
                    )
                    logger.critical(error_message)
                    raise ValueError(error_message)

                # Set negative numbers to zero
                damage_results_OQ_asset[damage_results_OQ_asset < 0] = 0
                
                # Recalculate the other values so as to keep the total number of buildings
                damage_results_OQ_asset = (
                    damage_results_OQ_asset / damage_results_OQ_asset.sum() * total_bdgs
                )

                # Transfer back to the original DataFrame (damage_results_OQ)
                damage_results_OQ_adjusted.loc[asset_id, "value"] = (
                    damage_results_OQ_asset.to_numpy()
                )

        return damage_results_OQ_adjusted