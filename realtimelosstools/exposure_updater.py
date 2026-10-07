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
from realtimelosstools.losses import Losses
from realtimelosstools.oq_handler import OpenQuakeUtils


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
                        census (float):
                            Total number of occupants in this asset irrespective of the time of
                            the day, the damage state of the building or the health status of
                            its occupants.
                        occupancy (str):
                            "Res" (residential), "Com" (commercial) or "Ind" (industrial).
                        id_X, name_X (str):
                            ID and name of the administrative units to which the asset belongs.
                            "X" is the administrative level.

        Returns:
            id_asset_building_mapping (Pandas DataFrame):
                Pandas DataFrame with the mapping between asset_id (index of the DataFrame) and
                building_id (column).
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
    def create_mapping_asset_id_to_original_asset_id(exposure):
        """
        This method retrieves the connection between 'asset_id' and 'original_asset_id' from
        'exposure'.

        The difference between "asset_id", "original_asset_id" and "building_id" is that
        "building_id" represents a physical entity, which can be either one "real" building or
        an aggregation of buildings at a location (e.g. the centre of a tile), while
        "original_asset_id" refers to a specific building class associated with the
        "building_id", and "asset_id" is the ID of each individual row in the exposure CSV input
        for OpenQuake. If "building_id" is one physical entity, each "original_asset_id"
        represents a building class that could be the building class of the building, with a
        certain probability. If "building_id" is an aggregation of buildings, each
        "original_asset_id" represents a building class to which a number of those aggregated
        buildings belong. Each "original_asset_id" may be associated with different "asset_id",
        used herein to specify different damage states of the same "original_asset_id".

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
                        census (float):
                            Total number of occupants in this asset irrespective of the time of
                            the day, the damage state of the building or the health status of
                            its occupants.
                        occupancy (str):
                            "Res" (residential), "Com" (commercial) or "Ind" (industrial).
                        id_X, name_X (str):
                            ID and name of the administrative units to which the asset belongs.
                            "X" is the administrative level.

        Returns:
            asset_id_original_asset_id_mapping (Pandas DataFrame):
                Pandas DataFrame with the mapping between asset_id (index of the DataFrame) and
                original_asset_id (column), indicating as well the number of buildings in each
                case ("number" column).
        """

        aux_df = exposure.reset_index()
        aux_df = aux_df.groupby(["asset_id", "original_asset_id"]).sum(numeric_only=True)

        asset_id_original_asset_id_mapping = pd.DataFrame(
            {
                "original_asset_id": aux_df.index.get_level_values("original_asset_id"),
                "number": aux_df["number"].to_numpy()
            },
            index=aux_df.index.get_level_values("asset_id")
        )
        asset_id_original_asset_id_mapping.index.name = "asset_id"

        if (asset_id_original_asset_id_mapping.shape[0] != len(exposure.index.unique())):
            error_message = (
                "Method 'create_mapping_original_asset_id_building_id' cannot run because "
                "input 'exposure' associates the same 'asset_id' with different "
                "values of 'original_asset_id'."
            )
            logger.critical(error_message)
            raise OSError(error_message)

        return asset_id_original_asset_id_mapping

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
            damage_results_SHM (Pandas Series):
                Pandas Series with probabilities of monitored buildings being in each damage
                state. This is output from SHM activities. It comprises the following fields:
                    Index is multiple:
                        building_id (str):
                            ID of the building.
                        dmg_state (str):
                            Damage states.
                    Values of the series (float): Probability of 'dmg_state' for 'building_id'.
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

        # Number of asset_ids associated with each building_id (computed once)
        n_assets_per_building = id_asset_building_mapping.groupby("building_id").size()

        # Keep only the asset_ids whose building_id has SHM results
        building_ids_with_shm = damage_results_SHM.index.get_level_values(0).unique()
        assets_with_shm = id_asset_building_mapping[
            id_asset_building_mapping["building_id"].isin(building_ids_with_shm)
        ]

        if assets_with_shm.shape[0] == 0:
            return damage_results_merged

        # One entry per (asset_id, damage state) to be replaced
        unique_damage_states = damage_results_SHM.index.get_level_values("dmg_state").unique()
        n_dmg = len(unique_damage_states)

        asset_ids = np.repeat(assets_with_shm.index.to_numpy(), n_dmg)
        building_ids = np.repeat(assets_with_shm["building_id"].to_numpy(), n_dmg)
        dmg_states = np.tile(unique_damage_states.to_numpy(), assets_with_shm.shape[0])

        # SHM values for each (building_id, damage state), divided by the number of asset_ids
        # that share that building_id
        shm_values = damage_results_SHM.loc[
            pd.MultiIndex.from_arrays([building_ids, dmg_states])
        ].to_numpy().astype("float32")
        how_many_asset_ids = n_assets_per_building.loc[building_ids].to_numpy().astype("float32")
        new_values = shm_values / how_many_asset_ids

        # Replace all the values in one go
        rows_to_replace = damage_results_merged.index.get_indexer(
            pd.MultiIndex.from_arrays([asset_ids, dmg_states])
        )
        if (rows_to_replace < 0).any():  # rows_to_replace will be -1 if index not found
            error_message = (
                "Method 'ExposureUpdater.merge_damage_results_OQ_SHM': some (asset_id, dmg_state) "
                "combinations with SHM results are not present in 'damage_results_OQ'."
            )
            logger.critical(error_message)
            raise ValueError(error_message)

        damage_results_merged.iloc[
            rows_to_replace, damage_results_merged.columns.get_loc("value")
        ] = new_values

        return damage_results_merged

    @staticmethod
    def get_damage_results_by_orig_asset_id(damage_results, asset_id_original_asset_id_mapping):
        """
        This method returns the damage results from 'damage_results', which are listed by their
        "asset_id", by "original_asset_id" instead. The link between "original_asset_id" and
        "asset_id" is specified in the input 'asset_id_original_asset_id_mapping'. Several
        values of "asset_id" can be associated with the same "original_asset_id", but not the
        other way around.

        Args:
            damage_results (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state. It comprises the following fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' or number of buildings in 'dmg_state' for
                            'asset_id'.
            asset_id_original_asset_id_mapping (Pandas DataFrame):
                Pandas DataFrame with the mapping between asset_id (index of the DataFrame) and
                original_asset_id (column).

        Returns:
            damage_results_by_orig_asset_id (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state. It comprises the following fields:
                    Index is multiple:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model. It can be the value of 'asset_id' in the undamaged version or
                            any other unique ID per row that refers to a combination of a
                            building ID and a building class with no initial damage.
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' or number of buildings in 'dmg_state' for
                            'original_asset_id'.
        """

        damage_results_by_orig_asset_id = damage_results.join(asset_id_original_asset_id_mapping)

        # Transform the index of damage_results_by_orig_asset_id into "normal" columns
        asset_ids = (
            damage_results_by_orig_asset_id.index.get_level_values("asset_id")
        )
        dmg_states = (
            damage_results_by_orig_asset_id.index.get_level_values("dmg_state")
        )
        damage_results_by_orig_asset_id = damage_results_by_orig_asset_id.reset_index()
        damage_results_by_orig_asset_id["asset_id"] = asset_ids
        damage_results_by_orig_asset_id["dmg_state"] = dmg_states

        # Group
        damage_results_by_orig_asset_id = damage_results_by_orig_asset_id.groupby(
            ["original_asset_id", "dmg_state"]
        ).sum(numeric_only=True)  # index becomes multiple ("original_asset_id", "dmg_state")

        # Discard unnecessary columns (if they exist)
        damage_results_by_orig_asset_id = damage_results_by_orig_asset_id[["value"]]

        return damage_results_by_orig_asset_id

    @staticmethod
    def ensure_all_damage_states(occurrence_by_orig_asset_id, mapping_damage_states):
        """
        This method ensures that 'occurrence_by_orig_asset_id' contains all damage states
        defined as keys of 'mapping_damage_states' for each original_asset_id. If a damage state
        is missing, it is appended and assigned a "value" of zero.

        Args:
            occurrence_by_orig_asset_id (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state. It comprises the following fields:
                    Index is multiple:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability or number of buildings of 'dmg_state' for 'asset_id'.
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

        Returns:
            occurrence_by_orig_asset_id_filled (Pandas DataFrame):
                Pandas DataFrame with the same structure as 'occurrence_by_orig_asset_id', but
                filled in with zeros where necessary to cover all damage states in
                'mapping_damage_states' for each 'original_asset_id'.
        """

        occurrence_by_orig_asset_id_filled = deepcopy(occurrence_by_orig_asset_id)

        original_asset_ids = occurrence_by_orig_asset_id.index.get_level_values(
            "original_asset_id"
        ).unique()

        for original_asset_id in original_asset_ids:
            current_dmg_states = occurrence_by_orig_asset_id_filled.loc[
                original_asset_id, :
            ].index.get_level_values("dmg_state")

            for dmg_state in mapping_damage_states.index:
                if dmg_state not in current_dmg_states:
                    # Append it to 'occurrence_by_orig_asset_id_filled'
                    to_append = pd.DataFrame(
                        {"value": [0.0]},
                        index=pd.MultiIndex.from_tuples(
                            [(original_asset_id, dmg_state)],
                            names=["original_asset_id", "dmg_state"]
                        )
                    )

                    occurrence_by_orig_asset_id_filled = pd.concat(
                        (occurrence_by_orig_asset_id_filled, to_append)
                    )

        # Sort because appending would result in different damage states of the same
        # original_asset_id to be represented separately
        occurrence_by_orig_asset_id_filled = occurrence_by_orig_asset_id_filled.sort_values(
            by=[("original_asset_id"), ("dmg_state")], ascending=True
        )

        return occurrence_by_orig_asset_id_filled

    @staticmethod
    def get_non_exceedance_by_orig_asset_id(occurrence_by_orig_asset_id, mapping_damage_states):
        """
        This method calculates the probability of non-exceedance of each damage state by each
        'original_asset_id' in 'occurrence_by_orig_asset_id'. It first ensures that
        'occurrence_by_orig_asset_id' contains all damage states defined as keys of
        'mapping_damage_states' for each 'original_asset_id'. If a damage state is missing, it
        is appended and assigned an occurrence value of zero. 'occurrence_by_orig_asset_id'
        contains either probabilities of occurence, when 'original_asset_id' refers to an
        individual building, or number of buildings in each damage grade, when
        'original_asset_id' refers to a group of buildings.

        It is assumed that 'mapping_damage_states' lists the damage states ordered from least
        severe to most severe. This assumption is needed to calculate exceedance/non-exceedance.

        Args:
            occurrence_by_orig_asset_id (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state. It comprises the following fields:
                    Index is multiple:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability or number of buildings of 'dmg_state' for 'asset_id'.
            mapping_damage_states (Pandas DataFrame):
                Mapping between the names of damage states as output by OpenQuake (index) and as
                labelled in the fragility model (value), ordered from least severe to most
                severe. E.g.:
                              fragility
                    dmg_state
                    no_damage       DS0
                    dmg_1           DS1
                    dmg_2           DS2
                    dmg_3           DS3
                    dmg_4           DS4

        Returns:
            prob_non_exceedance (Pandas DataFrame):
                Pandas DataFrame with probabilities of non-exceedance of each damage state by
                each 'original_asset_id'. It comprises the following fields:
                    Index is multiple:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        dmg_state (str):
                            Damage states.
                    Columns:
                        prob_non_exceedance (float):
                            Probability of non-exceedance of 'dmg_state' for
                            'original_asset_id'.
        """

        prob_non_exceedance = deepcopy(occurrence_by_orig_asset_id)

        # Check all damage states from mapping_damage_states exist for each original_asset_id in
        # occurrence_by_orig_asset_id
        prob_non_exceedance = ExposureUpdater.ensure_all_damage_states(
            prob_non_exceedance, mapping_damage_states
        )

        # Initialise
        prob_non_exceedance["prob_occurrence"] = np.zeros([prob_non_exceedance.shape[0]])
        prob_non_exceedance["prob_exceedance"] = np.zeros([prob_non_exceedance.shape[0]])

        original_asset_ids = (
            prob_non_exceedance.index.get_level_values("original_asset_id").unique()
        )

        for original_asset_id in original_asset_ids:
            # Get probability of occurrence (redundant if original_asset_id is already one
            # building but necessary if original_asset_id is a group of buildings
            total_bdgs = prob_non_exceedance.loc[original_asset_id, "value"].sum()
            prob_non_exceedance.loc[original_asset_id, "prob_occurrence"] = (
                prob_non_exceedance.loc[original_asset_id, "value"].to_numpy() / total_bdgs
            )

            # Get probability of exceedance
            remaining_dmg_states = list(mapping_damage_states.index)

            for dmg_state in mapping_damage_states.index:  # ordered from least to most severe
                if dmg_state == "no_damage":
                    prob_non_exceedance.loc[
                        (original_asset_id, dmg_state), "prob_exceedance"
                    ] = 1.0
                else:
                    prob_non_exceedance.loc[
                        (original_asset_id, dmg_state), "prob_exceedance"
                    ] = (
                        prob_non_exceedance.loc[
                            (original_asset_id, remaining_dmg_states), "prob_occurrence"
                        ].sum()
                    )
                remaining_dmg_states.remove(dmg_state)

        # Get probability of non-exceedance
        prob_non_exceedance["prob_non_exceedance"] = (
            np.ones(prob_non_exceedance.shape[0])
            - prob_non_exceedance["prob_exceedance"].to_numpy()
        )

        # Keep only the probability of non-exceedance
        prob_non_exceedance = prob_non_exceedance[["prob_non_exceedance"]]

        return prob_non_exceedance

    @staticmethod
    def get_prob_occurrence_from_independent_non_exceedance(
        prob_nonexceed_by_orig_asset_id_previous,
        prob_nonexceed_by_orig_asset_id_current,
        asset_id_original_asset_id_mapping,
        mapping_damage_states,
    ):
        """
        This method calculates the probability of occurrence of each damage state in
        'mapping_damage_states' by each 'original_asset_id' in
        'prob_nonexceed_by_orig_asset_id_previous' and
        'prob_nonexceed_by_orig_asset_id_current', which contain probabilities of non-exceedance
        of the damage states for a previous and current earthquake, respectively. These two
        realisations of probabilities of exceedance are assumed to be independent. Using the
        total number of buildings indicated in 'mapping_damage_states', the method also
        calculates the number of buildings in each damage state.

        It is assumed that 'mapping_damage_states' lists the damage states ordered from least
        severe to most severe. This assumption is needed to calculate the probabilities of
        occurrence.

        Args:
            prob_nonexceed_by_orig_asset_id_previous (Pandas DataFrame):
                Pandas DataFrame with probabilities of non-exceedance of each damage state by
                each 'original_asset_id', due to a previous earthquake. It comprises the
                following fields:
                    Index is multiple:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        dmg_state (str):
                            Damage states.
                    Columns:
                        prob_non_exceedance (float):
                            Probability of non-exceedance of 'dmg_state' for
                            'original_asset_id'.
            prob_nonexceed_by_orig_asset_id_current (Pandas DataFrame):
                Same as prob_nonexceed_by_orig_asset_id_previous, but due to a current
                earthquake.
            asset_id_original_asset_id_mapping (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class and damage state).
                    Columns:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        number (float):
                            Number of buildings associated with 'asset_id'.
            mapping_damage_states (Pandas DataFrame):
                Mapping between the names of damage states as output by OpenQuake (index) and as
                labelled in the fragility model (value), ordered from least severe to most
                severe. E.g.:
                              fragility
                    dmg_state
                    no_damage       DS0
                    dmg_1           DS1
                    dmg_2           DS2
                    dmg_3           DS3
                    dmg_4           DS4

        Returns:
            prob_of_occurrence (Pandas DataFrame):
                Pandas DataFrame with probabilities of occurrence of each damage state by each
                'original_asset_id' and associated number of buildings. It comprises the
                following fields:
                    Index is multiple:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        dmg_state (str):
                            Damage states.
                    Columns:
                        prob_occurrence_cumulative (float):
                            Probability of occurrence of 'dmg_state' for 'original_asset_id'.
                        number_occurrence_cumulative (float):
                            Number of buildings associated with 'dmg_state' and
                            'original_asset_id'.
        """

        prob_of_occurrence = prob_nonexceed_by_orig_asset_id_current.join(
            prob_nonexceed_by_orig_asset_id_previous,
            lsuffix="_current",
            rsuffix="_previous"
        )

        prob_of_occurrence["prob_non_exceedance_cumulative"] = (
            prob_of_occurrence["prob_non_exceedance_current"]
            * prob_of_occurrence["prob_non_exceedance_previous"]
        )

        prob_of_occurrence["prob_exceedance_cumulative"] = (
            np.ones([prob_of_occurrence.shape[0]])
            - prob_of_occurrence["prob_non_exceedance_cumulative"]
        )

        # Number of buildings per 'original_asset_id'
        number_bdgs = asset_id_original_asset_id_mapping.groupby(["original_asset_id"]).sum(
            numeric_only=True
        )

        # Get probability of occurrence
        original_asset_ids = (
            prob_of_occurrence.index.get_level_values("original_asset_id").unique()
        )

        prob_of_occurrence["prob_occurrence_cumulative"] = np.zeros(
            [prob_of_occurrence.shape[0]]
        )
        prob_of_occurrence["number_occurrence_cumulative"] = np.zeros(
            [prob_of_occurrence.shape[0]]
        )

        for original_asset_id in original_asset_ids:
            for i, dmg_state in enumerate(mapping_damage_states.index):
                # (ordered from least to most severe)
                if i == (mapping_damage_states.shape[0] - 1):
                    # Most severe damage state --> PoO = PoE
                    prob_of_occurrence.loc[
                        (original_asset_id, dmg_state), "prob_occurrence_cumulative"
                    ] = (
                        prob_of_occurrence.loc[
                            (original_asset_id, dmg_state), "prob_exceedance_cumulative"
                        ]
                    )
                else:
                    prob_of_occurrence.loc[
                        (original_asset_id, dmg_state), "prob_occurrence_cumulative"
                    ] = (
                        prob_of_occurrence.loc[
                            (original_asset_id, dmg_state), "prob_exceedance_cumulative"
                        ]
                        - prob_of_occurrence.loc[
                            (original_asset_id, mapping_damage_states.index[i+1]),
                            "prob_exceedance_cumulative"
                        ]
                    )
                prob_of_occurrence.loc[
                    (original_asset_id, dmg_state), "number_occurrence_cumulative"
                ] = (
                    prob_of_occurrence.loc[
                        (original_asset_id, dmg_state), "prob_occurrence_cumulative"
                    ]
                    * number_bdgs.loc[original_asset_id, "number"]
                )

        prob_of_occurrence = prob_of_occurrence[
            ["prob_occurrence_cumulative", "number_occurrence_cumulative"]
        ]

        return prob_of_occurrence

    @staticmethod
    def update_damage_results(
        damage_results_original,
        damage_occurrence_by_orig_asset_id,
        asset_id_original_asset_id_mapping
    ):
        """
        This method updates 'damage_results_original', which specifies the number of buildings
        per 'asset_id' and damage state ('dmg_state'), as per the damage specified in
        'damage_occurrence_by_orig_asset_id' as a function of 'original_asset_id' and damage
        state ('dmg_state'). The mapping between 'asset_id' and 'original_asset_id' is provided
        by 'asset_id_original_asset_id_mapping'. The number of buildings for a specific
        combination of 'original_asset_id' and 'dmg_state' are distributed across all 'asset_id'
        associated with 'original_asset_id' proportionally to the original values in
        'damage_results_original'. If, for a certain 'original_asset_id' and 'dmg_state', all
        the original values in 'damage_results_original' are zero, the new number of buildings
        for the damage state is equally distributed across all the associated 'asset_id'.

        Args:
            damage_results_original (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state for each asset_id. It comprises the following fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class and initial damage state).
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'asset_id', or number of buildings of
                            'asset_id' associated with 'dmg_state'.
            damage_occurrence_by_orig_asset_id (Pandas DataFrame):
                Pandas DataFrame with probabilities of occurrence of each damage state by each
                'original_asset_id' and associated number of buildings. It comprises the
                following fields:
                    Index is multiple:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        dmg_state (str):
                            Damage states.
                    Columns:
                        prob_occurrence_cumulative (float):
                            Probability of occurrence of 'dmg_state' for 'original_asset_id'.
                        number_occurrence_cumulative (float):
                            Number of buildings associated with 'dmg_state' and
                            'original_asset_id'.
            asset_id_original_asset_id_mapping (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class and damage state).
                    Columns:
                        original_asset_id (str):
                            ID of the asset in the initial undamaged version of the exposure
                            model.
                        number (float):
                            Number of buildings associated with 'asset_id'.
        Returns:
            damage_results_updated (Pandas DataFrame):
                Pandas DataFrame with the same structure as 'damage_results_original', but
                updated values.
        """

        damage_results_updated = deepcopy(damage_results_original)

        original_asset_ids = damage_occurrence_by_orig_asset_id.index.get_level_values(
            "original_asset_id"
        ).unique()

        dmg_states = damage_occurrence_by_orig_asset_id.index.get_level_values(
            "dmg_state"
        ).unique()

        for original_asset_id in original_asset_ids:
            asset_ids = asset_id_original_asset_id_mapping[
                asset_id_original_asset_id_mapping.original_asset_id == original_asset_id
            ].index

            for dmg_state in dmg_states:
                denominator = damage_results_original.loc[(asset_ids, dmg_state), "value"].sum()
                if denominator > 1E-15:
                    proportions = (
                        damage_results_original.loc[(asset_ids, dmg_state), "value"]
                        / denominator
                    )  # result is Pandas Series
                else:  # sum of buildings is zero
                    proportions = pd.Series(
                        np.ones([len(asset_ids)]) / float(len(asset_ids)),
                    )
                    new_index = pd.MultiIndex.from_arrays(
                        [
                            asset_ids,
                            [dmg_state for d in range(len(asset_ids))]
                        ],
                        names=["asset_id", "dmg_state"]
                    )
                    proportions.index = new_index

                for asset_id in asset_ids:
                    damage_results_updated.loc[(asset_id, dmg_state), "value"] = (
                        proportions.loc[(asset_id, dmg_state)]
                        * damage_occurrence_by_orig_asset_id.loc[
                            (original_asset_id, dmg_state), "number_occurrence_cumulative"
                        ]
                    )

        return damage_results_updated

    @staticmethod
    def update_exposure_with_damage_states(
        state_dependent,
        previous_exposure_model,
        original_exposure_model,
        damage_results_OQ,
        mapping_damage_states,
        earthquake_time_of_day,
        damage_results_SHM=None,
    ):
        """
        This method creates the exposure model for the next earthquake in the sequence, starting
        from the exposure model for the previous earthquake ('previous_exposure_model') and its
        associated damage results ('damage_results_OQ' from OpenQuake and 'damage_results_SHM'
        from Structural Health Monitoring, if provided). The existing assets are distributed to
        different damage states as per 'damage_results_OQ' and 'damage_results_SHM', and their
        repair costs and occupants are distributed accordingly, without yet updating the number
        of occupants to reflect injuries and deaths.

        If 'state_dependent' is True, it assumes that 'damage_results_OQ' and
        'damage_results_SHM' stem from state-dependent fragility models and thus represent
        cumulative damage without need of further processing. If 'state_dependent' is False, it
        assumes that 'damage_results_OQ' and 'damage_results_SHM' stem from state-independent
        fragility models and thus represent damage due only to the last earthquake, assuming
        undamaged previous conditions. In this last case, the probabilities of not exceeding
        each damage state due to the current and previous earthquakes are combined (considering
        them statistically independent) to calculate the probability of not exceeding the damage
        states aftere both earthquakes and, from there, the probability of occurrence of each
        damage state.

        Args:
            state_dependent (bool):
                If True, it is assumed that the damage results have been calculated using state-
                dependent fragility models. If False, it is assumed that the damage results have
                been calculated using state-independent fragility models.
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
                        census (float):
                            Total number of occupants in this asset irrespective of the time of
                            the day, the damage state of the building or the health status of
                            its occupants.
                        earthquake_time_of_day (float):
                            Total number of occupants in this asset at the time indicated by the
                            input string 'earthquake_time_of_day'.
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
            earthquake_time_of_day (str):
                Time of the day at which the earthquake occurs: "day", "night" or "transit".
            damage_results_SHM (Pandas Series):
                Pandas Series with probabilities of monitored buildings being in each damage
                state. This is output from SHM activities. It comprises the following fields:
                    Index is multiple:
                        building_id (str):
                            ID of the building.
                        dmg_state (str):
                            Damage states.
                    Values of the series (float): Probability of 'dmg_state' for 'building_id'.

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
                        census (float):
                            Total number of occupants in this asset irrespective of the time of
                            the day, the damage state of the building or the health status of
                            its occupants.
                        earthquake_time_of_day (float):
                            Total number of occupants in this asset at the time indicated by the
                            input string 'earthquake_time_of_day', irrespective of the damage
                            state of the building or the health status of its occupants.
                        occupancy (str):
                            "Res" (residential), "Com" (commercial) or "Ind" (industrial).
                        id_X, name_X (str):
                            ID and name of the administrative units to which the asset belongs.
                            "X" is the administrative level.
        """

        assert original_exposure_model["original_asset_id"].is_unique

        # Replace probabilities in damage_results_OQ by probabilities from damage_results_SHM
        # for buildings that are monitored
        if damage_results_SHM is not None:
            logger.debug(
                "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
                "merging externally-input damage states and OpenQuake damage states"
                % (np.datetime64('now'))
            )
            # Create mapping between asset_id (used by OQ) and building_id (used by SHM)
            id_asset_building_mapping = ExposureUpdater.create_mapping_asset_id_building_id(
                previous_exposure_model
            )

            damage_results_merged = ExposureUpdater.merge_damage_results_OQ_SHM(
                damage_results_OQ, damage_results_SHM, id_asset_building_mapping
            )
        else:
            damage_results_merged = deepcopy(damage_results_OQ)

        # When using state-independent fragilities, 'damage_results_merged' needs adjustments
        if state_dependent:  # State-dependent fragility model assumed
            # No need to update damage states, take 'damage_results_merged' as it is
            damage_results_merged_updated = deepcopy(damage_results_merged)
        else:  # State-independent fragility model assumed, cumulative damage to be calculated
            logger.debug(
                "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
                "carrying out preliminary steps for using state-independent fragilities"
                % (np.datetime64('now'))
            )
            # Damage results (probab. of occurrence) by original_asset_id (instead of by
            # asset_id) for the current earthquake (the one just run)
            asset_id_original_asset_id_mapping = (
                ExposureUpdater.create_mapping_asset_id_to_original_asset_id(
                    previous_exposure_model
                )
            ) # DataFrame with asset_id as index and original_asset_id as values
            damage_results_by_orig_asset_id_current = (
                ExposureUpdater.get_damage_results_by_orig_asset_id(
                    damage_results_merged, asset_id_original_asset_id_mapping
                )
            )

            # Probability of non-exceedance by original_asset_id of this earthquake
            damage_prob_nonexceed_by_orig_asset_id_current = (
                ExposureUpdater.get_non_exceedance_by_orig_asset_id(
                    damage_results_by_orig_asset_id_current, mapping_damage_states
                )
            )

            # Damage results (probab. of occurrence) by asset_id for previous earthquake
            # (implicitly contained in the input exposure model)
            damage_results_by_asset_id_previous = (
                OpenQuakeUtils.create_OQ_existing_damage(
                    previous_exposure_model,
                    mapping_damage_states,
                    loss_type="structural"
                )
            )
            new_index = pd.MultiIndex.from_arrays(
                [
                    damage_results_by_asset_id_previous["asset_id"],
                    damage_results_by_asset_id_previous["dmg_state"]
                ]
            )
            damage_results_by_asset_id_previous.index = new_index
            damage_results_by_asset_id_previous = (
                damage_results_by_asset_id_previous.drop(columns=["asset_id", "dmg_state"])
            )

            # Damage results (probab. of occurrence) by original_asset_id for previous earthquake
            damage_results_by_orig_asset_id_previous = (
                ExposureUpdater.get_damage_results_by_orig_asset_id(
                    damage_results_by_asset_id_previous, asset_id_original_asset_id_mapping
                )
            )

            # Probability of non-exceedance by original_asset_id of previous earthquake
            damage_prob_nonexceed_by_orig_asset_id_previous = (
                ExposureUpdater.get_non_exceedance_by_orig_asset_id(
                    damage_results_by_orig_asset_id_previous, mapping_damage_states
                )
            )

            # Calculate probability (and numbers) of occurrence due to cumulative probabilities
            # (this earthquake and previous earthquake)
            damage_occurrence_by_orig_asset_id_current = (
                ExposureUpdater.get_prob_occurrence_from_independent_non_exceedance(
                    damage_prob_nonexceed_by_orig_asset_id_previous,
                    damage_prob_nonexceed_by_orig_asset_id_current,
                    asset_id_original_asset_id_mapping,
                    mapping_damage_states,
                )
            )

            # Update 'damage_results_merged' with new numbers of occurrence
            damage_results_merged_updated = (
                ExposureUpdater.update_damage_results(
                    damage_results_merged,
                    damage_occurrence_by_orig_asset_id_current,
                    asset_id_original_asset_id_mapping
                )
            )

        # Eliminate rows for which the number of buildings is zero (defined as < 1E-10)
        # (it is faster to do it on the damage results than after merging the damage results
        # with the previous exposure model; the fundamental concept behind this is that
        # buildings cannot heal themselves, but OQ will still output the rows associated with
        # "healing", i.e. going backwards in damage, and these rows will have zero buildings)
        logger.debug(
            "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
            "eliminating assets with number of buildings <= 1e-10"
            % (np.datetime64('now'))
        )
        filter_keep_nonzeros = damage_results_merged_updated["value"] > 1e-10
        damage_results_merged_updated = damage_results_merged_updated[filter_keep_nonzeros]

        # Create new exposure model
        logger.debug(
            "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
            "joining previous exposure model with damage results"
            % (np.datetime64('now'))
        )
        # Detach the damage state from the building class string (it is faster to do it before
        # joining with the new damage states)
        previous_exposure_model_without_ds = previous_exposure_model.assign(
            taxonomy=previous_exposure_model["taxonomy"].str.rpartition("/")[0]
        )

        new_exposure_model = damage_results_merged_updated.join(
            previous_exposure_model_without_ds
        )

        # Re-calculate costs and people
        logger.debug(
            "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
            "distributing costs and occupants across different damage states"
            % (np.datetime64('now'))
        )
        for col_name in ["structural", "census", earthquake_time_of_day]:
            if col_name in new_exposure_model.columns:
                new_exposure_model[col_name] = (
                    new_exposure_model["value"].to_numpy() /
                    new_exposure_model["number"].to_numpy()
                ) * new_exposure_model[col_name].to_numpy()

        # Replace the contents of "number" with the contents of "value"
        new_exposure_model["number"] = new_exposure_model["value"]

        # Eliminate columns "value", "rlz", "loss_type"
        new_exposure_model = new_exposure_model.drop(columns=["value", "rlz", "loss_type"])

        # Re-write the taxonomy strings
        logger.debug(
            "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
            "re-generating taxonomy strings"
            % (np.datetime64('now'))
        )
        dmg_state = new_exposure_model.index.get_level_values("dmg_state")
        fragility = pd.Series(
            dmg_state.map(mapping_damage_states["fragility"]),
            index=new_exposure_model.index,
        )
        new_exposure_model["taxonomy"] = new_exposure_model["taxonomy"] + "/" + fragility

        # Group same damage states for same original_asset_id (e.g. for the same building and
        # class, two instances of "ClassA/DS1" should be grouped in the same row)
        logger.debug(
            "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
            "grouping same damage states for same original_asset_id"
            % (np.datetime64('now'))
        )
        columns_by_original_asset_id_raw = [  # These values depend only on the original_asset_id
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
        columns_by_original_asset_id = []
        for col in columns_by_original_asset_id_raw:
            if col in new_exposure_model:  # not all columns might exist in exposure CSV
                columns_by_original_asset_id.append(col)
        new_exposure_model = new_exposure_model.drop(columns=columns_by_original_asset_id)
        # Sum number of buildings, people, costs for rows that need to be grouped
        new_exposure_model = new_exposure_model.groupby(
            ["original_asset_id", "taxonomy"]
        ).sum(numeric_only=True)
        # Re-assign values of columns that only depend on original_asset_id (retrieve from the
        # original exposure model)
        logger.debug(
            "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
            "re-assigning values of columns that only depend on original_asset_id"
            % (np.datetime64('now'))
        )

        original_asset_ids = new_exposure_model.index.get_level_values("original_asset_id")

        values_by_original_asset_id = (
            original_exposure_model
            .set_index("original_asset_id")[columns_by_original_asset_id]
            .loc[original_asset_ids]
        )

        for col in columns_by_original_asset_id:
            new_exposure_model[col] = values_by_original_asset_id[col].to_numpy()

        # Re-arrange index (up to now it is MultiIndex on (original_asset_id, taxonomy), need to
        # make it based again on ("asset_id", "dmg_state"), "asset_id" being that of the
        # previous exposure model; also need to recover contents of "taxonomy" as a column)
        logger.debug(
            "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
            "re-defining index"
            % (np.datetime64('now'))
        )
        new_exposure_model["taxonomy"] = new_exposure_model.index.get_level_values("taxonomy")

        # fragility name -> OpenQuake damage-state name (first match)
        fragility_to_dmg_state = pd.Series(
            mapping_damage_states.index,
            index=mapping_damage_states["fragility"].to_numpy(),
        )
        fragility_to_dmg_state = fragility_to_dmg_state[
            ~fragility_to_dmg_state.index.duplicated(keep="first")
        ]

        new_exposure_model["dmg_state"] = (
            new_exposure_model["taxonomy"].str.rpartition("/")[2].map(fragility_to_dmg_state)
        )

        if new_exposure_model["dmg_state"].isna().any():
            unknown = new_exposure_model["taxonomy"][new_exposure_model["dmg_state"].isna()].unique()
            error_message = (
                "Method 'ExposureUpdater.update_exposure_with_damage_states': taxonomies with "
                "damage states not in 'mapping_damage_states': %s" % (unknown,)
            )
            logger.critical(error_message)
            raise ValueError(error_message)

        original_asset_id = new_exposure_model.index.get_level_values("original_asset_id")
        new_index = pd.MultiIndex.from_arrays(
            [original_asset_id, new_exposure_model["dmg_state"]]
        )
        new_exposure_model.index = new_index
        new_exposure_model.index = new_exposure_model.index.rename(["asset_id", "dmg_state"])
        new_exposure_model = new_exposure_model.drop(columns=["dmg_state"])
        new_exposure_model["original_asset_id"] = original_asset_id

        # Order the DataFrame by asset_id and dmg_state (ascending order)
        logger.debug(
            "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
            "re-ordering by asset_id and dmg_state"
            % (np.datetime64('now'))
        )
        new_exposure_model = new_exposure_model.sort_values(
            by=[("asset_id"), ("dmg_state")], ascending=True
        )

        # Create new asset_id for the next calculation
        logger.debug(
            "%s Method 'ExposureUpdater.update_exposure_with_damage_states': "
            "creating asset_id for the next earthquake"
            % (np.datetime64('now'))
        )
        new_exposure_model["id"] = [
            "exp_%s" % (j) for j in range(1, new_exposure_model.shape[0] + 1)
        ]

        # Re-order columns
        col_names_for_order = list(original_exposure_model.columns)
        if earthquake_time_of_day not in col_names_for_order:
            col_names_for_order.append(earthquake_time_of_day)

        new_exposure_model = new_exposure_model.reindex(
            columns=["id", *col_names_for_order]
        )

        return new_exposure_model

    @staticmethod
    def update_exposure_occupants(
        calculate_casualties,
        exposure_full_occupants,
        recovery_occupancy_factors,
        time_of_day_factors,
        earthquake_time_of_day,
        earthquake_datetime,
        injured_still_away_oelf,
        main_path,
        injured_still_away_rla=None
    ):
        """
        If 'calculate_casualties' is True, this method calculates the number of occupants in
        each asset of 'exposure_full_occupants' at the time of 'earthquake_datetime',
        considering whether people are allowed to return to the buildings, as well as their
        health status. No action is carried out if 'calculate_casualties' is False.

        It assumes that files containing the number of injured people still not able to return
        to their buildings as well as files with factors that define whether people are allowed
        into the buildings or not as a function of their damage state are available under
        'main_path'/current/occupants for all earthquakes that have occurred before
        'earthquake_datetime'. It takes into account as well factors to multiply the number of
        occupants to represent number of people at different times of the day
        ('time_of_day_factors', 'earthquake_time_of_day').

        If no such files exist under 'main_path'/current/occupants, which is the case when no
        previous earthquakes have been run, the method still returns the occupants expected,
        which result simply from multiplying the census occupants by the corresponding
        'time_of_day_factors'.

        Args:
            exposure_full_occupants (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake, in
                which the occupants are all those allocated from distribution of the census
                population to the buildings, irrespective of the time of the day, the damage
                state of the building or the health status of its occupants. It comprises the
                following fields:
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
                            Total number of occupants in this asset irrespective of the time of
                            the day, the damage state of the building or the health status of
                            its occupants.
                        occupancy (str):
                            "Res" (residential), "Com" (commercial) or "Ind" (industrial).
                        id_X, name_X (str):
                            ID and name of the administrative units to which the asset belongs.
                            "X" is the administrative level.
            recovery_occupancy_factors  (dict of int: 0 or 1):
                Dictionary whose keys are the damage states, and whose contents are 1 or 0,
                indicating that by the date of the current earthquake, people will be allowed
                back in (1) or not (0) to buildings with the damage state (key). Example :
                {"DS0": 1, "DS1": 1, "DS2": 0, "DS3": 0, "DS4": 0}.
            time_of_day_factors (dict):
                Factors by which the census population per building can be multiplied to obtain
                an estimate of the people in the building at a certain time of the day. It
                should contain one key per occupancy case present in the exposure model (e.g.
                "residential", "commercial", "industrial"), and each key should be subdivided
                into:
                    - "day": approx. 10 am to 6 pm;
                    - "night": approx. 10 pm to 6 am;
                    - "transit": approx. 6 am to 10 am and 6 pm to 10 pm.
            earthquake_time_of_day (str):
                Time of the day of the earthquake, i.e. "day", "night" or "transit".
            earthquake_datetime (numpy.datetime64):
                UTC date and time of the earthquake.
            injured_still_away_oelf (dict of Pandas DataFrame):
                Injuries recovery timelines of the previous OELF earthquakes of the stochastic
                event set being run (keys: IDs of the earthquakes; elements: DataFrames as
                output by Losses.calculate_injuries_recovery_timeline()). Use None when running
                RLA, i.e. when there are no OELF earthquakes to consider.
            main_path (str):
                Path to the main running directory, assumed to have the needed structure.
            injured_still_away_rla (dict of Pandas DataFrame):
                Injuries recovery timelines of the RLA earthquakes, as output by
                Losses.read_injured_still_away_RLA(). If None (default), they are read from the
                CSV files stored under 'main_path'/current/occupants.

        Returns:
            exposure_updated_occupants (Pandas DataFrame):
                Pandas DataFrame containing all the fields and rows of the input
                'exposure_full_occupants', plus an additional column whose name is
                'earthquake_time_of_day' (i.e. "day", "night" or "transit") and whose content is
                the number of occupants in each asset of 'exposure_full_occupants' at the time
                of 'earthquake_datetime', considering whether people are allowed to return to
                the buildings and their health status.
        """

        if not calculate_casualties:
            return exposure_full_occupants

        exposure_updated_occupants = deepcopy(exposure_full_occupants)

        # Evaluate if all factors in occupancy_factor are zero (to avoid reading injuries if so)
        all_factors_null = True
        for dmg_state in recovery_occupancy_factors:
            if recovery_occupancy_factors[dmg_state] > 0.5:
                all_factors_null = False
                break  # once one factor is not zero, there is not need to keep on checking

        if not all_factors_null:
            logger.debug(
                "%s Method 'ExposureUpdater.update_exposure_occupants': "
                "there are some non-zero factors --> occupants are non-zero"
                % (np.datetime64('now'))
            )
            # Get occupancy factors (0 or 1) per asset of 'exposure_full_occupants'
            logger.debug(
                "%s Method 'ExposureUpdater.update_exposure_occupants': "
                "retrieving occupancy factors per asset"
                % (np.datetime64('now'))
            )
            occupancy_factors_per_asset = Losses.get_occupancy_factors_per_asset(
                exposure_updated_occupants["taxonomy"].to_numpy(), recovery_occupancy_factors
            )

            # Retrieve injuries (only for assets for which 'occupancy_factors_per_asset'=1)
            # (the method loops through the assets, hence looping only through necessary ones;
            # they will all be equal to zero if no earthquake has been run before)
            logger.debug(
                "%s Method 'ExposureUpdater.update_exposure_occupants': "
                "retrieving injuries"
                % (np.datetime64('now'))
            )
            original_asset_ids_unique = (
                exposure_updated_occupants["original_asset_id"].unique()
            )
            injured_still_away_vals = Losses.get_injured_still_away(
                original_asset_ids_unique,
                earthquake_datetime,
                main_path,
                injured_still_away_oelf=injured_still_away_oelf,
                injured_still_away_rla=injured_still_away_rla,
            )  # 'injured_still_away_vals' in the order of 'original_asset_ids_unique'
            injured_still_away = pd.DataFrame(
                {"number_injured": injured_still_away_vals},
                index=original_asset_ids_unique
            )
            injured_still_away.index.rename("original_asset_id")

            # Get time-of-day factors per asset
            logger.debug(
                "%s Method 'ExposureUpdater.update_exposure_occupants': "
                "getting time-of-day factors per asset"
                % (np.datetime64('now'))
            )
            time_of_day_factors_per_asset = Losses.get_time_of_day_factors_per_asset(
                exposure_updated_occupants["occupancy"].to_numpy(),
                earthquake_time_of_day,
                time_of_day_factors,
            )

            if injured_still_away["number_injured"].sum() > 1E-15:
                # Calculate number of injured still away per asset of 'exposure_updated_occupants',
                # done going one by one the 'original_asset_id' in the exposure model, and
                # distributing the number of injured still away for that 'original_asset_id' into
                # the different damage states (i.e. different rows of exposure associated with
                # 'original_asset_id') proportionally to the distribution of 'census' occupants
                logger.debug(
                    "%s Method 'ExposureUpdater.update_exposure_occupants': "
                    "calculating number of injured still away for each asset"
                    % (np.datetime64('now'))
                )

                census = exposure_updated_occupants["census"].to_numpy()

                # Total census occupants of the 'original_asset_id' of each row
                # (i.e. rows associated with the same 'original_asset_id' will have the same value)
                census_total_per_orig_asset_id = (
                    exposure_updated_occupants.groupby("original_asset_id")["census"]
                    .transform("sum")
                    .to_numpy()
                )

                # Number of injured still away of the 'original_asset_id' of each row
                injured_per_orig_asset_id = (
                    exposure_updated_occupants["original_asset_id"]
                    .map(injured_still_away["number_injured"])
                    .to_numpy()
                )

                # Distribute the injured still away of each 'original_asset_id' across its
                # rows (damage states) proportionally to the 'census' occupants
                injured_still_away_per_asset = injured_per_orig_asset_id * (
                    census / census_total_per_orig_asset_id
                )

                # Calculate the occupants at the time of the day of 'earthquake_time_of_day'
                logger.debug(
                    "%s Method 'ExposureUpdater.update_exposure_occupants': "
                    "calculating %s occupants for current earthquake"
                    % (np.datetime64('now'), earthquake_time_of_day)
                )
                occupants_at_time_of_day = (
                    time_of_day_factors_per_asset
                    * occupancy_factors_per_asset
                    * (census - injured_still_away_per_asset)
                )

            else: # ( injured_still_away["number_injured"].sum() <= 1E-15)
                logger.debug(
                    "%s Method 'ExposureUpdater.update_exposure_occupants': "
                    "no injured still away"
                    % (np.datetime64('now'))
                )
                occupants_at_time_of_day = (
                    time_of_day_factors_per_asset
                    * occupancy_factors_per_asset
                    * exposure_updated_occupants["census"].to_numpy()  # no injuries to subtract
                )
        else:
            # Do not retrieve injuries, occupants are zero for all assets
            logger.debug(
                "%s Method 'ExposureUpdater.update_exposure_occupants': "
                "all zero factors --> occupants are zero for all assets"
                % (np.datetime64('now'))
            )
            occupants_at_time_of_day = np.zeros([exposure_updated_occupants.shape[0]])

        exposure_updated_occupants[earthquake_time_of_day] = occupants_at_time_of_day

        return exposure_updated_occupants


    @staticmethod
    def summarise_damage_states_per_building_id(exposure):
        """
        This method returns the probability of a building with a certain building_id resulting
        in a certain damage state, when building_id refers to individual buildings, or the
        number of buildings with a certain building_id resulting in a certain damage state, when
        building_id refers to a group of buildings.

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
                    occupancy (str):
                        "Res" (residential), "Com" (commercial) or "Ind" (industrial).
        Returns:
            damage_summary (Pandas DataFrame):
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
        """

        # Initialise output
        damage_summary = exposure.drop(
            columns=["lon", "lat", "occupancy", "original_asset_id"]
        )
        if "id" in damage_summary.columns:
            damage_summary = damage_summary.drop(columns=["id"])

        # Create separate column for damage state
        building_classes = damage_summary["taxonomy"].to_numpy()
        damage_summary["damage_state"] = [
            building_classes[i].split("/")[-1] for i in range(damage_summary.shape[0])
        ]

        damage_summary = damage_summary.groupby(
            ["building_id", "damage_state"]
        ).sum(numeric_only=True)

        damage_summary = damage_summary[["number"]]

        return damage_summary


    @staticmethod
    def summarise_nuisance_states_per_building_id(
        nuisance_results_OQ, exposure, mapping_nuisance_states, occupants_column=None
    ):
        """
        This method returns the probability of a building with a certain building_id resulting
        in a certain level of nuisance, when building_id refers to individual buildings, or the
        number of buildings with a certain building_id resulting in a certain level of nuisance,
        when building_id refers to a group of buildings.

        Args:
            nuisance_results_OQ (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                nuisance state. This is output from running OpenQuake. It comprises the
                following fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'asset_id'.
                        loss_type (str):
                            "structural" (kept for format compatibility).
                        rlz (int):
                            0 (kept for format compatibility).
            exposure (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake. It
                comprises at least the following fields:
                    asset_id (str):
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
                    occupancy (str):
                        "Res" (residential), "Com" (commercial) or "Ind" (industrial).
                    census (float, optional):
                        Number of census occupants per asset.
            mapping_nuisance_states (Pandas DataFrame):
                Mapping between the names of nuisance states as output by OpenQuake (index) and
                as labelled in the nuisance model (value). E.g.:
                              fragility
                    dmg_state
                    no_damage       nonfelt
                    dmg_1           CDI2
                    dmg_2           CDI3
                    dmg_3           CDI4
                    dmg_4           CDI5
                    dmg_5           CDI6
            occupants_column (str):
                Name of the column with occupants. E.g., "day", "night", "transit". It can be
                None if no occupants column exists or is relevant.

        Returns:
            nuisance_summary (Pandas DataFrame):
                Pandas DataFrame with the following structure:
                    Index is multiple:
                        building_id (str):
                            ID of the building.
                        nuisance_level (str):
                            Nuisance level.
                    Columns:
                        number (float):
                            Probability of 'nuisance_level' for 'building_id', if building_id is
                            one individual building, or number of buildings of 'building_id'
                            under 'nuisance_level', if building_id is a group of buildings.
                        census (float, only if "census" exists in input 'exposure'):
                            Number of census people exposed to each 'nuisance_level' in each
                            'building_id'.
                        occupants (float, only if "occupants_column" is not None):
                            Number of occupants (at the time of the day of the earthquake)
                            exposed to each 'nuisance_level' in each 'building_id'.
        """

        # Replace dmg_state with fragility names
        df = nuisance_results_OQ.copy()
        df = df.reset_index()   # expose asset_id and dmg_state as columns

        df["fragility"] = df["dmg_state"].map(mapping_nuisance_states["fragility"])
        df.drop(columns=["dmg_state"], inplace=True)
        df.rename(columns={"fragility": "nuisance_level"}, inplace=True)

        # Columns to consider/aggregate
        cols_to_aggregate = {"value": "sum"}
        cols_to_merge = ["building_id"]

        if "census" in exposure.columns:
            cols_to_aggregate["census"] = "sum"
            cols_to_merge.append("census")

        if not (occupants_column is None):
            cols_to_aggregate[occupants_column] = "sum"
            cols_to_merge.append(occupants_column)

        # Attach building_id from exposure (exposure.asset_id == asset_id)
        df = df.merge(
            exposure[cols_to_merge],
            left_on="asset_id",
            right_on="asset_id",
            how="left"
        )

        if "census" in exposure.columns:
            sum_per_asset_id = df.groupby("asset_id")["value"].transform("sum")
            df["census"] = df["census"] * df["value"] / sum_per_asset_id

        if not (occupants_column is None):
            sum_per_asset_id = df.groupby("asset_id")["value"].transform("sum")
            df[occupants_column] = df[occupants_column] * df["value"] / sum_per_asset_id

        # Group by building_id and nuisance_level
        nuisance_summary = (
            df.groupby(["building_id", "nuisance_level"])
              .agg(cols_to_aggregate)
        )
        nuisance_summary.rename(columns={"value": "number"}, inplace=True)
        if not (occupants_column is None):
            nuisance_summary.rename(columns={occupants_column: "occupants"}, inplace=True)

        return nuisance_summary


    @staticmethod
    def get_unique_exposure_locations(exposure):
        """
        This method identifies and returns the longitude and latitude of the unique locations of
        'exposure'.

        Args:
            exposure (PandasDataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake. It
                comprises at least the following fields:
                    lon (float):
                        Longitude of the asset in degrees.
                    lat (float):
                        Latitude of the asset in degrees.

        Returns:
            unique_lons (float):
                Longitude of the unique locations.
            unique_lats (float):
                Latitude of the unique locations.
        """

        all_lons = exposure["lon"].to_numpy().astype(str)
        all_lats = exposure["lat"].to_numpy().astype(str)

        # Identify unique points by combining lon and lat as strings
        all_points = ["%s_%s" % (all_lons[i], all_lats[i]) for i in range(len(all_lons))]
        unique_points = np.unique(all_points)

        # Split unit into lons and lats and transform back to floats
        unique_lons = np.array([
            unique_points[i].split("_")[0] for i in range(len(unique_points))
        ]).astype(float)
        unique_lats = np.array([
            unique_points[i].split("_")[1] for i in range(len(unique_points))
        ]).astype(float)

        return unique_lons, unique_lats
