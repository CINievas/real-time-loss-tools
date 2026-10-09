#!/usr/bin/env python3

# Copyright (C) 2026:
#   Cecilia Nievas: cecilia.nievas@gfz.de
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
import getpass
import configparser
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from copy import deepcopy
from openquake.baselib import parallel
from openquake.commands.run import main
from openquake.commands.purge import purge_one
from openquake.calculators.extract import extract
from openquake.commonlib.datastore import read


logger = logging.getLogger()


class OpenQuakeJob():
    """
    This class handles the running of a job in OpenQuake.
    """

    def __init__(
        self,
        description,
        earthquake_id,
        path_to_job_ini,
        store_intermediate,
        output_type,
        oq_previous_hazard_id=None,
        path_to_oq_outputs=None
    ):
        """
        Attrs:
            description (str):
                Description of the OpenQuake job.
            earthquake_id (str):
                ID of the earthquake (as input by the user or within the catalogue).
            path_to_job_ini (str):
                Full path (directory and filename) to OpenQuake's job.ini file,
                which needs to exist.
            store_intermediate (bool):
                If True, the damage results as directly output by OpenQuake, as well as their
                adjusted version (so that they do not include negative numbers of buildings)
                are stored under 'path_to_oq_outputs'.
            output_type (str):
                Type of output of the calculation. Currently supported values are "damage" and
                "nuisance".
            oq_previous_hazard_id (int):
                ID of an OpenQuake calculation previously run from which hazards results will
                be retrieved for the new calculation.
            path_to_oq_outputs (str):
                Full path where OpenQuake's results are to be stored as CSV files, if
                'store_intermediate' is True.
            calc_id (int):
                Calculation ID assigned by OpenQuake when the job is run.
            oq_ran (bool):
                True if the OpenQuake job ran without crashing.
        """

        self.description = description
        self.earthquake_id = earthquake_id
        self.path_to_job_ini  = path_to_job_ini
        self.store_intermediate = store_intermediate
        if output_type.lower() == "damage" or output_type.lower() == "nuisance":
            self.output_type = output_type.lower()
        else:
            error_message = (
                "Error initialising OpenQuakeJob class: 'output_type' can only be 'damage' or "
                "'nuisance'. Current value: %s." % (output_type)
            )
            raise ValueError(error_message)
        self.oq_previous_hazard_id = oq_previous_hazard_id
        self.path_to_oq_outputs = path_to_oq_outputs
        self.calc_id = None
        self.oq_ran = False

    def run(self, exposure_run, mapping_states):
        """
        This method runs the OpenQuake job and retrieves its damage results. In detail:
        1) It runs the OpenQuake job.
        2) If the OpenQuake job runs without raising errors:
           - It retrieves damage/nuisance results, accounting for logic tree weights.
           - It adjusts them to ensure there are no small negative values (due to float
             precision).
        3) It the OpenQuake job raises known errors, it outputs the same input damage states
        (the earthquake does not cause additional damage) and zero nuisance.
        4) It stores these damage/nuisance results (in OpenQuake format) if
        'self.store_intermediate' is True.

        Args:
            exposure_run (Pandas DataFrame):
                Exposure model used to run OpenQuake.
            mapping_states (Pandas DataFrame):
                Mapping between the names of damage states as output by OpenQuake
                (index) and the damage states or nuisance levels as labelled in the fragility/
                nuisance model (value). E.g.:
                              fragility
                    asset_id
                    no_damage       DS0
                    dmg_1           DS1
                    dmg_2           DS2
                    dmg_3           DS3
                    dmg_4           DS4

        Returns:
            damage_results_OQ (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state/nuisance level. This is output from running OpenQuake. It comprises
                the following fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                        dmg_state (str):
                            Damage states or nuisance levels.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'asset_id'.
                        loss_type (str):
                            "structural" (kept for format compatibility).
                        rlz (int):
                            0 (kept for format compatibility).
        """
        try:
            # Run OpenQuake
            self.calc_id = main(self.path_to_job_ini, hc=self.oq_previous_hazard_id)
            self.oq_ran = True

            # Retrieve damage states from OpenQuake output
            dstore = read(self.calc_id)

            try:
                damage_results_OQ_raw325 = dstore.read_df("damages-rlzs")

                oq_asset_id_mapping = self.retrieve_exposure_asset_id_mapping(
                    extract(dstore, "assetcol").array
                )

                damage_results_OQ = self.backward_compatibility_damage(
                    damage_results_OQ_raw325, oq_asset_id_mapping
                )

                # Check if different realisations exist
                # (i.e., from different GMPE logic tree branches)
                if len(damage_results_OQ["rlz"].unique()) != 1:
                    rlz_weights_raw = extract(dstore, "full_lt").get_realizations()
                    rlz_weights = {}
                    for i, rlz in enumerate(rlz_weights_raw):
                        rlz_weights[i] = rlz.weight["weight"]
                    # Incorporate logic tree weights to damage results
                    damage_results_OQ = self.update_OQ_damage_w_logic_tree_weights(
                        damage_results_OQ, rlz_weights
                    )

            except KeyError as ke:
                if (len(ke.args) == 1) and ("damages-rlzs" in ke.args):
                    # OpenQuake's "There is no damage, perhaps the hazard is too small?"
                    damage_results_OQ = self._no_consequences(
                        exposure_run, mapping_states
                    )
                else:
                    error_message = (
                        "OpenQuake has not found 'damages-rlzs' for %s "
                        "and Real-Time Loss Tools has not been able to solve it"
                        % (self.description)
                    )
                    logger.critical(error_message)
                    raise OSError(error_message)

            dstore.close()

        except RuntimeError as run_e:
            if (len(run_e.args) == 1) and ("No GMFs were generated" in run_e.args[0]):
                # OpenQuake's "No GMFs were generated, perhaps they were all below
                # the minimum_intensity threshold"
                damage_results_OQ = self._no_consequences(exposure_run, mapping_states)
            else:
                error_message = (
                    "OpenQuake has crashed for %s "
                    "and Real-Time Loss Tools has not been able to solve it"
                    % (self.description)
                )
                logger.critical(error_message)
                raise OSError(error_message)

        except AssertionError as ae:
                error_message = (
                    "OpenQuake has crashed for %s with an AssertionError "
                    "and Real-Time Loss Tools has not been able to solve it"
                    % (self.description)
                )
                logger.critical(error_message)
                raise OSError(error_message)

        if "asset_id" in damage_results_OQ.columns and "dmg_state" in damage_results_OQ.columns:
            # If damage_results_OQ has been adjusted as per GMPE logic tree weights,
            # this step is not necessary
            new_index = pd.MultiIndex.from_arrays(
                [damage_results_OQ["asset_id"], damage_results_OQ["dmg_state"]]
            )
            damage_results_OQ.index = new_index
            damage_results_OQ = damage_results_OQ.drop(columns=["asset_id", "dmg_state"])

        logger.info(
            "%s Running of OpenQuake and retrieval of results done" % (np.datetime64('now'))
        )

        # Store damage states from OpenQuake output to CSV (incl. potential negative values)
        if self.store_intermediate:
            damage_results_OQ.to_csv(
                os.path.join(
                    self.path_to_oq_outputs,
                    "%s_%s_OQ_raw.csv" % (self.earthquake_id, self.output_type)
                ),
            )

        # Ensure no negative number of buildings in 'damage_results_OQ'
        logger.info(
            "%s Ensuring no negative number of buildings in 'damage_results_OQ'"
            % (np.datetime64('now'))
        )
        damage_results_OQ = self.ensure_no_negative_damage_results_OQ(
            damage_results_OQ
        )

        # Store adjusted damage states from OpenQuake output to CSV
        if self.store_intermediate:
            damage_results_OQ.to_csv(
                os.path.join(
                    self.path_to_oq_outputs,
                    "%s_%s_OQ.csv" % (self.earthquake_id, self.output_type)
                ),
            )

        return damage_results_OQ


    def delete_OpenQuake_job(self):
        """
        This method removes this calculation's ID from OpenQuake's database and its associated
        HDF5 file.
        """

        if not (self.calc_id is None):
            purge_one(self.calc_id, getpass.getuser(), True)

        return


    def _no_consequences(self, exposure_run, mapping_states):
        """
        This method simulates OpenQuake's damage output assuming the earthquake does not cause
        any nuisance or any further damage. The kind of output (damage or nuisance) depends on
        self.output_type.

        Args:
            exposure_run (Pandas DataFrame):
                Exposure model used to run OpenQuake.
            mapping_states (Pandas DataFrame):
                Mapping between the names of damage states as output by OpenQuake
                (index) and the damage states or nuisance levels as labelled in the fragility/
                nuisance model (value). E.g.:
                              fragility
                    asset_id
                    no_damage       DS0
                    dmg_1           DS1
                    dmg_2           DS2
                    dmg_3           DS3
                    dmg_4           DS4

        Returns:
            damage_results_OQ (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state/nuisance level. This is output from running OpenQuake. It comprises
                the following fields:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                        dmg_state (str):
                            Damage states or nuisance levels.
                    Columns:
                        value (float):
                            Probability of 'dmg_state' for 'asset_id'.
                        loss_type (str):
                            "structural" (kept for format compatibility).
                        rlz (int):
                            0 (kept for format compatibility).
        """

        damage_results_OQ = None

        if self.output_type == "damage":
            damage_results_OQ = OpenQuakeUtils.create_OQ_existing_damage(
                    exposure_run,
                    mapping_states,
                    loss_type="structural"
            )
        elif self.output_type == "nuisance":
            damage_results_OQ = OpenQuakeUtils.create_OQ_no_nuisance(
                    exposure_run,
                    mapping_states,
                    loss_type="structural"
            )

        return damage_results_OQ


    @staticmethod
    def retrieve_exposure_asset_id_mapping(oq_exposure):
        """
        This method retrieves the mapping between exposure asset_ids input by the user
        ("id" field of 'oq_exposure') and the id used internally by OpenQuake for them
        ("ordinal" field of 'oq_exposure'), and outputs it as a dictionary in which the keys are
        the ordinals and the values are the user-defined asset_ids.

        Args:
            oq_exposure (numpy structured array):
                Structured array containing the input exposure model as stored by OpenQuake. The
                "id" field matches the asset_ids input by the user in the exposure CSV file,
                while the "ordinal" field contains the id used by OpenQuake in its DataStore.

        Returns:
            mapping (dict):
                Dictionary mapping the contents of "ordinal" (key) to "id" (value). E.g.:
                {ordinal_0: id_0, ordinal_1:, id_1, ...}.
        """

        mapping = {}
        for row in oq_exposure:
            ordinal = int(row["ordinal"])
            identifier = row["id"].decode()
            mapping[ordinal] = identifier

        return mapping


    @staticmethod
    def backward_compatibility_damage(datastore_damage_results, oq_asset_id_mapping):
        """
        This method adapts the damage results output from OpenQuake v3.25 to the format of
        OpenQuake v3.15.

        Args:
            datastore_damage_results (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state as output by OpenQuake's DataStore class, with the following
                    structure:
                    Index: consecutive integers starting with 0.
                    Columns:
                        asset_id (str): Asset IDs from 'exposure'.
                        rlz_id (int): All zeroes.
                        value (numpy.void): Number of buildings of 'asset_id' in each damage
                            state. Example structure:
                                np.void(
                                    (value_DS0, value_DS1, value_DS2, value_DS3, value_DS4),
                                    dtype=[
                                        ('name_DS0', '<f4'),
                                        ('name_DS1', '<f4'),
                                        ('name_DS2', '<f4'),
                                        ('name_DS3', '<f4'),
                                        ('name_DS4', '<f4')
                                    ]
                                )
                                Where "value_DSX" is the number of buildings in damage state
            oq_asset_id_mapping (dict):
                Dictionary mapping the exposure asset IDs used internally by OpenQuake (keys)
                and the user-defined asset_ids. E.g.: {0: "user_name_1", 1: "user_name_2", ...}.

        Returns:
            damage_results_OQ (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state and the following structure:
                    Index: consecutive integers starting with 0.
                    Columns:
                        asset_id (str): Asset IDs.
                        dmg_state (str): Damage states as "dmg_X" (OpenQuake v3.15).
                        rlz (int): All zeroes.
                        loss_type (str): Type of OpenQuake loss (not used).
                        value (float): Number of buildings of 'asset_id' in each 'dmg_state'.
        """

        if datastore_damage_results.shape[0] == 0:  # nothing to convert
            return pd.DataFrame([])

        n_assets = datastore_damage_results.shape[0]

        # Each element of the "value" column is a np.void with the number of buildings in each
        # damage state (all of them have the same structure), which is turned into one
        # structured array and then into a 2D array with one row per asset and one column per
        # damage state
        record_dtype = datastore_damage_results["value"].iloc[0].dtype
        structured_values = np.array(
            datastore_damage_results["value"].tolist(), dtype=record_dtype
        )
        values_2d = np.column_stack(
            [structured_values[field] for field in record_dtype.names]
        )  # shape (n_assets, n_damage_states)
        n_fields = len(record_dtype.names)

        # Loss type and damage state, which only depend on the names of the fields
        loss_types = [field.split("-")[0] for field in record_dtype.names]
        dmg_states = []
        for i, field in enumerate(record_dtype.names):  # names of the damage states
            dmg_state = field.split("-")[1]
            if dmg_state != "no_damage":
                dmg_state = "dmg_%s" % (i)  # "dmg_X" as used in OpenQuake v3.15
            dmg_states.append(dmg_state)

        # User-defined asset IDs (KeyError if an asset ID of OpenQuake is not in the mapping)
        asset_ids = np.array(
            [oq_asset_id_mapping[a] for a in datastore_damage_results["asset_id"]],
            dtype=object
        )

        # Rows are ordered by asset and, within each asset, by damage state
        damage_results_OQ = pd.DataFrame(
            {
                "asset_id": np.repeat(asset_ids, n_fields),
                "dmg_state": np.tile(np.array(dmg_states, dtype=object), n_assets),
                "rlz": np.repeat(datastore_damage_results["rlz_id"].to_numpy(), n_fields),
                "loss_type": np.tile(np.array(loss_types, dtype=object), n_assets),
                "value": values_2d.reshape(-1),
            }
        )

        return damage_results_OQ


    @staticmethod
    def update_OQ_damage_w_logic_tree_weights(damage_results_OQ, logic_tree_weights):
        """
        This method processes 'damage_results_OQ' so that the output
        ('damage_results_OQ_weighted') takes into account the weights of the GMPE logic tree
        branches as indicated in 'logic_tree_weights'.

        When the GMPE logic tree has more than one branch, the damage results by asset_id
        retrieved from OpenQuake are full results for each logic tree branch (and the total sum
        of "value" adds up to many more buildings than in the exposure model). This method
        multiplies these results by the weight of the corresponding logic tree branch and
        adds up all values for the same 'asset_id' and 'dmg_state' (and different 'rlz').

        Args:
            damage_results_OQ (Pandas DataFrame):
                Pandas DataFrame with numbers of buildings/probabilities of buildings in each
                damage state. This is output from running OpenQuake. It comprises the following
                fields:
                    Index: consecutive integers starting with 0.
                    Columns:
                        asset_id (str): Asset IDs from 'exposure'.
                        rlz (int): All zeroes.
                        loss_type (str): All as per 'loss_type'.
                        dmg_state (str): Damage states as per 'mapping_damage_states' and
                            'exposure'.
                        value (float): Number of buildings of 'asset_id' in each
                            'dmg_state', as per 'exposure'.
            logic_tree_weights (dict):
                Dictionary whose keys are the realisation IDs (i.e., column 'rlz' of
                'damage_results_OQ') and whose values are the weights of each realisation (i.e.,
                each branch of the GMPE logic tree).

        Returns:
            damage_results_OQ_weighted (Pandas DataFrame):
                The contents are the same as 'damage_results_OQ' in the input, but adjusted so
                that "value" (i.e., the probability of or number of buildings in 'dmg_state' for
                'asset_id') takes into account the weights of the GMPE logic tree branches as
                indicated in 'logic_tree_weights', and the following structure:
                    Index is multiple:
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                        dmg_state (str):
                            Damage states.
                    Columns:
                        value (float):
                            Probability of or number of buildings in 'dmg_state' for 'asset_id'.
                        rlz (int):
                            OpenQuake realisation ID. As the RTLT run only with scenario damage
                            calculations, this are the IDs of the GMPE logic tree branches.
                        (Column "loss_type", which is part of OpenQuake's output, is not used).
        """

        damage_results_OQ_weighted = deepcopy(damage_results_OQ)
        assigned_weights = np.array(
            [
                logic_tree_weights[damage_results_OQ["rlz"].to_numpy()[i]]
                for i in range(damage_results_OQ.shape[0])
            ]
        )
        damage_results_OQ_weighted["weighted_value"] = (
            assigned_weights * damage_results_OQ_weighted["value"]
        )

        # Add weighted numbers of buildings
        damage_results_OQ_weighted = damage_results_OQ_weighted.groupby(
            ["asset_id", "dmg_state"]
        ).agg(
            {
                "rlz": "first",  # keep first value of rlz, i.e. 0
                "loss_type": "first",  # keep first value of 'loss_type'
                "weighted_value": "sum"
            }
        )

        damage_results_OQ_weighted = damage_results_OQ_weighted.rename(
            columns={"weighted_value": "value"}
        )

        return damage_results_OQ_weighted


    @staticmethod
    def ensure_no_negative_damage_results_OQ(
        damage_results_OQ, tolerance=0.0001, n_rows_simplified=100000
    ):
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
            n_rows_simplified (int):
                If the number of rows of 'damage_results_OQ' is smaller than 'n_rows_simplified',
                the adjustment of the number of buildings is carried out by 'asset_id'. If it is
                larger, the adjustment is done in a simplified way, taking 'damage_results_OQ'
                as a whole.

        Returns:
            damage_results_OQ_adjusted (Pandas DataFrame):
                Same structure as 'damage_results_OQ' in the input, but adjusted so that there
                are no negative values in the column "value".
        """

        if np.all(damage_results_OQ.loc[:, "value"] >= 0):  # Nothing to be done
            return damage_results_OQ

        damage_results_OQ_adjusted = deepcopy(damage_results_OQ)

        filter_neg_vals = (damage_results_OQ_adjusted.value < 0.0)

        if filter_neg_vals.sum() == 0:  # there are no negative values
            # nothing to be done
            return damage_results_OQ_adjusted

        if damage_results_OQ_adjusted.shape[0] > n_rows_simplified:
            # Simplified procedure (adjusting the whole DataFrame all together)
            total_bdgs = damage_results_OQ_adjusted["value"].sum()
            negative_bdgs = abs(damage_results_OQ_adjusted[filter_neg_vals]["value"].sum())

            if negative_bdgs / total_bdgs > tolerance:
                error_message = (
                    "There total of negative values in the damage results from OpenQuake "
                    "exceeds the %s tolerance. The program cannot continue running"
                    % (tolerance)
                )
                logger.critical(error_message)
                raise ValueError(error_message)

            # Set negative numbers to zero
            damage_results_OQ_adjusted.loc[filter_neg_vals, "value"] = 0

            # Recalculate the other values so as to keep the total number of buildings
            damage_results_OQ_adjusted["value"] = (
                damage_results_OQ_adjusted["value"] / damage_results_OQ_adjusted["value"].sum()
                * total_bdgs
            )

            return damage_results_OQ_adjusted

        # If there are negative values and damage_results_OQ_adjusted.shape[0]
        # <= n_rows_simplified, adjust the number of buildings by 'asset_id'
        asset_ids_neg_vals = damage_results_OQ_adjusted[filter_neg_vals].index.get_level_values(
            "asset_id"
        ).unique()

        for i, asset_id in enumerate(asset_ids_neg_vals):
            damage_results_OQ_asset = damage_results_OQ_adjusted.loc[asset_id, "value"]

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
            damage_results_OQ_asset = damage_results_OQ_asset.clip(0.0, np.inf)

            # Recalculate the other values so as to keep the total number of buildings
            damage_results_OQ_asset = (
                damage_results_OQ_asset / damage_results_OQ_asset.sum() * total_bdgs
            )

            # Transfer back to the original DataFrame (damage_results_OQ)
            damage_results_OQ_adjusted.loc[asset_id, "value"] = (
                damage_results_OQ_asset.to_numpy()
            )

        return damage_results_OQ_adjusted


class OpenQuakeFragility():
    """
    This class represents a fragility model read from an XML in OpenQuake format.

    Attrs:
        path (str): Path to the input XML file of the fragility model.
        ims (list of str): List of unique intensity measure types used by the fragility model.
    """
    def __init__(self):
        self.path = None
        self.ims = None

    def from_xml(self, xml_path):
        self.path = xml_path
        self.ims = self.read_IMs()

    def from_ini(self, main_path, job_ini_path):
        """
        This method retrieves the name of the fragility XML file from OpenQuake's job.ini (read
        from 'job_ini_path') and assumes it is located under 'main_path'/static.
        """
        jobini = configparser.ConfigParser(interpolation=None)
        jobini.read(job_ini_path)

        frag_relative = jobini["fragility"]["structural_fragility_file"]

        self.path = os.path.join(main_path, "static", frag_relative.split("/")[-1])
        self.ims = self.read_IMs()

    def read_IMs(self):
        """
        This method retrieves the unique intensity measure types used by the fragility model.
        """

        def local(tag):
            # Strip namespace: "{uri}fragilityFunction" → "fragilityFunction"
            if "}" in tag:
                return tag.split("}", 1)[1]
            return tag

        unique_imts = set()

        for event, elem in ET.iterparse(self.path, events=("start",)):
            if local(elem.tag) == "imls":      # namespace-safe
                imt = elem.get("imt")
                if imt:
                    unique_imts.add(imt)

        return list(unique_imts)


class OpenQuakeUtils():
    """
    Class with methods associated with running OpenQuake.
    """

    @staticmethod
    def set_number_of_cores(number_oq_cores):
        """
        This method tells OpenQuake how many cores it can use to run its jobs. It needs to be
        called once, before running any OpenQuake job (OpenQuake reads the settings that this
        method modifies every time it starts a job).

        - "all": nothing is changed, i.e. OpenQuake uses its own default, which is the number
        of cores that it can use (the smaller of the number of cores visible to the process and
        the memory in GB) unless 'OQ_NUM_CORES' or the OpenQuake configuration file say
        otherwise.
        - 1: OpenQuake is told not to use its pool of worker processes, i.e. the environment
        variable 'OQ_DISTRIBUTE' is set to "no" for this program and all its jobs run in the
        main process (this overrides the value that 'OQ_DISTRIBUTE' may have in the
        environment).
        - An integer larger than 1 and not larger than the number of cores that OpenQuake can
        use: OpenQuake's pool of worker processes is limited to this number of workers.
        - An integer larger than the number of cores that OpenQuake can use: it is treated as
        "all" (nothing is changed) and a warning is logged.

        Args:
            number_oq_cores (str or int):
                "all" or an integer >= 1.

        Returns:
            message (str):
                Short description of what OpenQuake will use, which is also logged.
        """

        if number_oq_cores == "all":
            message = "OpenQuake uses its default number of cores (all the cores it can use)"

        elif number_oq_cores == 1:
            os.environ["OQ_DISTRIBUTE"] = "no"
            message = "OpenQuake runs its jobs without worker processes (1 core)"

        else:
            if not (hasattr(parallel, "num_cores") and hasattr(parallel, "tot_cores")):
                error_message = (
                    "The installed OpenQuake does not have the variables "
                    "'openquake.baselib.parallel.num_cores' and 'tot_cores', which are needed "
                    "to limit the number of cores of OpenQuake (tested with OpenQuake 3.26)."
                )
                logger.critical(error_message)
                raise OSError(error_message)

            if number_oq_cores > parallel.tot_cores:
                message = (
                    "WARNING: 'number_oq_cores' (%s) is larger than the number of cores that "
                    "OpenQuake can use (%s): OpenQuake uses its default number of cores "
                    "(all the cores it can use)" % (number_oq_cores, parallel.tot_cores)
                )
                logger.warning(message)
            else:
                parallel.num_cores = int(number_oq_cores)
                message = "OpenQuake uses up to %s cores" % (number_oq_cores)

        logger.info(message)

        return message

    @staticmethod
    def can_previous_hazard_be_reused(
        filepath_nuisance_xml,
        filepath_damage_job_ini,
        main_path,
        previousjobran,
    ):
        """
        This method assesses whether hazard outputs from a previous OpenQuake damage calculation
        can be re-used for an upcoming OpenQuake nuisance calculation. The output
        ('previous_hazard_id') can be passed on directly to OpenQuake commands such as
        openquake.commands.run.main(..., hc=previous_hazard_id): it is -1 if the calculation can
        be re-used, and None if it cannot.

        Previously-calculated hazard results can be re-used if:
        - the intensity measures needed by the nuisance model are part of the set of intensity
        measures needed by the previously-ran damage calculation, and
        - the previous damage calculation has, in fact, run (i.e., OpenQuake did not crash).

        Args:
            filepath_nuisance_xml (str):
                Complete path (dir+filename) to the XML file that contains the nuisance model.
            filepath_damage_job_ini (str):
                Complete path (dir+filename) to the job.ini file used to run the previous damage
                job in OpenQuake.
            main_path (str):
                RTLTs' "main_path" used to  run the previous damage job in OpenQuake.
            previousjobran (bool):
                True if the previous damage calculation actually ran, False otherwise (e.g. if
                OpenQuake crashed).

        Returns:
            previous_hazard_id (int)
                - -1 if the previously-calculated hazard can be re-used,
                - None if the previously-calculated hazard cannot be re-used.
        """

        # Which IMs does the nuisance calculation need?
        nuisance_frag = OpenQuakeFragility()
        nuisance_frag.from_xml(filepath_nuisance_xml)

        # Retrieve IMs for which ground motion fields exist already
        if previousjobran:
            damage_frag = OpenQuakeFragility()
            damage_frag.from_ini(main_path, filepath_damage_job_ini)
            existing_ims = damage_frag.ims
        else:
            # The damage calculation did not run successfully
            existing_ims = []

        # Check if nuisance_frag.ims are in existing_ims
        # to determine whether to run hazard again or re-use from damage calculation
        if set(nuisance_frag.ims).issubset(existing_ims):
            previous_hazard_id = -1
        else:
            previous_hazard_id = None

        return previous_hazard_id


    @staticmethod
    def create_OQ_existing_damage(exposure, mapping_damage_states, loss_type="structural"):
        """
        This method simulates the output of OpenQuake's DataStore.read_df() assigning no damage
        (apart from the pre-existing damage) to all asset IDs in 'exposure'.

        OpenQuake's DataStore.read_df() raises a KeyError when asked to retrieve the damage
        results of a calculation that has resulted in no damage (for which OpenQuake logs the
        warning "There is no damage, perhaps the hazard is too small?").

        Args:
            exposure (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake whose
                damage results are to be simulated. It must comprise at least the following
                fields:
                    Index (simple):
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                    Columns:
                        taxonomy (str):
                            Building class.
                        number (float):
                            Number of buildings in this asset.
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
            loss_type (str):
                Type of loss to be output by OpenQuake. Default: "structural".

        Returns:
            damage_results_no_damage (DataFrame):
                DataFrame with the following structure:
                    Index: consecutive integers starting with 0.
                    Columns:
                        asset_id (str): Asset IDs from 'exposure'.
                        rlz (int): All zeroes.
                        loss_type (str): All as per 'loss_type'.
                        dmg_state (str): Damage states as per 'mapping_damage_states' and
                            'exposure'.
                        value (float): Number of buildings of 'asset_id' in each
                            'dmg_state', as per 'exposure'.
        """

        asset_ids = exposure.index.to_numpy()
        buildings_per_asset_id = exposure["number"].to_numpy()

        # Extract damage state from taxonomy
        dmg_states_exposure = exposure["taxonomy"].str.split("/").str[-1].to_numpy()

        dmg_states_oq = mapping_damage_states.index.to_numpy()
        fragility_codes = mapping_damage_states["fragility"].to_numpy()

        n_assets = len(asset_ids)
        n_states = len(dmg_states_oq)

        # Repeat each asset_id the number of nuisance states we have (n_states)
        # (creates blocks of "asset_id_1", ..., "asset_id_1", "asset_id_2", ..., "asset_id_2")
        asset_ids_all = np.repeat(asset_ids, n_states)
        # Repeat each block of nuisance states to match each block of assets
        # (creates blocks of "no_dmg", "dmg_1", ..., "dmg_X", "no_dmg", "dmg_1", ..., "dmg_X")
        dmg_states_oq_all = np.tile(dmg_states_oq, n_assets)

        # Build array with the same damage states as in "dmg_states_oq_all"
        # but in terms of mapping_damage_states["fragility"]
        fragility_all = np.tile(fragility_codes, n_assets)

        # Build exposure damage state array aligned with dmg_states_oq_all
        exposure_ds_all = np.repeat(dmg_states_exposure, n_states)

        # Assign buildings only where fragility dmg state matches exposure dmg state
        values_all = np.zeros(n_assets * n_states, dtype=float)

        mask = (fragility_all == exposure_ds_all)
        values_all[mask] = np.repeat(buildings_per_asset_id, n_states)[mask]

        # Build DataFrame
        damage_results_no_damage = pd.DataFrame(
            {
                "asset_id": asset_ids_all,
                "rlz": 0,
                "loss_type": loss_type,
                "dmg_state": dmg_states_oq_all,
                "value": values_all,
            }
        )

        return damage_results_no_damage


    @staticmethod
    def create_OQ_no_nuisance(exposure, mapping_nuisance_states, loss_type="structural"):
        """
        This method simulates the output of OpenQuake's DataStore.read_df() assigning no
        nuisance to all asset IDs in 'exposure'.

        Args:
            exposure (Pandas DataFrame):
                Pandas DataFrame representation of the exposure CSV input for OpenQuake whose
                damage results are to be simulated. It must comprise at least the following
                fields:
                    Index (simple):
                        asset_id (str):
                            ID of the asset (i.e. specific combination of building_id and a
                            particular building class).
                    Columns:
                        taxonomy (str):
                            Building class.
                        number (float):
                            Number of buildings in this asset.
            mapping_nuisance_states (Pandas DataFrame):
                Mapping between the names of nuisance states as output by OpenQuake (index) and
                as labelled in the nuisance model (value), assumed to be ordered from least
                extreme to most extreme. E.g.:
                              fragility
                    dmg_state
                    no_damage       nonfelt
                    dmg_1           CDI2
                    dmg_2           CDI3
                    dmg_3           CDI4
                    dmg_4           CDI5
                    dmg_5           CDI6
            loss_type (str):
                Type of loss to be output by OpenQuake. Default: "structural".

        Returns:
            damage_results_no_nuisance (DataFrame):
                DataFrame with the following structure:
                    Index: consecutive integers starting with 0.
                    Columns:
                        asset_id (str): Asset IDs from 'exposure'.
                        rlz (int): All zeroes.
                        loss_type (str): All as per 'loss_type'.
                        dmg_state (str): Nuisance levels as per the index of
                            'mapping_nuisance_states'.
                        value (float): Number of buildings of 'asset_id' in each
                            'dmg_state', as per 'exposure'.
        """

        asset_ids = exposure.index.to_numpy()
        buildings_per_asset_id = exposure["number"].to_numpy()
        dmg_states_oq = mapping_nuisance_states.index.to_numpy()

        n_assets = len(asset_ids)
        n_states = len(dmg_states_oq)

        # Repeat each asset_id the number of nuisance states we have (n_states)
        # (creates blocks of "asset_id_1", ..., "asset_id_1", "asset_id_2", ..., "asset_id_2")
        asset_ids_all = np.repeat(asset_ids,  (n_states))
        # Repeat each block of nuisance states to match each block of assets
        # (creates blocks of "no_dmg", "dmg_1", ..., "dmg_X", "no_dmg", "dmg_1", ..., "dmg_X")
        dmg_states_oq_all = np.tile(dmg_states_oq, n_assets)

        # All buildings go to the "no nuisance" level, all other levels are zero
        values_all = np.zeros(n_assets * n_states, dtype=float)

        mask_first_state = (dmg_states_oq_all == dmg_states_oq[0])
        values_all[mask_first_state] = np.repeat(buildings_per_asset_id, 1)

        # Build DataFrame
        damage_results_no_nuisance = pd.DataFrame(
            {
                "asset_id": asset_ids_all,
                "rlz": 0,
                "loss_type": loss_type,
                "dmg_state": dmg_states_oq_all,
                "value": values_all,
            }
        )

        return damage_results_no_nuisance
