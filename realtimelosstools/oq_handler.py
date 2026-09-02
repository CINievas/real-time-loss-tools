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
import numpy as np
import pandas as pd
from openquake.commands.run import main
from openquake.calculators.extract import extract
from openquake.commonlib.datastore import read
from realtimelosstools.exposure_updater import ExposureUpdater
from realtimelosstools.writers import Writer


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
        store_openquake,
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
            store_openquake (bool):
                If True, OpenQuake HDF5 files will be stored and jobs will be kept in
                OpenQuake's database. If false, OpenQuake's database will be purged of this
                job after running.
            path_to_oq_outputs (str):
                Full path where OpenQuake's results are to be stored as CSV files, if
                'store_intermediate' is True
            calc_id (int):
                Calculation ID assigned by OpenQuake when the job is run.
        """

        self.description = description
        self.earthquake_id = earthquake_id
        self.path_to_job_ini  = path_to_job_ini
        self.store_intermediate = store_intermediate
        self.store_openquake = store_openquake
        self.path_to_oq_outputs = path_to_oq_outputs
        self.calc_id = None

    def run(self, exposure_run, mapping_damage_states):
        """
        This method runs the OpenQuake job and retrieves its damage results. In detail:
        1) It runs the OpenQuake job.
        2) If the OpenQuake job runs without raising errors:
           - It retrieves damage results, accounting for logic tree weights.
           - It adjusts them to ensure there are no small negative values (due to float
             precision).
        3) It the OpenQuake job raises known errors, it outputs the same input damage states
        (the earthquake does not cause additional damage).
        4) It stores these damage results (in OpenQuake format) if 'self.store_intermediate'.
        5) It deletes OpenQuake's HDF5 for this job if 'self.store_openquake' is False.

        Args:
            exposure_run (Pandas DataFrame)
            mapping_damage_states (Pandas DataFrame):
                Mapping between the names of damage states as output by OpenQuake
                (index) and as labelled in the fragility model (value). E.g.:
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
                        loss_type (str):
                            "structural" (kept for format compatibility).
                        rlz (int):
                            0 (kept for format compatibility).
        """
        try:
            # Run OpenQuake
            self.calc_id = main(self.path_to_job_ini)

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
                    damage_results_OQ = ExposureUpdater.update_OQ_damage_w_logic_tree_weights(
                        damage_results_OQ, rlz_weights
                    )

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
                        % (self.description)
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

        if not self.store_openquake:  # Erase the job just run from OpenQuake's database (and HDF5)
            Writer.delete_OpenQuake_last_job()

        logger.info(
            "%s Running of OpenQuake and retrieval of results done" % (np.datetime64('now'))
        )

        # Store damage states from OpenQuake output to CSV (incl. potential negative values)
        if self.store_intermediate:
            damage_results_OQ.to_csv(
                os.path.join(
                    self.path_to_oq_outputs, "%s_damages_OQ_raw.csv" % (self.earthquake_id)
                ),
            )

        # Ensure no negative number of buildings in 'damage_results_OQ'
        logger.info(
            "%s Ensuring no negative number of buildings in 'damage_results_OQ'"
            % (np.datetime64('now'))
        )
        damage_results_OQ = ExposureUpdater.ensure_no_negative_damage_results_OQ(
            damage_results_OQ
        )

        # Store adjusted damage states from OpenQuake output to CSV
        if self.store_intermediate:
            damage_results_OQ.to_csv(
                os.path.join(
                    self.path_to_oq_outputs, "%s_damages_OQ.csv" % (self.earthquake_id)
                ),
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

        rows = []

        for _, r in datastore_damage_results.iterrows():
            # each row of the DataFrame = 1 exposure asset
            record = r["value"]  # this is the np.void with number of buildings per damage state
            for i, field in enumerate(record.dtype.names):  # names of the damage states
                losstype = field.split("-")[0]
                dmg_state = field.split("-")[1]
                if dmg_state != "no_damage":
                    dmg_state = "dmg_%s" % (i)  # "dmg_X" as used in OpenQuake v3.15
                rows.append({
                    "asset_id": oq_asset_id_mapping[r["asset_id"]],
                    "dmg_state": dmg_state,
                    "rlz": r["rlz_id"],
                    "loss_type": losstype,
                    "value": record[field]
                })

        damage_results_OQ = pd.DataFrame(rows)

        return damage_results_OQ
