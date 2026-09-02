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


logger = logging.getLogger()


class OpenQuakeDataStoreHandler():
    """This class handles operations associated with OpenQuake's DataStore class.
    """

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
