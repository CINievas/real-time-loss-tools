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
import pandas as pd
from realtimelosstools.losses import EconomicLosses


def test_expected_economic_loss():
    # Read exposure model
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "expected_exposure_model_cycle_2.csv"
    )
    exposure = pd.read_csv(filepath)

    # Read economic consequence model
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "consequences_economic.csv"
    )
    consequence_model = pd.read_csv(filepath)
    consequence_model.set_index(
        consequence_model["Taxonomy"], drop=True, inplace=True
    )
    consequence_model = consequence_model.drop(columns=["Taxonomy"])

    returned_loss_summary = EconomicLosses.expected_economic_loss(exposure, consequence_model)

    # Expected output
    building_id = ["osm_1", "tile_8", "shm_1"]
    loss = [43362.325643, 2765141.383750, 31697.200000]

    expected_loss_summary = pd.DataFrame(
        {"loss": loss},
        index=building_id,
    )

    assert len(returned_loss_summary.columns) == len(expected_loss_summary.columns)
    assert len(returned_loss_summary.index) == len(expected_loss_summary.index)

    for index in expected_loss_summary.index:
        assert round(returned_loss_summary.loc[index, "loss"], 5) == round(
            expected_loss_summary.loc[index, "loss"], 5
        )
