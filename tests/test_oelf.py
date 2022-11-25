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
import numpy as np
import pandas as pd
from realtimelosstools.oelf import OperationalEarthquakeLossForecasting


def test_format_seismicity_forecast():
    filepath = os.path.join(os.path.dirname(__file__), "data", "oef_catalogue.csv")
    forecast = pd.read_csv(filepath)

    # Test case 1: add both event_id and depth
    returned_forecast = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast, add_event_id=True, add_depth=True
    )
    expected_forecast = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_expected_add_both.csv"),
        sep=",",
    )
    expected_forecast["datetime"] = pd.to_datetime(expected_forecast["datetime"])
    expected_forecast.set_index("EQID", drop=True, inplace=True)

    assert len(returned_forecast.index) == len(expected_forecast.index)
    assert len(returned_forecast.columns) == len(expected_forecast.columns)

    for eqid in expected_forecast.index:
        for column in expected_forecast.columns:
            if column in ["longitude", "latitude", "magnitude"]:
                assert round(returned_forecast.loc[eqid, column], 5) == round(
                    expected_forecast.loc[eqid, column], 5
                )
            elif column in ["datetime", "ses_id", "event_id"]:
                assert returned_forecast.loc[eqid, column] == expected_forecast.loc[eqid, column]
            elif column in ["depth"]:
                if np.isnan(expected_forecast.loc[eqid, column]):
                    assert np.isnan(returned_forecast.loc[eqid, column])
                else:
                    assert round(returned_forecast.loc[eqid, column], 5) == round(
                        expected_forecast.loc[eqid, column], 5
                    )

    # Test case 2: add only event_id
    returned_forecast = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast, add_event_id=True, add_depth=False
    )
    expected_forecast = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_expected_add_event_id.csv"),
        sep=",",
    )
    expected_forecast["datetime"] = pd.to_datetime(expected_forecast["datetime"])
    expected_forecast.set_index("EQID", drop=True, inplace=True)

    assert len(returned_forecast.index) == len(expected_forecast.index)
    assert len(returned_forecast.columns) == len(expected_forecast.columns)
    assert "depth" not in returned_forecast.columns

    for eqid in expected_forecast.index:
        for column in expected_forecast.columns:
            if column in ["longitude", "latitude", "magnitude"]:
                assert round(returned_forecast.loc[eqid, column], 5) == round(
                    expected_forecast.loc[eqid, column], 5
                )
            elif column in ["datetime", "ses_id", "event_id"]:
                assert returned_forecast.loc[eqid, column] == expected_forecast.loc[eqid, column]
            elif column in ["depth"]:
                if np.isnan(expected_forecast.loc[eqid, column]):
                    assert np.isnan(returned_forecast.loc[eqid, column])
                else:
                    assert round(returned_forecast.loc[eqid, column], 5) == round(
                        expected_forecast.loc[eqid, column], 5
                    )

    # Test case 3: add only depth
    returned_forecast = OperationalEarthquakeLossForecasting.format_seismicity_forecast(
        forecast, add_event_id=False, add_depth=True
    )
    expected_forecast = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_expected_add_depth.csv"),
        sep=",",
    )
    expected_forecast["datetime"] = pd.to_datetime(expected_forecast["datetime"])

    assert len(returned_forecast.index) == len(expected_forecast.index)
    assert len(returned_forecast.columns) == len(expected_forecast.columns)
    assert "event_id" not in returned_forecast.columns

    for row in expected_forecast.index:
        for column in expected_forecast.columns:
            if column in ["longitude", "latitude", "magnitude"]:
                assert round(returned_forecast.loc[row, column], 5) == round(
                    expected_forecast.loc[row, column], 5
                )
            elif column in ["datetime", "ses_id", "event_id"]:
                assert returned_forecast.loc[row, column] == expected_forecast.loc[row, column]
            elif column in ["depth"]:
                if np.isnan(expected_forecast.loc[row, column]):
                    assert np.isnan(returned_forecast.loc[row, column])
                else:
                    assert round(returned_forecast.loc[row, column], 5) == round(
                        expected_forecast.loc[row, column], 5
                    )


def test_filter_forecast():
    filepath = os.path.join(os.path.dirname(__file__), "data", "oef_catalogue.csv")
    input_forecast = pd.read_csv(filepath)
    input_forecast["aux_id"] = [i for i in range(input_forecast.shape[0])]
    input_forecast = input_forecast.rename(columns={
        "Mag": "magnitude", "Lon": "longitude", "Lat": "latitude"
    })

    filepath = os.path.join(os.path.dirname(__file__), "data", "oef_catalogue_filtered.csv")
    expected_filtered_cat = pd.read_csv(filepath)

    expected_kept = np.array(
        [True, False, False, True, False, False, False, False, False, False]
    )

    exposure_lons = np.array([13.400949, 13.3888, 13.400949])
    exposure_lats = np.array([42.344967, 42.344967 ,42.3358])

    magnitude_min = 4.0
    distance_max = 2.5

    returned_filtered_cat, returned_kept = OperationalEarthquakeLossForecasting.filter_forecast(
        input_forecast, exposure_lons, exposure_lats, magnitude_min, distance_max
    )

    assert returned_filtered_cat.shape[0] == expected_filtered_cat.shape[0]

    assert np.all(returned_filtered_cat.index == expected_filtered_cat["aux_id"].to_numpy())

    assert np.all(returned_kept == expected_kept)

    for aux_id in expected_filtered_cat["aux_id"].to_numpy():
        assert aux_id in returned_filtered_cat["aux_id"].to_numpy()
