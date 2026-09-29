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
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from realtimelosstools.utils import (
    MultilinearStepFunction, Operations, Time, Files, Loader
)


def test_MultilinearStepFunction():
    # Test case 1: floats
    thresholds = np.array([1.0, 3.0, 0.0, 2.0])
    values = np.array([20.0, 5.0, 30.0, 10.0])
    f1 = MultilinearStepFunction(thresholds, values)

    # Test correct order
    np.testing.assert_almost_equal(
        f1.thresholds,
        np.array([0.0, 1.0, 2.0, 3.0])
    )
    np.testing.assert_almost_equal(
        f1.values,
        np.array([30.0, 20.0, 10.0, 5.0])
    )

    target_xvals = np.array([
        -1E-5, 0.0, 0.5, 1.0, 1.01, 2.0, 2.01, 3.0, 3.01, 999.9
    ])
    expected_outputs = np.array([
        0.0, 30.0, 30.0, 30.0, 20.0, 20.0, 10.0, 10.0, 5.0, 5.0
    ])

    # Evaluate and test all target_xvals together:
    returned_outputs = f1.evaluate(target_xvals)
    assert np.isclose(returned_outputs, expected_outputs).all()

    # Evaluate each element of target_xvals individually
    for i in range(len(target_xvals)):
        assert round(f1.evaluate(target_xvals[i]), 10) == expected_outputs[i]

    # Test case 2: dates
    thresholds = np.array([
        "2009-04-06T02:37:00",
        "2009-04-07T09:26:00",
        "2009-04-06T01:32:00",
        "2009-04-06T23:15:00"
    ], dtype=np.datetime64)
    values = np.array([15.0, 3.0, 25.0, 8.0])
    f2 = MultilinearStepFunction(thresholds, values)

    # Test correct order
    np.testing.assert_almost_equal(
        f2.values,
        np.array([25.0, 15.0, 8.0, 3.0])
    )
    expected_thresholds = np.array([
            "2009-04-06T01:32:00",
            "2009-04-06T02:37:00",
            "2009-04-06T23:15:00",
            "2009-04-07T09:26:00"
    ], dtype=np.datetime64)

    for i in range(len(expected_thresholds)):
        assert f2.thresholds[i] == expected_thresholds[i]

    target_xvals = np.array([
        "2009-04-06T01:31:59", "2009-04-06T01:32:00",
        "2009-04-06T02:37:00", "2009-04-06T02:37:01",
        "2009-04-06T23:15:00", "2009-04-06T23:15:01",
        "2009-04-07T09:26:00", "2009-04-07T09:26:01",
        "2019-12-31T09:26:01"
    ])
    target_xvals = target_xvals.astype(np.datetime64)
    expected_outputs = np.array([
        0.0, 25.0, 25.0, 15.0, 15.0, 8.0, 8.0, 3.0, 3.0
    ])

    # Evaluate and test all target_xvals together:
    returned_outputs = f2.evaluate(target_xvals)
    assert np.isclose(returned_outputs, expected_outputs).all()

    # Evaluate each element of target_xvals individually
    for i in range(len(target_xvals)):
        assert round(f2.evaluate(target_xvals[i]), 10) == expected_outputs[i]


def test_Operations_shorten_x():

    filepath = os.path.join(
        os.path.dirname(__file__), "data", "nuisance_x_array.csv"
    )

    df = pd.read_csv(filepath)
    original_x = df["x_val"].to_numpy()

    returned_x = Operations.shorten_x(original_x, max_n=100)

    assert len(returned_x) <= 100
    assert round(returned_x[0], 8) == round(original_x[0], 8)
    assert round(returned_x[-1], 8) == round(original_x[-1], 8)

    for x_val in returned_x:
        assert x_val in original_x


def test_interpret_time_of_the_day():
    # Simulate an input catalogue DataFrame (including expected outputs)
    cat_df = pd.DataFrame(
        {
            "local_time": [
                datetime(2026, 8, 24, 0, 0, 0),
                datetime(2026, 8, 24, 0, 17, 0),
                datetime(2026, 8, 24, 1, 17, 0),
                datetime(2026, 8, 24, 2, 17, 0),
                datetime(2026, 8, 24, 3, 17, 0),
                datetime(2026, 8, 24, 4, 17, 0),
                datetime(2026, 8, 24, 5, 17, 0),
                datetime(2026, 8, 24, 6, 17, 0),
                datetime(2026, 8, 24, 7, 17, 0),
                datetime(2026, 8, 24, 8, 17, 0),
                datetime(2026, 8, 24, 9, 17, 0),
                datetime(2026, 8, 24, 10, 17, 0),
                datetime(2026, 8, 24, 11, 17, 0),
                datetime(2026, 8, 24, 12, 17, 0),
                datetime(2026, 8, 24, 13, 17, 0),
                datetime(2026, 8, 24, 14, 17, 0),
                datetime(2026, 8, 24, 15, 17, 0),
                datetime(2026, 8, 24, 16, 17, 0),
                datetime(2026, 8, 24, 17, 17, 0),
                datetime(2026, 8, 24, 18, 17, 0),
                datetime(2026, 8, 24, 19, 17, 0),
                datetime(2026, 8, 24, 20, 17, 0),
                datetime(2026, 8, 24, 21, 59, 59),
                datetime(2026, 8, 24, 22, 0, 0),
                datetime(2026, 8, 24, 23, 17, 0),
                datetime(2026, 8, 24, 23, 59, 59)
            ],
            "expected": [
                "night",
                "night",
                "night",
                "night",
                "night",
                "night",
                "night",
                "transit",
                "transit",
                "transit",
                "transit",
                "day",
                "day",
                "day",
                "day",
                "day",
                "day",
                "day",
                "day",
                "transit",
                "transit",
                "transit",
                "transit",
                "night",
                "night",
                "night"
            ]  # Expected outputs
        }
    )

    cat_df["returned"] = Time.interpret_time_of_the_day(cat_df["local_time"])

    assert (cat_df["returned"] == cat_df["expected"]).all()


def test_determine_local_time_from_utc():
    # Simulate an input catalogue DataFrame (including expected outputs)
    cat_df = pd.DataFrame(
        {
            "datetime": [
                datetime(2023, 1, 3, 17, 45, 27),
                datetime(2023, 4, 3, 17, 45, 27),
                datetime(2009, 4, 6, 1, 32, 0),
                datetime(2016, 8, 24, 1, 36, 0),
                datetime(2016, 10, 26, 19, 18, 0),
                datetime(2016, 10, 30, 6, 40, 0)
            ],
            "comments": [
                "Test Central Europe winter time",
                "Test Central Europe summer time",
                "L'Aquila earthquake main shock",
                "2016 Central It, 1st large shock",
                "2016 Central It, 2nd large shock of 2nd phase, days before end of daylight saving time",
                "2016 Central It, 2nd large shock of 2nd phase, on day of end of daylight saving time",
            ],
            "expected": [
                datetime(2023, 1, 3, 18, 45, 27),
                datetime(2023, 4, 3, 19, 45, 27),
                datetime(2009, 4, 6, 3, 32, 0),
                datetime(2016, 8, 24, 3, 36, 0),
                datetime(2016, 10, 26, 21, 18, 0),
                datetime(2016, 10, 30, 7, 40, 0)
            ]  # Expected outputs
        }
    )
    cat_df["expected"] = (
        pd.to_datetime(cat_df["expected"])
        .dt.tz_localize("Europe/Rome")
    )  # Assign time zone to this originally-naive column

    # Local timezone to convert to
    local_timezone = "Europe/Rome"

    returned_timestamps = Time.determine_local_time_from_utc(cat_df["datetime"], local_timezone)

    assert (returned_timestamps.astype("int64") == cat_df["expected"].astype("int64")).all()


def test_integration_determine_local_time_from_utc_and_interpret_time_of_the_day():
    # Simulate an input catalogue DataFrame (including expected outputs)
    cat_df = pd.DataFrame(
        {
            "datetime": [
                datetime(2009, 4, 6, 1, 32, 0),
                datetime(2016, 8, 24, 1, 36, 0),
                datetime(2016, 10, 26, 19, 18, 0),
                datetime(2016, 10, 30, 6, 40, 0)
            ],
            "comments": [
                "L'Aquila earthquake main shock",
                "2016 Central It, 1st large shock",
                "2016 Central It, 2nd large shock of 2nd phase, days before end of daylight saving time",
                "2016 Central It, 2nd large shock of 2nd phase, on day of end of daylight saving time",
            ],
            "expected": [
                "night", "night", "transit", "transit"
            ]  # Expected outputs
        }
    )

    # Local timezone to convert to
    local_timezone = "Europe/Rome"

    local_timestamps = Time.determine_local_time_from_utc(cat_df["datetime"], local_timezone)
    returned_time_of_day = Time.interpret_time_of_the_day(local_timestamps)

    assert (returned_time_of_day == cat_df["expected"]).all()


def test_find_string_in_file():

    filepath = os.path.join(
        os.path.dirname(__file__), "data", "job.ini"
    )

    assert Files.find_string_in_file(filepath, "gmpe_logic_tree") is True
    assert Files.find_string_in_file(filepath, "something_else") is False


def test_load_triggers():
    # Test 1:
    # Columns "catalogue_filename" and "type_analysis" exist, "rupture_xml" does not
    path_to_file = os.path.join(os.path.dirname(__file__), "data", "triggering_01.csv")
    catalogues_path = os.path.join(os.path.dirname(__file__), "data")

    returned_triggers = Loader.load_triggers(path_to_file, catalogues_path)

    expected_triggers = pd.DataFrame(
        {
            "catalogue_filename": ["triggering_01_oelf_01.txt", "triggering_01_rla_01.csv"],
            "type_analysis": ["OELF", "RLA"],
            "rupture_xml": ["", ""],
        }
    )

    assert returned_triggers.shape[0] == expected_triggers.shape[0]
    assert returned_triggers.shape[1] == expected_triggers.shape[1]

    for col_name in expected_triggers.columns:
        assert col_name in returned_triggers.columns

    for col_name in expected_triggers.columns:
        for row_index in expected_triggers.index:
            assert (
                returned_triggers.loc[row_index, col_name]
                == expected_triggers.loc[row_index, col_name]
            )

    # Test 2:
    # Column "catalogue_filename" has wrong name
    path_to_file = os.path.join(os.path.dirname(__file__), "data", "triggering_02.csv")
    catalogues_path = os.path.join(os.path.dirname(__file__), "data")

    with pytest.raises(OSError) as excinfo:
        Loader.load_triggers(path_to_file, catalogues_path)
    assert "OSError" in str(excinfo.type)

    # Test 3:
    # Column "type_analysis" has a value different from "OELF" or "RLA"
    path_to_file = os.path.join(os.path.dirname(__file__), "data", "triggering_03.csv")
    catalogues_path = os.path.join(os.path.dirname(__file__), "data")

    with pytest.raises(ValueError) as excinfo:
        Loader.load_triggers(path_to_file, catalogues_path)
    assert "ValueError" in str(excinfo.type)

    # Test 4:
    # One of the catalogue files does not exist
    path_to_file = os.path.join(os.path.dirname(__file__), "data", "triggering_04.csv")
    catalogues_path = os.path.join(os.path.dirname(__file__), "data")

    with pytest.raises(OSError) as excinfo:
        Loader.load_triggers(path_to_file, catalogues_path)
    assert "OSError" in str(excinfo.type)

    # Test 5:
    # Column "rupture_xml" exists and has empty values (NaNs to be treated as "")
    path_to_file = os.path.join(os.path.dirname(__file__), "data", "triggering_05.csv")
    catalogues_path = os.path.join(os.path.dirname(__file__), "data")

    returned_triggers = Loader.load_triggers(path_to_file, catalogues_path)

    expected_triggers = pd.DataFrame(
        {
            "catalogue_filename": ["triggering_01_oelf_01.txt", "triggering_01_rla_01.csv"],
            "type_analysis": ["OELF", "RLA"],
            "rupture_xml": ["", "rupture_01.xml"],
        }
    )

    assert returned_triggers.shape[0] == expected_triggers.shape[0]
    assert returned_triggers.shape[1] == expected_triggers.shape[1]

    for col_name in expected_triggers.columns:
        assert col_name in returned_triggers.columns

    for col_name in expected_triggers.columns:
        for row_index in expected_triggers.index:
            assert (
                returned_triggers.loc[row_index, col_name]
                == expected_triggers.loc[row_index, col_name]
            )


def test_load_OELF_weights():
    # Test 1: all three columns exist, weights add up to 1
    path_to_file = os.path.join(
        os.path.dirname(__file__), "data", "seismicity_models_weights_01.csv"
    )

    returned_weights = Loader.load_OELF_weights("from_file", 2, path_to_file)

    expected_weights = pd.DataFrame(
        {"model_id": [1, 2], "model_name": ["Model_A", "Model_B"], "weight": [0.25, 0.75]}
    )
    expected_weights.set_index(expected_weights["model_id"], inplace=True)
    expected_weights.drop(columns=["model_id"], inplace=True)

    assert returned_weights.shape[0] == expected_weights.shape[0]
    assert returned_weights.shape[1] == expected_weights.shape[1]

    for model_id in expected_weights.index:
        assert (
            returned_weights.loc[model_id, "model_name"] ==
            expected_weights.loc[model_id, "model_name"]
        )
        assert (
            round(returned_weights.loc[model_id, "weight"], 5) ==
            round(expected_weights.loc[model_id, "weight"], 5)
        )

    # Test 2: all three columns exist, weights do not add up to 1
    path_to_file = os.path.join(
        os.path.dirname(__file__), "data", "seismicity_models_weights_02.csv"
    )

    with pytest.raises(OSError) as excinfo:
        Loader.load_OELF_weights("from_file", 2, path_to_file)
    assert "OSError" in str(excinfo.type)

    # Test 3: the "model_name" column is missing, weights add up to 1
    path_to_file = os.path.join(
        os.path.dirname(__file__), "data", "seismicity_models_weights_03.csv"
    )

    returned_weights = Loader.load_OELF_weights("from_file", 2, path_to_file)

    expected_weights = pd.DataFrame(
        {"model_id": [1, 2], "model_name": ["1", "2"], "weight": [0.25, 0.75]}
    )
    expected_weights.set_index(expected_weights["model_id"], inplace=True)
    expected_weights.drop(columns=["model_id"], inplace=True)

    assert returned_weights.shape[0] == expected_weights.shape[0]
    assert returned_weights.shape[1] == expected_weights.shape[1]

    for model_id in expected_weights.index:
        assert (
            returned_weights.loc[model_id, "model_name"] ==
            expected_weights.loc[model_id, "model_name"]
        )
        assert (
            round(returned_weights.loc[model_id, "weight"], 5) ==
            round(expected_weights.loc[model_id, "weight"], 5)
        )

    # Test 4: all three columns exist, but "weight" column is called "weights"
    # and thus does not exist
    path_to_file = os.path.join(
        os.path.dirname(__file__), "data", "seismicity_models_weights_04.csv"
    )

    with pytest.raises(OSError) as excinfo:
        Loader.load_OELF_weights("from_file", 2, path_to_file)
    assert "OSError" in str(excinfo.type)

    # Test 5: create uniform weights
    path_to_file = os.path.join(
        os.path.dirname(__file__), "data", "name_is_irrelevant.csv"
    )

    returned_weights = Loader.load_OELF_weights("uniform", 5, path_to_file)

    expected_weights = pd.DataFrame(
        {
            "model_id": [1, 2, 3, 4, 5],
            "model_name": ["1", "2", "3", "4", "5"],
            "weight": [0.2, 0.2, 0.2, 0.2, 0.2]
        }
    )
    expected_weights.set_index(expected_weights["model_id"], inplace=True)
    expected_weights.drop(columns=["model_id"], inplace=True)

    assert returned_weights.shape[0] == expected_weights.shape[0]
    assert returned_weights.shape[1] == expected_weights.shape[1]

    for model_id in expected_weights.index:
        assert (
            returned_weights.loc[model_id, "model_name"] ==
            expected_weights.loc[model_id, "model_name"]
        )
        assert (
            round(returned_weights.loc[model_id, "weight"], 5) ==
            round(expected_weights.loc[model_id, "weight"], 5)
        )

    # Test 6: "seismicity_models_weights" parameter is not supported
    path_to_file = os.path.join(
        os.path.dirname(__file__), "data", "name_is_irrelevant.csv"
    )

    with pytest.raises(OSError) as excinfo:
        Loader.load_OELF_weights("something_random", 2, path_to_file)
    assert "OSError" in str(excinfo.type)
