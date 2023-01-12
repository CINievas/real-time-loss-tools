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
import pytz
import numpy as np
from datetime import datetime
from realtimelosstools.utils import MultilinearStepFunction, Time, Files


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

    # Test correct outputs
    assert round(f1.evaluate_as_float(-1E-5), 10) == 0.0
    assert round(f1.evaluate_as_float(0.0), 10) == 30.0
    assert round(f1.evaluate_as_float(0.5), 10) == 30.0
    assert round(f1.evaluate_as_float(1.0), 10) == 30.0
    assert round(f1.evaluate_as_float(1.01), 10) == 20.0
    assert round(f1.evaluate_as_float(2.0), 10) == 20.0
    assert round(f1.evaluate_as_float(2.01), 10) == 10.0
    assert round(f1.evaluate_as_float(3.0), 10) == 10.0
    assert round(f1.evaluate_as_float(3.01), 10) == 5.0
    assert round(f1.evaluate_as_float(999.9), 10) == 5.0

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

    # Test correct outputs
    assert round(f2.evaluate_as_datetime(np.datetime64("2009-04-06T01:31:59")), 10) == 0.0
    assert round(f2.evaluate_as_datetime(np.datetime64("2009-04-06T01:32:00")), 10) == 25.0
    assert round(f2.evaluate_as_datetime(np.datetime64("2009-04-06T02:37:00")), 10) == 25.0
    assert round(f2.evaluate_as_datetime(np.datetime64("2009-04-06T02:37:01")), 10) == 15.0
    assert round(f2.evaluate_as_datetime(np.datetime64("2009-04-06T23:15:00")), 10) == 15.0
    assert round(f2.evaluate_as_datetime(np.datetime64("2009-04-06T23:15:01")), 10) == 8.0
    assert round(f2.evaluate_as_datetime(np.datetime64("2009-04-07T09:26:00")), 10) == 8.0
    assert round(f2.evaluate_as_datetime(np.datetime64("2009-04-07T09:26:01")), 10) == 3.0
    assert round(f2.evaluate_as_datetime(np.datetime64("2019-12-31T09:26:01")), 10) == 3.0


def test_interpret_time_of_the_day():
    assert Time.interpret_time_of_the_day(0) == "night"
    assert Time.interpret_time_of_the_day(1) == "night"
    assert Time.interpret_time_of_the_day(2) == "night"
    assert Time.interpret_time_of_the_day(3) == "night"
    assert Time.interpret_time_of_the_day(4) == "night"
    assert Time.interpret_time_of_the_day(5) == "night"
    assert Time.interpret_time_of_the_day(6) == "transit"
    assert Time.interpret_time_of_the_day(7) == "transit"
    assert Time.interpret_time_of_the_day(8) == "transit"
    assert Time.interpret_time_of_the_day(9) == "transit"
    assert Time.interpret_time_of_the_day(10) == "day"
    assert Time.interpret_time_of_the_day(11) == "day"
    assert Time.interpret_time_of_the_day(12) == "day"
    assert Time.interpret_time_of_the_day(13) == "day"
    assert Time.interpret_time_of_the_day(14) == "day"
    assert Time.interpret_time_of_the_day(15) == "day"
    assert Time.interpret_time_of_the_day(16) == "day"
    assert Time.interpret_time_of_the_day(17) == "day"
    assert Time.interpret_time_of_the_day(18) == "transit"
    assert Time.interpret_time_of_the_day(19) == "transit"
    assert Time.interpret_time_of_the_day(20) == "transit"
    assert Time.interpret_time_of_the_day(21) == "transit"
    assert Time.interpret_time_of_the_day(22) == "night"
    assert Time.interpret_time_of_the_day(23) == "night"
    assert Time.interpret_time_of_the_day(24) == "error"
    assert Time.interpret_time_of_the_day(28) == "error"


def test_determine_local_time_from_utc():
    # Test Central Europe winter time
    returned_timestamp = Time.determine_local_time_from_utc(
        datetime(2023, 1, 3, 17, 45, 27), "Europe/Rome"
    ).timestamp()

    expected_timestamp = (
        pytz.timezone("Europe/Rome").localize(datetime(2023, 1, 3, 18, 45, 27)).timestamp()
    )

    assert round(returned_timestamp, 10) == round(expected_timestamp, 10)

    # Test Central Europe summer time
    returned_timestamp = Time.determine_local_time_from_utc(
        datetime(2023, 4, 3, 17, 45, 27), "Europe/Rome"
    ).timestamp()

    expected_timestamp = (
        pytz.timezone("Europe/Rome").localize(datetime(2023, 4, 3, 19, 45, 27)).timestamp()
    )

    assert round(returned_timestamp, 10) == round(expected_timestamp, 10)

    # Test with times of relevant Italian earthquakes
    ## Italian #1, L'Aquila main shock
    returned_timestamp = Time.determine_local_time_from_utc(
        datetime(2009, 4, 6, 1, 32, 0), "Europe/Rome"
    ).timestamp()

    expected_timestamp = (
        pytz.timezone("Europe/Rome").localize(datetime(2009, 4, 6, 3, 32, 0)).timestamp()
    )

    assert round(returned_timestamp, 10) == round(expected_timestamp, 10)

    ## Italian #2, first large shock of 2016 Central Italy sequence
    returned_timestamp = Time.determine_local_time_from_utc(
        datetime(2016, 8, 24, 1, 36, 0), "Europe/Rome"
    ).timestamp()

    expected_timestamp = (
        pytz.timezone("Europe/Rome").localize(datetime(2016, 8, 24, 3, 36, 0)).timestamp()
    )

    assert round(returned_timestamp, 10) == round(expected_timestamp, 10)

    ## Italian #3, second large shock of second phase of 2016 Central Italy sequence
    ## (a few days before the end of the end of the 2016 daylight saving time)
    returned_timestamp = Time.determine_local_time_from_utc(
        datetime(2016, 10, 26, 19, 18, 0), "Europe/Rome"
    ).timestamp()

    expected_timestamp = (
        pytz.timezone("Europe/Rome").localize(datetime(2016, 10, 26, 21, 18, 0)).timestamp()
    )

    assert round(returned_timestamp, 10) == round(expected_timestamp, 10)

    ## Italian #4, third large shock of second phase of 2016 Central Italy sequence
    ## (on the day of the end of the 2016 daylight saving time, a few hours later)
    returned_timestamp = Time.determine_local_time_from_utc(
        datetime(2016, 10, 30, 6, 40, 0), "Europe/Rome"
    ).timestamp()

    expected_timestamp = (
        pytz.timezone("Europe/Rome").localize(datetime(2016, 10, 30, 7, 40, 0)).timestamp()
    )

    assert round(returned_timestamp, 10) == round(expected_timestamp, 10)


def test_integration_determine_local_time_from_utc_and_interpret_time_of_the_day():
    # Test with times of relevant Italian earthquakes
    ## Italian #1, L'Aquila main shock
    returned_datetime = Time.determine_local_time_from_utc(
        datetime(2009, 4, 6, 1, 32, 0), "Europe/Rome"
    )

    assert Time.interpret_time_of_the_day(returned_datetime.hour) == "night"

    ## Italian #2, first large shock of 2016 Central Italy sequence
    returned_datetime = Time.determine_local_time_from_utc(
        datetime(2016, 8, 24, 1, 36, 0), "Europe/Rome"
    )

    assert Time.interpret_time_of_the_day(returned_datetime.hour) == "night"

    ## Italian #3, second large shock of second phase of 2016 Central Italy sequence
    ## (a few days before the end of the end of the 2016 daylight saving time)
    returned_datetime = Time.determine_local_time_from_utc(
        datetime(2016, 10, 26, 19, 18, 0), "Europe/Rome"
    )

    assert Time.interpret_time_of_the_day(returned_datetime.hour) == "transit"

    ## Italian #4, third large shock of second phase of 2016 Central Italy sequence
    ## (on the day of the end of the 2016 daylight saving time, a few hours later)
    returned_datetime = Time.determine_local_time_from_utc(
        datetime(2016, 10, 30, 6, 40, 0), "Europe/Rome"
    )

    assert Time.interpret_time_of_the_day(returned_datetime.hour) == "transit"


def test_find_string_in_file():

    filepath = os.path.join(
        os.path.dirname(__file__), "data", "job.ini"
    )

    assert Files.find_string_in_file(filepath, "gmpe_logic_tree") is True
    assert Files.find_string_in_file(filepath, "something_else") is False
