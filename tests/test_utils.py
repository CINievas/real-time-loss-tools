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

import numpy as np
from realtimelosstools.utils import MultilinearStepFunction


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
