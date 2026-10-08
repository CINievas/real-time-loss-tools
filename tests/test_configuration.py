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

import pytest
from realtimelosstools.configuration import Configuration


@pytest.mark.parametrize(
    "value, expected",
    [
        ("all", "all"),
        ("ALL", "all"),
        (" All ", "all"),
        (1, 1),
        (4, 4),
        ("4", 4),
        (4.0, 4),
        (0, None),
        (-1, None),
        ("0", None),
        (2.5, None),
        ("many", None),
        (True, None),
        (False, None),
    ],
)
def test_assign_number_oq_cores(value, expected):
    # The method does not need the contents of a configuration file
    config_object = Configuration.__new__(Configuration)

    config = {"number_oq_cores": value}
    returned = config_object.assign_number_oq_cores(config, "number_oq_cores")

    assert returned == expected
    assert type(returned) is type(expected)


def test_assign_number_oq_cores_missing():
    config_object = Configuration.__new__(Configuration)

    assert config_object.assign_number_oq_cores({}, "number_oq_cores") is None
