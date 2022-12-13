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
import numpy as np


logger = logging.getLogger()

class MultilinearStepFunction():
    """
    This class defines a multilinear step function whose independent and dependent variables are
    labelled "thresholds" and "values", respectively.

    When initialised, the two input arrays are re-ordered so that the elements of "thresholds"
    are listed in increasing order.

    When evaluated, the function returns zero for input values smaller than the smallest value
    of "thresholds" and the last value of "values" for input values larger than the largest
    element of "thresholds". When input the value of the smallest element of "thresholds", it
    returns the first element of "values". In all other cases where the input value is an
    element in position "i" of "thresholds", it returns the element in position "i-1" of
    "values".

    If "thresholds" are floats, use the method 'evaluate_as_float'. If "thresholds" are
    numpy.datetime64 objects, use the method 'evaluate_as_datetime'.
    """

    def __init__(self, array_x, array_y):
        order = np.argsort(array_x)
        self.thresholds = array_x[order]
        self.values = array_y[order]

    def evaluate_as_float(self, x):
        """Evaluate the MultilinearStepFunction at "x", with "thresholds" being floats."""

        if x < self.thresholds[0]:
            return 0.0

        if round(x, 12) == round(self.thresholds[0], 12):
            return self.values[0]

        which = np.searchsorted(self.thresholds, x)
        return self.values[which-1]

    def evaluate_as_datetime(self, x):
        """Evaluate the MultilinearStepFunction at "x", with "thresholds" being numpy.datetime64
        objects."""

        if x < self.thresholds[0]:
            return 0.0

        if x == self.thresholds[0]:
            return self.values[0]

        which = np.searchsorted(self.thresholds, x)
        return self.values[which-1]
