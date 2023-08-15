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
import logging
import numpy as np
import pandas as pd
import pytz


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


class Time():
    """This class handles operations associated with time.
    """

    @staticmethod
    def determine_local_time_from_utc(utc_time_naive, local_timezone):
        """
        This method converts the datetime object 'utc_time', assumed to be in UTC, into the
        specified 'local_timezone'.

        Args:
            utc_time_naive (datetime object):
                Datetime object, assumed to be in UTC, with no timezone defined.
            local_timezone (str):
                Local time zone in the format of the IANA Time Zone Database.
                E.g. "Europe/Rome".

        Returns:
            local_time (datetime object):
                Datetime object equivalent to 'utc_time' in the target 'local_timezone'.
        """

        # Assume time zone of 'utc_time' is UTC (assign it)
        utc_time_aware = utc_time_naive.replace(tzinfo=pytz.UTC)

        # Convert into local time
        local_time_aware = utc_time_aware.astimezone(pytz.timezone(local_timezone))

        return local_time_aware

    @staticmethod
    def interpret_time_of_the_day(local_hour):
        """This method interprets a time of the day as corresponding to the "day", "night" or
        "transit" period, in the following way:
            Day: 10 am (inclusive) to 6 pm (exclusive).
            Night: 10 pm (inclusive) to 6 am (exclusive).
            Transit: 6 am (inclusive) to 10 am (exclusive), and 6 pm (inclusive) to 10 pm
                (exclusive).

        Args:
            local_hour (int):
                Hour of the day (in local time), as an integer equal to or larger than 0 and smaller
                than 24.

        Returns:
            time_of_day (str):
                "day", "night", "transit", "error" (if local_hour is an integer smaller than 0 or
                equal to or larger than 24).
        """

        if local_hour >= 10 and local_hour < 18:
            time_of_day = "day"
        elif (local_hour >= 22 and local_hour < 24) or (local_hour >= 0 and local_hour < 6):
            time_of_day = "night"
        elif (local_hour >= 6 and local_hour < 10) or (local_hour >= 18 and local_hour < 22):
            time_of_day = "transit"
        else:
            time_of_day = "error"

        return time_of_day


class Files():
    """This class handles operations associated with data/text files.
    """

    @staticmethod
    def find_string_in_file(path_to_file, target_str):
        """
        This method searches for the string 'target_str' within the text file 'path_to_file' and
        returns True if it finds it and false otherwise.
        """

        exists = False  # initialise

        with open(path_to_file, "r") as file:
            content = file.read()
            if target_str in content:
                exists = True

        return exists


class Loader():
    """This class handles operations associated with loading and/or validating input files.
    """

    @staticmethod
    def load_triggers(path_to_file, catalogues_path):
        """
        This method loads the triggering CSV file and:
        - checks that the columns "catalogue_filename" and "type_analysis" exist (it raises an
        error if at least one of the two does not exist);
        - checks that the column "type_analysis" only contains "OELF" or "RLA" and raises an
        error otherwise;
        - adds the "rupture_xml" column if it does not exist, and fills it in with empty
        strings;
        - verifies that all catalogue CSV files exist, and raises an error if at least one of
        them does not.

        Args:
            path_to_file (str):
                Path to the triggering CSV file (including name and extension of file).
            catalogues_path (str):
                Path to directory where earthquake catalogues are located.

        Returns:
            triggers (Pandas DataFrame):
                DataFrame with the contents of the triggering CSV file and the following
                columns:
                    catalogue_filename (str):
                        Name of the catalogue CSV files (which exist under
                        main_path/catalogues).
                    type_analysis (str):
                        Type of analysis to run with the corresponding catalogue, either "RLA"
                        (rapid loss assessment) or "OELF" (operational earthquake loss
                        forecast).
                    rupture_xml (str):
                        Name of the rupture XML files for the RLA analyses (which exist under
                        main_path/ruptures/rla). This field can be empty for some/all rows.
        """

        triggers = pd.read_csv(path_to_file)
        triggers = triggers.fillna("")

        # Make sure columns "catalogue_filename" and "type_analysis" exist
        error_message = None
        if "catalogue_filename" not in triggers.columns:
            error_message = "catalogue_filename column does not exist in triggering.csv"
        if "type_analysis" not in triggers.columns:
            error_message = "type_analysis column does not exist in triggering.csv"

        if error_message is not None:
            raise OSError(error_message)

        # Make sure "type_analysis" contains only "OELF" or "RLA"
        type_analysis_unique = triggers["type_analysis"].unique()
        non_recognised_types = []

        for type_val in type_analysis_unique:
            if type_val not in ["OELF", "RLA"]:
                non_recognised_types.append(type_val)

        if len(non_recognised_types) != 0:
            raise ValueError(
                "Analysis type(s) not recognised in triggering.csv file "
                "(supported: OELF, RLA): %s"
                % (", ".join(non_recognised_types))
            )

        # Add "rupture_xml" column if it does not exist (fill in with empty strings)
        if "rupture_xml" not in triggers.columns:
            triggers["rupture_xml"] = ["" for i in range(triggers.shape[0])]

        # Check if all catalogue files exist
        missing_triggers = []
        for i in range(triggers.shape[0]):
            # Check catalogue file exists
            if not os.path.isfile(
                os.path.join(catalogues_path, triggers.loc[i, "catalogue_filename"])
            ):
                missing_triggers.append(triggers.loc[i, "catalogue_filename"])

            if len(missing_triggers) != 0:
                raise OSError(
                    "The following trigger catalogue files cannot be found under %s: %s"
                    % (catalogues_path, ", ".join(missing_triggers))
                )

        return triggers
