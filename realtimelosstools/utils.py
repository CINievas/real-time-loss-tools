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
    """

    def __init__(self, array_x, array_y):
        order = np.argsort(array_x)
        self.thresholds = array_x[order]
        self.values = array_y[order]

    def evaluate(self, x):
        """
        Evaluate the MultilinearStepFunction at "x".

        "x" can be a scalar or an array, made of floats or datetime64 objects.
        """

        x_arr = np.asarray(x)

        # Indices of first threshold >= x (for all elements of x_array)
        idx = np.searchsorted(self.thresholds, x_arr, side="left")

        # Initialise output with zeroes (will remain zeroes for x_arr < self.thresholds[0])
        out = np.zeros_like(x_arr, dtype=self.values.dtype)

        # Evaluate for all x_arr > self.thresholds[0]
        valid = idx > 0
        out[valid] = self.values[idx[valid] - 1]

        # Evaluate for x_arr == self.thresholds[0], if it exists in x_arr
        mask_first = (x_arr == self.thresholds[0])
        out[mask_first] = self.values[0]

        return out if out.ndim > 0 else out.item()


class Operations():

    @staticmethod
    def shorten_x(x, max_n=100):
        """Reduce a sorted array of unique values to at most max_n of its own values.

        Half of the points are taken at evenly spaced ranks (following the data density), the
        other half at log-spaced targets (covering the tails), each snapped to the nearest
        actual value of input 'x'. The first and last elements of 'x' are always kept.

        Args:
            x (numpy array):
                1D array, sorted from smallest to largest.
            max_n (int):
                Maximum number of points of 'x' to keep.
        """
        # Number of values in the input array (already sorted and unique)
        n = x.shape[0]

        if n <= max_n:
            return x

        # Part 1: points evenly spaced by their position
        # (max - min) / number of points (but here only positions are determined, not values)
        # Use half of the maximum allowed points for this part
        n_rank = max_n // 2

        # Evenly spaced positions between the first (0) and last (n-1) index of 'x', ends incl.
        rank_idx = np.linspace(0, n - 1, n_rank).round().astype(int)

        # Part 2: points evenly spaced in value, snapped to the nearest actual data value of 'x'
        # Target values between the minimum and maximum of 'x', using the remaining points.
        # Logarithmic spacing is used when possible (all values positive), otherwise linear
        # spacing is used.
        if x[0] > 0:
            targets = np.geomspace(x[0], x[-1], max_n - n_rank)
        else:
            targets = np.linspace(x[0], x[-1], max_n - n_rank)

        # For each target value, find the index of the first value of 'x' that is >= target
        # (clip(1, n - 1) makes sure that both 'right' and 'right - 1' are valid indices).
        right = np.searchsorted(x, targets).clip(1, n - 1)
        # Index of the value of 'x' immediately below (or equal to) each target
        left = right - 1

        # For each target, keep whichever of the two neighbouring values of 'x' is closer
        # (ties go to the left one).
        snap_idx = np.where(targets - x[left] <= x[right] - targets, left, right)

        # Combine both sets of points, keep unique and return them sorted
        return x[np.unique(np.concatenate([rank_idx, snap_idx]))]


class Time():
    """This class handles operations associated with time.
    """

    @staticmethod
    def determine_local_time_from_utc(utc_time_naive, local_timezone):
        """
        This method converts the datetime64 objects in the 'utc_time_naive' Pandas Series,
        assumed to be in UTC, into the specified 'local_timezone'.

        Args:
            utc_time_naive (Pandas Series, dtype: naive datetime64):
                Pandas Series of UTC times, with no timezone defined (i.e., "naive").
            local_timezone (str):
                Local time zone in the format of the IANA Time Zone Database.
                E.g. "Europe/Rome".

        Returns:
            local_time_aware (Pandas Series, dtype: timezone-aware datetime64):
                Pandas Series of the equivalent local times of 'utc_time_naive' in the target
                'local_timezone'.
        """

        local_time_aware = (
            utc_time_naive
            .dt.tz_localize("UTC") # Assume time zone of 'utc_time' is UTC (assign it)
            .dt.tz_convert(local_timezone) # Convert into local time
        )

        return local_time_aware

    @staticmethod
    def interpret_time_of_the_day(local_time):
        """This method interprets a time of the day as corresponding to the "day", "night" or
        "transit" period, in the following way:
            Day: 10 am (inclusive) to 6 pm (exclusive).
            Night: 10 pm (inclusive) to 6 am (exclusive).
            Transit: 6 am (inclusive) to 10 am (exclusive), and 6 pm (inclusive) to 10 pm
                (exclusive).

        If the method encounters an element in 'local_time' that makes it impossible to assign a
        time of the day, an error is raised.

        Args:
            local_time (Pandas Series, dtype: timezone-aware datetime64):
                Hour of the day.

        Returns:
            time_of_day (Pandas Series):
                Pandas Series containing "day", "night", "transit".
        """
        hours = local_time.dt.hour

        bins = [-1, 5, 9, 17, 21, 24]  # -1 because pd.cut bins are left-open, right-closed
        labels = ["night", "transit", "day", "transit", "night"]

        time_of_day = pd.cut(hours, bins=bins, labels=labels, ordered=False)
        time_of_day = time_of_day.cat.add_categories(["error"]).fillna("error")

        if "error" in time_of_day:
            error_message = (
                "Something went wrong when determining the time of the day of at least "
                "one earthquake in the catalogue. The program cannot run."
            )
            logger.critical(error_message)
            raise OSError(error_message)

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

    @staticmethod
    def load_OELF_weights(config_weights_source, number_models, filepath, tolerance=1E-5):
        """
        This method returns a Pandas DataFrame with the weights of different seismicity models
        used to generate the input OELF catalogues.

        If 'config_weights_source' is "uniform", uniform weights (1/number_models) are assigned
        to all seismicity models.

        If 'config_weights_source' is "from_file", the weights are retrieved from the CSV file
        under 'static/seismicity_models_weights.csv', which is assumed to exist. This method
        checks that all weights add up to 1.0. If the CSV file does not contain a "model_name"
        column, it creates it by copying the "model_id" column

        Args:
            config_weights_source (str):
                Either "uniform" or "from_file" (see above).
            number_models (int):
                Number of seismicity models used to generate the input OELF catalogues.
            filepath (str):
                Path to CSV files with weights.
            tolerance (float):
                Tolerance to be used when checking that the sum of weights is 1.0. Default: 1E-5.

        Returns:
            oelf_model_weights (Pandas DataFrame):
                DataFrame with index "model_id" and the following columns:
                    model_name (str):
                        Name of the model.
                    weight (float):
                        Weight of the model.
        """

        if config_weights_source=="from_file":
            oelf_model_weights = pd.read_csv(filepath)
            if (
                ("model_id" not in oelf_model_weights.columns) or
                ("weight" not in oelf_model_weights.columns)
            ):
                error_message = (
                    "Loader.load_OELF_weights(): either 'model_id' or 'weight' are missing from "
                    "static/seismicity_models_weights.csv."
                )
                raise OSError(error_message)

            weights_sum = oelf_model_weights["weight"].sum()

            if ((weights_sum > 1.0 + tolerance) or (weights_sum < 1.0 - tolerance)):
                error_message = (
                    "Loader.load_OELF_weights(): the sum of weights of the different seismicity "
                    "models is %s, outside the tolerance (1-%s, 1+%s). Check "
                    "static/seismicity_models_weights.csv."
                    % (
                        "{:.10f}".format(weights_sum),
                        "{:.10f}".format(tolerance),
                        "{:.10f}".format(tolerance)
                    )
                )
                raise OSError(error_message)

        elif config_weights_source=="uniform":
            oelf_model_weights = pd.DataFrame(
                {
                    "model_id": list(range(1,number_models+1)),
                    "weight": [1.0/number_models for i in range(number_models)]
                }
            )

        else:
            error_message = (
                "Loader.load_OELF_weights() cannot recognise config_weights_source=%s. "
                "Possible values are 'from_file' and 'uniform'."
                % (config_weights_source)
            )
            raise OSError(error_message)

        if "model_name" not in oelf_model_weights.columns:
            oelf_model_weights["model_name"] = oelf_model_weights["model_id"].astype(str)

        oelf_model_weights.set_index(oelf_model_weights["model_id"], inplace=True)
        oelf_model_weights.drop(columns=["model_id"], inplace=True)

        return oelf_model_weights
