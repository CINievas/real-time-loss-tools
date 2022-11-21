#!/usr/bin/env python3

# Copyright (C) 2021:
#   Helmholtz-Zentrum Potsdam Deutsches GeoForschungsZentrum GFZ
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

import sys
import logging
import yaml
import pandas as pd

logger = logging.getLogger()


class Configuration:
    """This class handles the configuration parameters of the Real Time Loss Tools.

    Attributes:
        self.description_general (str):
            General description of the run to be used for the OpenQuake job.ini files.
        self.main_path (str):
            Path to the directory that contains the input files and where output files will be
            placed. It needs to have a pre-defined structure (please see the documentation).
        self.oelf_source_model_filename (str):
            Name of the XML file with the earthquake source model to be used to stochastically
            generate rupture properties for Operational Earthquake Loss Forecasting (OELF).
            Assumed to be located under main_path/ruptures.
        self.mapping_damage_states (Pandas DataFrame):
            Mapping between the names of damage states as output by OpenQuake and as labelled in
            the fragility model. In the yml configuration file it is defined by means of a
            dictionary that is converted into a Pandas DataFrame when read. In the dictionary,
            the keys are the names of damage states as output by OpenQuake and the values are
            the names of damage states as labelled in the fragility model. E.g.: {"no_damage":
            "DS0", "dmg_1": "DS1", ...}. In the DataFrame:
                Index:
                    asset_id (str): Names of damage states as output by OpenQuake.
                Columns:
                    fragility (str): Names of damage states as labelled in the fragility model.
        self.store_intermediate (bool):
            If True, intermediate results including updated exposure files after each earthquake
            and OpenQuake HDF5 files will be stored. If False, these intermediate results will
            not be available after running the software. True option is intended for debugging.
    """

    REQUIRES = [
        "description_general",
        "main_path",
        "oelf_source_model_filename",
        "mapping_damage_states",
        "store_intermediate",
    ]

    def __init__(self, filepath):
        """
        Args:
            filepath (str):
                Full file path to the .yml configuration file.
        """

        config = self.read_config_file(filepath)

        self.description_general = self.assign_parameter(config, "description_general")

        self.main_path = self.assign_parameter(config, "main_path")

        self.oelf_source_model_filename = self.assign_parameter(
            config, "oelf_source_model_filename"
        )

        mapping_damage_states_aux = self.assign_hierarchical_parameters(
            config, "mapping_damage_states"
        )
        self.mapping_damage_states = pd.DataFrame.from_dict(
            mapping_damage_states_aux, orient='index', columns=["fragility"]
        )
        self.mapping_damage_states.index = (
            self.mapping_damage_states.index.rename("asset_id")
        )

        self.store_intermediate = self.assign_boolean_parameter(config, "store_intermediate")

        # Terminate if critical parameters are missing (not all parameters are critical)
        for key_parameter in self.REQUIRES:
            if getattr(self, key_parameter) is None:
                error_message = (
                    "Error: parameter '%s' could not be retrieved from "
                    "configuration file. The program cannot run." % (key_parameter)
                )
                logger.critical(error_message)
                raise OSError(error_message)

    def read_config_file(self, filepath):
        """This function attempts to open the configuration file. If not found, it logs a
        critical error and raises an OSError.

        Args:
            filepath (str):
                Full file path to the .yml configuration file.

        Returns:
            config (dictionary):
                The configuration file read as a dictionary, or an empty dictionary if the
                configuration file was not found.
        """

        try:
            with open(filepath, "r") as ymlfile:
                config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config = {}
            error_message = "Error instantiating Configuration: configuration file not found"
            logger.critical(error_message)
            raise OSError(error_message)

        return config

    def assign_parameter(self, config, input_parameter):
        """This function searches for the key input_parameter in the dictionary config. If
        found, it returns its value (a string or a dictionary). If not found, it returns None.

        Args:
            config (dictionary):
                The configuration file read as a dictionary. It may be an empty dictionary.
            input_parameter (str):
                Name of the desired parameter, to be searched for as a primary key of config.
        Returns:
            assigned_parameter (str, dictionary or None):
                The content of config[input_parameter], which can be a string or a dictionary.
                It is None if input_parameter is not a key of config.
        """

        try:
            assigned_parameter = config[input_parameter]
        except KeyError:
            logger.warning(
                "Warning: parameter '%s' is missing from configuration file" % (input_parameter)
            )
            assigned_parameter = None

        return assigned_parameter

    def assign_hierarchical_parameters(self, config, input_parameter, requested_nested=[]):
        """This function searches for the key input_parameter in the dictionary config, and for
        each of the elements of requested_nested as keys of config[input_parameter].

        If input_parameter is not a key of config, the output is None.

        If input_parameter is a key of config, but one of the elements of requested_nested is
        not a key of config[input_parameter], the output is None.

        Args:
            config (dictionary):
                The configuration file read as a dictionary. It may be an empty dictionary.
            input_parameter (str):
                Name of the desired parameter, to be searched for as a primary key of config.
            requested_nested (list of str):
                List of the names of the desired nested parameters, to be searched for as keys
                of config[input_parameter]. If empty, the function will retrieve all nested
                parameters available in 'config'.

        Returns:
            assigned_parameter (dictionary or None):
                The content of config[input_parameter], if input_parameter is a key of config
                and all elements of requested_nested are keys of config[input_parameter], or
                None otherwise.
        """

        assigned_parameter = self.assign_parameter(config, input_parameter)

        if assigned_parameter is None:
            return None

        if not isinstance(assigned_parameter, dict):
            return None

        if len(requested_nested) == 0:
            requested_nested = list(assigned_parameter.keys())

        sub_parameters_missing = False
        for requested_parameter in requested_nested:
            if requested_parameter not in assigned_parameter.keys():
                logger.critical(
                    "ERROR instantiating Configuration: parameter '%s' does not "
                    "exist in %s" % (requested_parameter, input_parameter)
                )
                sub_parameters_missing = True

        if sub_parameters_missing is True:
            return None

        return assigned_parameter

    def assign_boolean_parameter(self, config, input_parameter):
        """This function searches for the key input_parameter in the dictionary config, and
        converts it into a boolean.

        If input_parameter is not a key of config, the output is None.

        Args:
            config (dictionary):
                The configuration file read as a dictionary. It may be an empty dictionary.
            input_parameter (str):
                Name of the desired parameter, to be searched for as a primary key of config.

        Returns:
            assigned_parameter (bool):
                The content of config[input_parameter] converted into a boolean.

        """

        assigned_parameter = self.assign_parameter(config, input_parameter)

        if assigned_parameter is None:
            return None

        if not isinstance(assigned_parameter, bool):  # yaml tries to interpret data types
            if isinstance(assigned_parameter, str):
                if assigned_parameter.lower() in ["true", "yes"]:
                    assigned_parameter = True
                elif assigned_parameter.lower() in ["false", "no"]:
                    assigned_parameter = False
                else:
                    logger.critical(
                        "Error reading %s from configuration file: "
                        "string '%s' cannot be interpreted as boolean"
                        % (input_parameter, assigned_parameter)
                    )
                    assigned_parameter = None
            else:
                logger.critical(
                    "Error reading %s from configuration file: not a boolean"
                    % (input_parameter)
                )
                assigned_parameter = None

        return assigned_parameter
