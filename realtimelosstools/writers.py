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
import fileinput
import getpass
from openquake.commonlib import datastore
from openquake.commonlib.logs import get_calc_ids
from openquake.commands.purge import purge_one


logger = logging.getLogger()


class Writer:
    """This class handles methods associated with writing/updating files (.csv, .xml, .ini,
    .txt).
    """

    @staticmethod
    def write_rupture_xml(
        out_filename,
        strike,
        dip,
        rake,
        magnitude,
        hypocenter,
        rupture_plane,
    ):
        """ This method creates the XML file of the rupture in OpenQuake format.

        Args:
            out_filename (str):
                Path and name of the rupture XML file to be written.
            strike (float):
                Strike of the rupture, in degrees, measured from north.
            dip (float):
                Dip of the rupture, in degrees, measured downwards from the horizontal.
            rake (float):
                Rake of the rupture, in degrees.
            magnitude (float):
                Magnitude of the earthquake.
            hypocenter (dict):
                Dictionary defining the coordinates of the hypocentre through the following keys
                and values:
                    lat (float):
                        Latitude of the hypocentre, in degrees.
                    lon (float):
                        Longitude of the hypocentre, in degrees.
                    depth (float):
                        Depth of the hypocentre, in km.
            rupture_plane (dict):
                Dictionary defining the coordinates of the rupture plane, with the following
                keys:
                    topLeft (dict):
                        Coordinates of the top left corner of the rupture.
                    topRight (dict):
                        Coordinates of the top right corner of the rupture.
                    bottomLeft (dict):
                        Coordinates of the bottom left corner of the rupture.
                    bottomRight (dict):
                        Coordinates of the bottom right corner of the rupture.
                Each of the four sub-dictionaries contains the following keys and values:
                    lat (float):
                        Latitude of the corner of the rupture, in degrees.
                    lon (float):
                        Longitude of the corner of the rupture, in degrees.
                    depth (float):
                        Depth of the corner of the rupture, in km.

        """

        f= open(out_filename, 'w')
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(
            '<nrml xmlns:gml="http://www.opengis.net/gml" '
            'xmlns="http://openquake.org/xmlns/nrml/0.4">\n'
        )
        f.write('    <singlePlaneRupture>\n')
        f.write('        <magnitude>%s</magnitude>\n' % ('{:.2f}'.format(magnitude)))
        f.write('        <rake>%s</rake>\n' % ('{:.1f}'.format(rake)))
        f.write(
            '        <hypocenter lat="%s" lon="%s" depth="%s"/>\n'
            % (
                '{:.5f}'.format(hypocenter["lat"]),
                '{:.5f}'.format(hypocenter["lon"]),
                '{:.5f}'.format(hypocenter["depth"])
            )
        )
        f.write(
            '        <planarSurface strike="%s" dip="%s">\n'
            % ('{:.0f}'.format(strike), '{:.0f}'.format(dip))
        )

        for corner in rupture_plane:  # topLeft, topRight, bottomLeft, bottomRight
            f.write(
                '            <%s lon="%s" lat="%s" depth="%s"/>\n'
                % (
                    corner,
                    '{:.5f}'.format(rupture_plane[corner]["lon"]),
                    '{:.5f}'.format(rupture_plane[corner]["lat"]),
                    '{:.5f}'.format(rupture_plane[corner]["depth"])
                )
            )

        f.write('        </planarSurface>\n')
        f.write('    </singlePlaneRupture>\n')
        f.write('</nrml>\n')

        f.close()

    @staticmethod
    def update_exposure_xml(filepath_exposure_xml, time_of_day, name_exposure_csv_file):
        """This method updates the name of the exposure CSV file as well as the time of the day
        of the earthquake in the exposure XML file. The exposure XML file
        'filepath_exposure_xml' needs to exist and contain data in the OpenQuake input format.

        Whether 'time_of_day' or 'name_exposure_csv_file' are appropriate strings is not
        verified by this method.

        Args:
            filepath_exposure_xml (str):
                Full path (directory and filename) to the exposure XML file in OpenQuake input
                format, which needs to exist.
            time_of_day (str):
                Time of day at which the earthquake occurs ("day", "night", "transit"), to be
                written in the exposure XML file.
            name_exposure_csv_file (str):
                Name of the exposure CSV file to be written in the XML file.
        """

        if not os.path.exists(filepath_exposure_xml):
            error_message = (
                "File '%s' not found. Method 'update_exposure_xml' cannot run."
                % (filepath_exposure_xml)
            )
            logger.critical(error_message)
            raise OSError(error_message)

        with fileinput.FileInput(filepath_exposure_xml, inplace=True) as file:
            for line in file:
                if "<assets>" in line:
                    text_to_search = line
                    replacement_text = "    <assets>%s</assets>\n" % (name_exposure_csv_file)
                    print(line.replace(text_to_search, replacement_text), end='')
                elif "<occupancyPeriods>" in line:
                    text_to_search = line
                    replacement_text = (
                        "    <occupancyPeriods>%s</occupancyPeriods>\n" % (time_of_day)
                    )
                    print(line.replace(text_to_search, replacement_text), end='')
                else:
                    text_to_search = line
                    print(line.replace(text_to_search, text_to_search), end='')

    @staticmethod
    def update_job_ini(
        filepath_job_ini, new_description, new_time_of_day, new_name_rupture_file
    ):
        """
        This method updates (1) the description, (2) the time of the day, and (3) the name of
        the rupture XML file in the job.ini file for OpenQuake, whose file path
        'filepath_job_ini' needs to exist and which needs to contain data in the OpenQuake input
        format.

        Args:
            filepath_job_ini (str):
                Full path (directory and filename) to the job.ini file in OpenQuake input
                format, which needs to exist.
            new_description (str):
                Description to be written to job.ini.
            new_time_of_day (str):
                Time of day ("day", "night", "transit") to be written to job.ini.
            new_name_rupture_file (str):
                Name of the rupture XML file to be written to job.ini.
        """

        if not os.path.exists(filepath_job_ini):
            error_message = (
                "File '%s' not found. Method 'update_job_ini' cannot run."
                % (filepath_job_ini)
            )
            logger.critical(error_message)
            raise OSError(error_message)

        with fileinput.FileInput(filepath_job_ini, inplace=True) as file:
            for line in file:
                if "description =" in line:
                    text_to_search = line
                    replacement_text = "description = %s\n" % (new_description)
                    print(line.replace(text_to_search, replacement_text), end='')
                elif "time_event =" in line:
                    text_to_search = line
                    replacement_text = "time_event = %s\n" % (new_time_of_day)
                    print(line.replace(text_to_search, replacement_text), end='')
                elif "rupture_model_file =" in line:
                    text_to_search = line
                    replacement_text = "rupture_model_file = %s\n" % (new_name_rupture_file)
                    print(line.replace(text_to_search, replacement_text), end='')
                else:
                    text_to_search = line
                    print(line.replace(text_to_search, text_to_search), end='')

    @staticmethod
    def delete_OpenQuake_last_job():
        """
        This method removes the last calculation ID from OpenQuake's database and its associated
        HDF5 file.
        """

        datadir = datastore.get_datadir()
        calc_id = get_calc_ids(datadir)
        purge_one(calc_id[-1], getpass.getuser(), True)

        return

    @staticmethod
    def write_txt_from_list(list_to_write, filepath):
        """
        This method writes the contents of 'list_to_write' to 'filepath'.

        Args:
            list_to_write (list of str): Content to be written.
            filepath (str): Full file to output file to be written.
        """

        f= open(filepath, "w")
        f.write("LOG FILE\n")
        for element in list_to_write:
            f.write(element+'\n')
        f.close()
