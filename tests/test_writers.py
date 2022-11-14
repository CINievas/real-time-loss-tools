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
import shutil
import pytest
from realtimelosstools.writers import Writer


def test_write_rupture_xml():
    # Expected contents of the XML file
    expected_lines = {}
    expected_lines[0] = '<?xml version="1.0" encoding="utf-8"?>\n'
    expected_lines[1] = (
        '<nrml xmlns:gml="http://www.opengis.net/gml" '
        'xmlns="http://openquake.org/xmlns/nrml/0.4">\n'
    )
    expected_lines[2] = "    <singlePlaneRupture>\n"
    expected_lines[3] = "        <magnitude>5.10</magnitude>\n"
    expected_lines[4] = "        <rake>-90.0</rake>\n"
    expected_lines[5] = '        <hypocenter lat="42.31400" lon="13.41930" depth="8.20000"/>\n'
    expected_lines[6] = '        <planarSurface strike="140" dip="50">\n'
    expected_lines[7] = '            <topLeft lon="13.40000" lat="42.42100" depth="0.50000"/>\n'
    expected_lines[8] = (
        '            <topRight lon="13.55600" lat="42.28300" depth="0.50000"/>\n'
    )
    expected_lines[9] = (
        '            <bottomLeft lon="13.31000" lat="42.36600" depth="11.98250"/>\n'
    )
    expected_lines[10] = (
        '            <bottomRight lon="13.46600" lat="42.22700" depth="11.98250"/>\n'
    )
    expected_lines[11] = "        </planarSurface>\n"
    expected_lines[12] = "    </singlePlaneRupture>\n"
    expected_lines[13] = "</nrml>\n"

    # Create temporary output directory
    output_path = os.path.join(os.path.dirname(__file__), "data", "temp_test_writers")
    os.mkdir(output_path)

    # Parameters for Writer.write_rupture_xml()
    out_filename = os.path.join(output_path, "temp_rupture.xml")
    strike = 140.0
    dip = 50.0
    rake = -90.0
    magnitude = 5.1
    hypocenter = {"lon": 13.4193, "lat": 42.314, "depth": 8.2}
    rupture_plane = {
        "topLeft": {"lon": 13.4, "lat": 42.421, "depth": 0.5},
        "topRight": {"lon": 13.556, "lat": 42.283, "depth": 0.5},
        "bottomLeft": {"lon": 13.31, "lat": 42.366, "depth": 11.9825},
        "bottomRight": {"lon": 13.466, "lat": 42.227, "depth": 11.9825},
    }

    # Execute function
    Writer.write_rupture_xml(
        out_filename,
        strike,
        dip,
        rake,
        magnitude,
        hypocenter,
        rupture_plane,
    )

    # Check that the XML file has been created
    assert os.path.exists(out_filename)

    # Check line by line the contents of the XML file
    openfile = open(out_filename, "r")
    lines = openfile.readlines()

    for i, line in enumerate(lines):
        assert line == expected_lines[i]

    openfile.close()

    # Delete created output file and temporary output directory
    os.remove(out_filename)
    os.rmdir(output_path)

    
def test_update_exposure_xml():
    # Expected contents of the XML file
    expected_lines = {}
    expected_lines[0] = '<?xml version="1.0" encoding="UTF-8"?>\n'
    expected_lines[1] = (
        '<nrml xmlns="http://openquake.org/xmlns/nrml/0.4" '
        'xmlns:gml="http://www.opengis.net/gml">\n'
    )
    expected_lines[2] = (
        '  <exposureModel category="buildings" id="exposure" taxonomySource="GEM taxonomy">\n'
    )
    expected_lines[3] = '    <description>exposure model</description>\n'
    expected_lines[4] = '    <conversions>\n'
    expected_lines[5] = '      <costTypes>\n'
    expected_lines[6] = '        <costType name="structural" type="aggregated" unit="EUR"/>\n'
    expected_lines[7] = '      </costTypes>\n'
    expected_lines[8] = '    </conversions>\n'
    expected_lines[9] = '    <occupancyPeriods>night</occupancyPeriods>\n'
    expected_lines[10] = '    <tagNames>occupancy id_3 name_3 building_id</tagNames>\n'
    expected_lines[11] = '    <assets>newcsvfilename.csv</assets>\n'
    expected_lines[12] = '  </exposureModel>\n'
    expected_lines[13] = '</nrml>\n'

    # Create temporary output directory
    output_path = os.path.join(os.path.dirname(__file__), "data", "temp_test_writers")
    os.mkdir(output_path)

    # Copy existing exposure_model.xml to temporary output directory
    in_filename = os.path.join(
        os.path.join(os.path.dirname(__file__), "data", "exposure_model.xml")
    )  # origin
    out_filename = os.path.join(output_path, "exposure_model.xml")
    _ = shutil.copyfile(in_filename, out_filename)

    # Execute function
    Writer.update_exposure_xml(out_filename, "night", "newcsvfilename.csv")

    # Check line by line the contents of the XML file
    openfile = open(out_filename, "r")
    lines = openfile.readlines()

    for i, line in enumerate(lines):
        assert line == expected_lines[i]

    openfile.close()

    # Delete created output file and temporary output directory
    os.remove(out_filename)
    os.rmdir(output_path)
    
    # Test case in which the file path is not found
    # (file path is the same as above but it has been erased already)
    with pytest.raises(OSError) as excinfo:
        Writer.update_exposure_xml(out_filename, "day", "someothername.csv")
    assert "OSError" in str(excinfo.type)


def test_update_job_ini():
    # Expected contents of the job.ini file
    expected_lines = {}
    expected_lines[0] = '[general]\n'
    expected_lines[1] = 'description = A new description\n'
    expected_lines[2] = 'calculation_mode = scenario_damage\n'
    expected_lines[3] = 'ses_seed = 777\n'
    expected_lines[4] = '\n'
    expected_lines[5] = '[exposure]\n'
    expected_lines[6] = 'exposure_file = exposure_model.xml\n'
    expected_lines[7] = 'time_event = night\n'
    expected_lines[8] = '\n'
    expected_lines[9] = '[fragility]\n'
    expected_lines[10] = 'structural_fragility_file = fragility_model.xml\n'
    expected_lines[11] = '\n'
    expected_lines[12] = '[Rupture information]\n'
    expected_lines[13] = 'rupture_model_file = new_rupture_file.xml\n'
    expected_lines[14] = 'rupture_mesh_spacing = 0.5\n'
    expected_lines[15] = '\n'
    expected_lines[16] = '[Site conditions]\n'
    expected_lines[17] = 'site_model_file = site_model.csv\n'
    expected_lines[18] = '\n'
    expected_lines[19] = '[Calculation parameters]\n'
    expected_lines[20] = 'gsim_logic_tree_file = gmpe_logic_tree.xml\n'
    expected_lines[21] = 'truncation_level = 3\n'
    expected_lines[22] = 'maximum_distance = 200.0\n'
    expected_lines[23] = 'number_of_ground_motion_fields = 1000\n'
    expected_lines[24] = 'minimum_intensity = {"AvgSA": 1E-5}\n'

    # Create temporary output directory
    output_path = os.path.join(os.path.dirname(__file__), "data", "temp_test_writers")
    os.mkdir(output_path)

    # Copy existing job.ini to temporary output directory
    in_filename = os.path.join(
        os.path.join(os.path.dirname(__file__), "data", "job.ini")
    )  # origin
    out_filename = os.path.join(output_path, "job.ini")
    _ = shutil.copyfile(in_filename, out_filename)

    # Execute function
    Writer.update_job_ini(
        out_filename, "A new description", "night", "new_rupture_file.xml"
    )

    # Check line by line the contents of the job.ini file
    openfile = open(out_filename, "r")
    lines = openfile.readlines()

    for i, line in enumerate(lines):
        assert line == expected_lines[i]

    openfile.close()

    # Delete created output file and temporary output directory
    os.remove(out_filename)
    os.rmdir(output_path)

    # Test case in which the file path is not found
    # (file path is the same as above but it has been erased already)
    with pytest.raises(OSError) as excinfo:
        Writer.update_job_ini(
            out_filename, "A new description", "day", "another_rupture_file.xml"
        )
    assert "OSError" in str(excinfo.type)
