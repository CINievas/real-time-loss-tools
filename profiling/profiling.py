#!/usr/bin/env python3

# Copyright (C) 2026:
#  GFZ Helmholtz-Zentrum für Geoforschung | GFZ Helmholtz Centre for Geosciences
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
import cProfile
import datetime
import pstats
from pstats import SortKey
from realtimelosstools.realtimelosstools import main
from realtimelosstools.configuration import Configuration


def profile_realtimelosstools():

    print("Profiling of Real-Time Loss Tools has begun")

    # Read configuration file
    config = Configuration("config.yml")

    # Create output profiling folder
    profiling_path = os.path.join(config.main_path, "output_profiling")
    if not os.path.exists(profiling_path):
        os.mkdir(profiling_path)

    # Output filename will be named as per the current running time
    now = datetime.datetime.now()    
    filename = "profiling_%s_%s_%s_%s_%s_%s.out" % (
        str(now.year),
        str(now.month).zfill(2),
        str(now.day).zfill(2),
        str(now.hour).zfill(2),
        str(now.minute).zfill(2),
        str(now.second).zfill(2),
    )
    output_filepath = os.path.join(profiling_path, filename)

    cProfile.run(
        'main()',
        output_filepath
    )

    print(
        "Profiling of Real-Time Loss Tools has finished. Results were output to %s\n"
        % (output_filepath)
    )

    p = pstats.Stats(output_filepath)
    print(
        "50 functions with longest cumulative time results (time spent including time spent in "
        "the functions/methods that it calls): \n"
    )
    p.sort_stats(SortKey.CUMULATIVE).print_stats(50)
    print(
        "50 functions with longest total time results (time spent excluding time spent in "
        "the functions/methods that it calls): \n"
    )
    p.sort_stats(SortKey.TIME).print_stats(50)

if __name__ == "__main__":
    profile_realtimelosstools()
