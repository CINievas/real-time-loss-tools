#!/usr/bin/env python3

# Copyright (C) 2022:
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

import logging
import sys


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def main():
    """Run the programme."""

    # Log the start of the run
    logger.info("Real-Time Loss Tools has started")

    # Leave the program
    logger.info("Real-Time Loss Tools has finished")
    sys.exit()


if __name__ == "__main__":
    main()
