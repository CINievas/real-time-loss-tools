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
from openquake.hazardlib import geo


logger = logging.getLogger()


class Rupture:
    """This class contains methods associated with defining/determining earthquake rupture
    properties.
    """

    @staticmethod
    def determine_local_time_from_utc(utc_time, timezone):
        """This method converts UTC time into local time.

        WARNING: THIS METHOD IS NOT IMPLEMENTED YET AND WILL RETURN THE INPUT UTC TIME.
        """

        return utc_time

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

    @staticmethod
    def define_rupture(event_id, source_params):
        """This method defines the strike, dip, rake, hypocenter and rupture plane of the rupture
        associated with earthquake with ID 'event_id'.

        WARNING: THE CURRENT IMPLEMENTATION OF THIS METHOD CALLS
        'build_rupture_from_ITACA_parameters', WHICH IS TAILORED TOWARDS RUPTURES DEFINED BY
        PARAMETERS RETRIEVED FROM THE ITACA DATABASE. IN THE FUTURE THIS SHOULD BECOME A GENERIC
        METHOD THAT DEALS WITH ANY KIND OF RUPTURE FROM THE SEISMICITY FORECAST.

        Args:
            event_id (str):
                Event ID within the ITACA catalogue for which the rupture will be built. It needs
                to exist as an index of 'source_params'.
            source_params (Pandas DataFrame):
                DataFrame containing parameters of earthquake sources retrieved from the ITalian
                ACcelerometric Archive (ITACA). See details in
                'build_rupture_from_ITACA_parameters'.

        Returns:
            strike (float):
                Strike of the rupture, in degrees, measured from north.
            dip (float):
                Dip of the rupture, in degrees, measured downwards from the horizontal.
            rake (float):
                Rake of the rupture, in degrees.
            hypocenter (dict):
                Dictionary defining the coordinates of the hypocentre through the following keys and
                values:
                    lat (float):
                        Latitude of the hypocentre, in degrees.
                    lon (float):
                        Longitude of the hypocentre, in degrees.
                    depth (float):
                        Depth of the hypocentre, in km.
            rupture_plane (dict):
                Dictionary defining the coordinates of the rupture plane, with the following keys:
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

        (
            strike,
            dip,
            rake,
            hypocenter,
            rupture_plane,
        ) = Rupture.build_rupture_from_ITACA_parameters(event_id, source_params)

        return strike, dip, rake, hypocenter, rupture_plane

    @staticmethod
    def distance_between_coordinates(lon1, lat1, lon2, lat2):
        """This method calculates the distance in km between two points defined by (lon1, lat1) and
        (lon2, lat2), both with an assumed depth of zero.

        Args:
            lon1 (float):
                Longitude of the first point, in degrees.
            lat1 (float):
                Latitude of the first point, in degrees.
            lon2 (float):
                Longitude of the second point, in degrees.
            lat2 (float):
                Latitude of the second point, in degrees.

        Returns:
            distance (float):
                Distance in km between the first and second point, both assumed to have zero depth.
        """

        distance = geo.geodetic.distance(lon1, lat1, 0.0, lon2, lat2, 0.0)

        return distance

    @staticmethod
    def calculate_depth_of_rupture_bottom(top_lon, top_lat, bottom_lon, bottom_lat, z_top, dip):
        """This method calculates the depth of the bottom coordinates of the rupture as a function
        of the depth to the top of the rupture ('z_top'), the dip of the fault and the geographic
        coordinates of bottom and upper points.

        WARNING: This calculation is very specific to the geometry of the faults in the L'Aquila
        case and would need revision in order to be generalised.

        Args:
            top_lon (float):
                Longitude of the point at the top of the rupture, in degrees.
            top_lat (float):
                Latitude of the point at the top of the rupture, in degrees.
            bottom_lon (float):
                Longitude of the point at the bottom of the rupture, in degrees.
            bottom_lat (float):
                Latitude of the point at the bottom of the rupture, in degrees.
            z_top (float):
                Depth to the top of the rupture, in km (i.e. depth of point defined by coordinates
                (top_lon, top_lat)).
            dip (float):
                Dip of the rupture, in degrees, measured downwards from the horizontal.

        Returns:
            bottom_depth (float):
                Depth to the bottom of the rupture, in km (i.e. depth of point defined by
                coordinates (bottom_lon, bottom_lat)).
        """

        horiz_dist = Rupture.distance_between_coordinates(
            top_lon, top_lat, bottom_lon, bottom_lat
        )
        vertical_diff = horiz_dist * np.tan(np.deg2rad(dip))
        bottom_depth = vertical_diff + z_top

        return bottom_depth

    @staticmethod
    def build_rupture_from_ITACA_parameters(event_id, source_params):
        """This method interprets the rupture parameters contained in 'source_params', reported as
        defined in the ITalian ACcelerometric Archive (ITACA), and returns a processed version of
        them for the earthquake with ID 'event_id'.

        Args:
            event_id (str):
                Event ID within the ITACA catalogue for which the rupture will be built. It needs
                to exist as an index of 'source_params'.
            source_params (Pandas DataFrame):
                DataFrame containing parameters of earthquake sources retrieved from the ITalian
                ACcelerometric Archive (ITACA), with the following columns (the names follow those
                used by ITACA):
                    event_id (str):
                        Event ID within the ITACA catalogue. This must also be the index of the
                        DataFrame.
                    Mw (float):
                        Moment magnitude of the event in the ITACA catalogue (preferred Mw).
                    nucleation_lon (float):
                        Longitude of the nucleation point (hypocentre), in degrees.
                    nucleation_lat (float):
                        Latitude of the nucleation point (hypocentre), in degrees.
                    nucleation_depth (float):
                        Depth of the nucleation point (hypocentre), in km.
                    LL_lon (float):
                        Longitude (in degrees) of the lower left corner of the rupture plane.
                    LL_lat (float):
                        Latitude (in degrees) of the lower left corner of the rupture plane.
                    UR_lon (float):
                        Longitude (in degrees) of the upper right corner of the rupture plane.
                    UR_lat (float):
                        Latitude (in degrees) of the upper right corner of the rupture plane.
                    LR_lon (float):
                        Longitude (in degrees) of the lower right corner of the rupture plane.
                    LR_lat (float):
                        Latitude (in degrees) of the lower right corner of the rupture plane.
                    UL_lon (float):
                        Longitude (in degrees) of the upper left corner of the rupture plane.
                    UL_lat (float):
                        Latitude (in degrees) of the upper left corner of the rupture plane.
                    Z_top (float):
                        Depth to the top of the rupture, in km.
                    Strike (float):
                        Strike of the rupture, in degrees, measured from north.
                    Dip (float):
                        Dip of the rupture, in degrees, measured downwards from the horizontal.
                    Rake (float):
                        Rake of the rupture, in degrees.

        Returns:
            strike (float):
                Strike of the rupture, in degrees, measured from north.
            dip (float):
                Dip of the rupture, in degrees, measured downwards from the horizontal.
            rake (float):
                Rake of the rupture, in degrees.
            hypocenter (dict):
                Dictionary defining the coordinates of the hypocentre through the following keys and
                values:
                    lat (float):
                        Latitude of the hypocentre, in degrees.
                    lon (float):
                        Longitude of the hypocentre, in degrees.
                    depth (float):
                        Depth of the hypocentre, in km.
            rupture_plane (dict):
                Dictionary defining the coordinates of the rupture plane, with the following keys:
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

        strike = source_params.loc[event_id, "Strike"]
        dip = source_params.loc[event_id, "Dip"]
        rake = source_params.loc[event_id, "Rake"]

        hypocenter = {}
        hypocenter["lat"] = source_params.loc[event_id, "nucleation_lat"]
        hypocenter["lon"] = source_params.loc[event_id, "nucleation_lon"]
        hypocenter["depth"] = source_params.loc[event_id, "nucleation_depth"]

        rupture_plane = {}

        rupture_plane["topLeft"] = {}
        rupture_plane["topLeft"]["lon"] = source_params.loc[event_id, "UL_lon"]
        rupture_plane["topLeft"]["lat"] = source_params.loc[event_id, "UL_lat"]
        rupture_plane["topLeft"]["depth"] = source_params.loc[event_id, "Z_top"]

        rupture_plane["topRight"] = {}
        rupture_plane["topRight"]["lon"] = source_params.loc[event_id, "UR_lon"]
        rupture_plane["topRight"]["lat"] = source_params.loc[event_id, "UR_lat"]
        rupture_plane["topRight"]["depth"] = source_params.loc[event_id, "Z_top"]

        rupture_plane["bottomLeft"] = {}
        rupture_plane["bottomLeft"]["lon"] = source_params.loc[event_id, "LL_lon"]
        rupture_plane["bottomLeft"]["lat"] = source_params.loc[event_id, "LL_lat"]

        rupture_plane["bottomRight"] = {}
        rupture_plane["bottomRight"]["lon"] = source_params.loc[event_id, "LR_lon"]
        rupture_plane["bottomRight"]["lat"] = source_params.loc[event_id, "LR_lat"]

        for side in ["Left", "Right"]:
            rupture_plane["bottom%s" % (side)][
                "depth"
            ] = Rupture.calculate_depth_of_rupture_bottom(
                rupture_plane["bottom%s" % (side)]["lon"],
                rupture_plane["bottom%s" % (side)]["lat"],
                rupture_plane["top%s" % (side)]["lon"],
                rupture_plane["top%s" % (side)]["lat"],
                rupture_plane["top%s" % (side)]["depth"],
                dip,
            )

        # The depth of the bottom corners needs to be the same
        average_bottom_depth = (
            rupture_plane["bottomLeft"]["depth"] + rupture_plane["bottomRight"]["depth"]
        ) / 2.0
        rupture_plane["bottomLeft"]["depth"] = average_bottom_depth
        rupture_plane["bottomRight"]["depth"] = average_bottom_depth

        return strike, dip, rake, hypocenter, rupture_plane
