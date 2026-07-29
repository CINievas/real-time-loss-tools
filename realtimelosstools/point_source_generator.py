#!/usr/bin/env python3

# Copyright (C) 2026:
#   Cecilia Nievas: cecilia.nievas@gfz.de
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


"""
Create "point sources" from input seismicity catalogues.
As OpenQuake does not have point sources strictly speaking, "point source" is hereby understood
as the smallest conceivable planar rupture that is 10 m x 10 m (generated with PointMSR).
"""

import os
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from openquake.baselib.node import Node
from openquake.hazardlib import nrml
from openquake.hazardlib.scalerel import PointMSR
from openquake.hazardlib.geo import Point, PlanarSurface


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PointSourceSet():
    """
    Class to construct a set of point sources for a simple earthquake catalogue.
    """

    def __init__(
        self,
        default_depth: float, # km
        default_strike: float = 0.0,
        default_dip: float = 90.0,
        default_rake: float = 0.0
    ):
        """
        Args:
            default_depth (float):
                Default depth (km) to use if the earthquake catalogue does not contain a "depth"
                column.
            default_strike (float):
                Default strike (deg) to use if the earthquake catalogue does not contain a
                "strike" column. Default: 0.0.
            default_dip (float):
                Default dip (deg) to use if the earthquake catalogue does not contain a "dip"
                column. Default: 90.0.
            default_rake (float):
                Default rake (deg) to use if the earthquake catalogue does not contain a "rake"
                column. Default: 0.0.
        """
        self.default_depth = default_depth
        self.default_strike = default_strike
        self.default_dip = default_dip
        self.default_rake = default_rake

    def generate_ruptures(
        self, catalogue: pd.DataFrame, export_path: str = "", export_type: str = "xml"
    ):
        """
        Args:
            catalogue (Pandas DataFrame):
                Earthquake catalogue with (at least) the following fields:
                    longitude (float): Longitude of the hypocentre.
                    latitude (float): Latitude of the hypocentre.
                    magnitude (float): Moment magnitude.
                    model_id (int): ID of the seismicity model used to generate the stochastic
                        event set to which the earthquake belongs.
                    ses_id (int): ID of the stochastic event set that the earthquake belongs to.
                    event_id (str): Unique identifier of an earthquake within a Stochastic Event
                        Set.
                    depth (float): Depth of the hypocentre.
            export_path (str):
                Path to which the point sources will be exported.
            export_type (str):
                Type of file for export. Currently supported: "xml".

        Returns:
            ruptures: Dictionary of ruptures indexed by ID.
        """
        # Assign the catalogue to zones
        if catalogue.index.name != "EQID":
            # Set a unique ID for each event by joining the catalog ID and the event ID
            # (include model ID if it exists)
            if "model_id" in catalogue.columns:
                catalogue["EQID"] = pd.Series([
                    "{:g}-{:g}-{:g}".format(mod_id, cat_id, ev_id)
                    for (mod_id, cat_id, ev_id) in zip(
                        catalogue["model_id"], catalogue["ses_id"], catalogue["event_id"]
                    )
                ])
            else:
                catalogue["EQID"] = pd.Series([
                    "{:g}-{:g}".format(cat_id, ev_id)
                    for (cat_id, ev_id) in zip(catalogue["ses_id"], catalogue["event_id"])])
            catalogue.set_index("EQID", drop=True, inplace=True)

        catalogue_full = self.fill_in_missing_fields(catalogue)

        ruptures = {}
        for evid, event in catalogue_full.iterrows():
            rupture = build_rupture_from_series(event)
            ruptures[evid] = rupture

            if export_path:
                # If export folder is specified then export
                rupture_file = os.path.join(export_path, "RUP_%s.xml" % (evid))
                with open(rupture_file, "wb") as f:
                    nrml.write([rupture], f, fmt="%s")

        logger.info("Exported %g ruptures to %s" % (len(ruptures), export_path))

        return ruptures


    def fill_in_missing_fields(self, catalogue):
        """
        This method checks if 'catalogue' contains columns "depth", "strike", "dip" and "rake".
        If any of them is missing, it is created and filled in with the class default values. If
        the columns exist, the self.default values are ignored.

        Args:
            catalogue (Pandas DataFrame)
        """
        catalogue_full = deepcopy(catalogue)

        if "depth" not in catalogue_full.columns:
            catalogue_full["depth"] = (
                self.default_depth * np.ones(catalogue_full.shape[0], dtype=float)
            )
        if "strike" not in catalogue_full.columns:
            catalogue_full["strike"] = (
                self.default_strike * np.ones(catalogue_full.shape[0], dtype=float)
            )
        if "dip" not in catalogue_full.columns:
            catalogue_full["dip"] = (
                self.default_dip * np.ones(catalogue_full.shape[0], dtype=float)
            )
        if "rake" not in catalogue_full.columns:
            catalogue_full["rake"] = (
                self.default_rake * np.ones(catalogue_full.shape[0], dtype=float)
            )

        return catalogue_full


def build_rupture_from_series(rup_data: pd.Series) -> Node:
    """Builds a rupture with a PointMSR scaling relation from a Pandas Series with the event
    information.

    Args:
        rup_data (Pandas Series):
            Pandas Series containing the event information:
                longitude (float)
                latitude (float)
                depth (float)
                magnitude (float)
                strike (float)
                dip (float)
                rake (float)
    Returns:
        Planar rupture surface as Openquake.baselib.node.Node object.
    """
    sfc = PlanarSurface.from_hypocenter(
        Point(rup_data.longitude, rup_data.latitude, rup_data.depth),
        msr=PointMSR(),
        mag=rup_data.magnitude,
        aratio=1.0,
        strike=rup_data.strike,
        dip=rup_data.dip,
        rake=rup_data.rake
    )

    return Node(
        "singlePlaneRupture",
        nodes=[
            Node("magnitude", text=rup_data.magnitude),
            Node("rake", text=rup_data.rake),
            Node("hypocenter", {"lon": rup_data.longitude,
                                "lat": rup_data.latitude,
                                "depth": rup_data.depth}),
            sfc.surface_nodes[0]
        ]
    )
