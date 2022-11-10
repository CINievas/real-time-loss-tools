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


"""
Suite of function and class to create synthetic ruptures for a minimal earthquake catalogue
using information provided by a source model
"""

import os
import math
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
from pyproj import Transformer
from openquake.baselib.node import Node
from openquake.hazardlib import nrml, mfd
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.geo import geodetic, Point, Polygon, PlanarSurface, Mesh, NodalPlane
from openquake.hazardlib.scalerel import BaseMSR, Leonard2014_Interplate


logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Set a default transformer for Europe
EUROPE_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def build_rupture(usd, lsd, mag, dims, strike, dip, rake, clon, clat, cdep):
    """
    Function to build the rupture from the available dimension and orientation
    information.

    Copied from the OpenQuake-engine codebase
    """
    # from the rupture center we can now compute the coordinates of the
    # four coorners by moving along the diagonals of the plane. This seems
    # to be better then moving along the perimeter, because in this case
    # errors are accumulated that induce distorsions in the shape with
    # consequent raise of exceptions when creating PlanarSurface objects
    # theta is the angle between the diagonal of the surface projection
    # and the line passing through the rupture center and parallel to the
    # top and bottom edges. Theta is zero for vertical ruptures (because
    # rup_proj_width is zero)
    half_length, half_width, half_height = dims / 2.
    rdip = math.radians(dip)

    # precalculated azimuth values for horizontal-only and vertical-only
    # moves from one point to another on the plane defined by strike
    # and dip:
    azimuth_right = strike
    azimuth_down = azimuth_right + 90
    azimuth_left = azimuth_down + 90
    azimuth_up = azimuth_left + 90

    # half height of the vertical component of rupture width
    # is the vertical distance between the rupture geometrical
    # center and it's upper and lower borders:
    # calculate how much shallower the upper border of the rupture
    # is than the upper seismogenic depth:
    vshift = usd - cdep + half_height
    # if it is shallower (vshift > 0) than we need to move the rupture
    # by that value vertically.
    if vshift < 0:
        # the top edge is below upper seismogenic depth. now we need
        # to check that we do not cross the lower border.
        vshift = lsd - cdep - half_height
        if vshift > 0:
            # the bottom edge of the rupture is above the lower seismo
            # depth; that means that we don't need to move the rupture
            # as it fits inside seismogenic layer.
            vshift = 0
        # if vshift < 0 than we need to move the rupture up.

    # now we need to find the position of rupture's geometrical center.
    # in any case the hypocenter point must lie on the surface, however
    # the rupture center might be off (below or above) along the dip.
    if vshift != 0:
        # we need to move the rupture center to make the rupture fit
        # inside the seismogenic layer.
        hshift = abs(vshift / math.tan(rdip))
        clon, clat = geodetic.point_at(
            clon, clat, azimuth_up if vshift < 0 else azimuth_down,
            hshift)
        cdep += vshift
    theta = math.degrees(math.atan(half_width / half_length))
    hor_dist = math.sqrt(half_length ** 2 + half_width ** 2)
    top_left = np.array([0., 0., cdep - half_height])
    top_right = np.array([0., 0., cdep - half_height])
    bottom_left = np.array([0., 0., cdep + half_height])
    bottom_right = np.array([0., 0., cdep + half_height])
    top_left[:2] = geodetic.point_at(
        clon, clat, strike + 180 + theta, hor_dist)
    top_right[:2] = geodetic.point_at(
        clon, clat, strike - theta, hor_dist)
    bottom_left[:2] = geodetic.point_at(
        clon, clat, strike + 180 - theta, hor_dist)
    bottom_right[:2] = geodetic.point_at(
        clon, clat, strike + theta, hor_dist)
    return top_left, top_right, bottom_left, bottom_right


def get_rupdims(area: float, dip: float, width: float, rar: float) -> np.ndarray:
    """
    Calculate and return the rupture length and width
    for given magnitude surface parameters.

    Args:
        area: Rupture area (km ^ 2)
        dip: Rupture dip (degrees from horizontal)
        width: Rupture width (km)
        rar: Rupture aspect ratio
    Returns:
        array of shape (1, 3) with rupture lengths, widths and heights

    The rupture area is calculated using method
    :meth:`~openquake.hazardlib.scalerel.base.BaseMSR.get_median_area`
    of source's
    magnitude-scaling relationship. In any case the returned
    dimensions multiplication is equal to that value. Than
    the area is decomposed to length and width with respect
    to source's rupture aspect ratio.
    If calculated rupture width being inclined by nodal plane's
    dip angle would not fit in between upper and lower seismogenic
    depth, the rupture width is shrunken to a maximum possible
    and rupture length is extended to preserve the same area.
    """
    out = np.zeros(3)
    rup_length = np.sqrt(area * rar)
    rup_width = area / rup_length
    rdip = math.radians(dip)
    max_width = width / math.sin(rdip)
    if rup_width > max_width:
        rup_width = max_width
        rup_length = area / rup_width
    out[0] = rup_length
    out[1] = rup_width * math.cos(rdip)
    out[2] = rup_width * math.sin(rdip)
    return out


def planar_rupture_to_xml(rup: Dict, filename: str):
    """
    Exports the planar rupture from the dictionary to an xml file

    Args:
        rup: Rupture as a dictionary
        filename: File to be written
    """
    # Generate surface nodes
    sfc_nodes = []
    for key in ["topLeft", "topRight", "bottomRight", "bottomLeft"]:
        sfc_nodes.append(
            Node(key,
                 dict([(crd, rup[key][crd]) for crd in ["lon", "lat", "depth"]]))
        )
    sfc_nodes = Node("planarSurface", nodes=sfc_nodes)
    rupture_model = Node(
        "singlePlaneRupture",
        attrib={"time": rup["time"]},
        nodes=[
            Node("magnitude", text=rup["magnitude"]),
            Node("rake", text=rup["rake"]),
            Node("hypocenter", attrib=rup["hypocenter"]),
            sfc_nodes
        ]
    )
    with open(filename, "wb") as f:
        nrml.write([rupture_model], f, fmt="%s")
    return


def ruptures_to_geodataframe(ruptures: Dict) -> gpd.GeoDataFrame:
    """
    Converts a set of ruptures from a rupture-ID indexed set of dictionaries
    to a geopandas GeoDataFrame

    Args:
        ruptures: Dictionary of ruptures (each as a dictionary) indexed by
                  rupture ID
    Returns:
        Rupture set as a geodataframe
    """
    # Build geometries
    nrup = len(ruptures)
    rup_dframe = dict([
        (key, [None] * nrup)
        for key in ["ID", "M", "RAKE", "HYPO_LON", "HYPO_LAT", "HYPO_DEPTH", "geometry"]
        ])
    for i, (i_d, rup) in enumerate(ruptures.items()):
        vertices = []
        for key in ["topLeft", "topRight", "bottomRight", "bottomLeft", "topLeft"]:
            vertices.append([rup[key]["lon"], rup[key]["lat"], rup[key]["depth"]])
        rup_dframe["ID"][i] = i_d
        rup_dframe["M"][i] = rup["magnitude"]
        rup_dframe["RAKE"][i] = rup["rake"]
        rup_dframe["HYPO_LON"][i] = rup["hypocenter"]["lon"]
        rup_dframe["HYPO_LAT"][i] = rup["hypocenter"]["lat"]
        rup_dframe["HYPO_DEPTH"][i] = rup["hypocenter"]["depth"]
        rup_dframe["geometry"][i] = geometry.Polygon(vertices)
    rup_dframe["geometry"] = gpd.GeoSeries(rup_dframe["geometry"])
    return gpd.GeoDataFrame(rup_dframe, crs="EPSG:4326")


def export_ruptures_to_xml(ruptures: Dict, export_folder: str):
    """
    If ruptures have been generated then these are exported to the
    folder specified

    Args:
        ruptures: Dictionary of ruptures (each as a dictionary) indexed by
                  rupture ID
        export_folder: Path to folder for export (will be generated if
                       the folder doesn't exist
    """
    nrups = len(list(ruptures))
    if not os.path.exists(export_folder):
        os.mkdir(export_folder)
    for event_id, rup in ruptures.items():
        fname = os.path.join(export_folder, "RUP_{:s}.xml".format(event_id))
        planar_rupture_to_xml(rup, fname)
    print("Exported %g ruptures to %s" % (nrups, export_folder))
    return


def export_ruptures_to_shp(ruptures: Dict, export_folder: str):
    """
    If ruptures have been generated then these are exported to the
    folder specified

    Args:
        ruptures: Dictionary of ruptures (each as a dictionary) indexed by
                  rupture ID
        export_folder: Path to folder for export (will be generated if
                       the folder doesn't exist
    """
    rup_dframe = ruptures_to_geodataframe(ruptures)
    rup_dframe.to_file(export_folder, index=False)
    print("Exported %g ruptures to %s" % (len(ruptures), export_folder))
    return


RUPTURE_SET_EXPORTER = {
    "shp": export_ruptures_to_shp,
    "xml": export_ruptures_to_xml,
}


NULL_SOURCE = pd.Series(
    dict([(key, pd.NA) for key in ["SRC_ID", "NAME", "TRT", "USD", "LSD", "RATE", "geometry"]])
)


class StochasticRuptureSet():
    """
    Class to construct a set of finite ruptures for a simple earthquake catalogue
    using the information provided by a relevant seismogenic source model.

    Attributes:
        source_model: Seismic source model containing the ID, name, upper seismogenic depth,
                      lower seismogenic depth, total rate and geometry
        pmfs: Dictionary of probability mass functions for hypocentral depth and nodal planes
              indexed by source ID
        msr: Magnitude scaling relation
        aspect_limits: Range of aspect ratios for sampling
        default_usd: Default upper seismogenic depth for regions outside the source model
        default_lsd: Default lower seismogenic depth for regions outside the source model
    """
    # Default nodal plane and hypocentral depth distributions for the case
    # that the epicentre is outside any zone.
    # For Italy this occurs only when the event is offshore
    DEFAULT_PMF = {
        "hdd": PMF([(0.25, 6.0), (0.5, 10.0), (0.25, 14.0)]),
        "npd": PMF([(0.25, NodalPlane(strike=0.0, dip=90.0, rake=0.0)),
                    (0.25, NodalPlane(strike=45.0, dip=90.0, rake=0.0)),
                    (0.25, NodalPlane(strike=90.0, dip=90.0, rake=0.0)),
                    (0.25, NodalPlane(strike=135.0, dip=90.0, rake=0.0))])
    }

    def __init__(self, source_model: gpd.GeoDataFrame,
                 pmfs: Dict, msr: Optional[BaseMSR] = Leonard2014_Interplate,
                 aspect_limits: Tuple = (1.0, 1.5),
                 default_usd: float = 0.0, default_lsd: float = 25.0):
        self.source_model = source_model
        self.pmfs = pmfs
        self.msr = msr()
        self.aspect_limits = aspect_limits
        self.default_lsd = default_lsd
        self.default_usd = default_usd
        self._source_model_xy = None

    @property
    def source_model_xy(self):
        """
        Creates a version of the source model translated into a Cartesian coordinate system
        """
        if self._source_model_xy is not None:
            return self._source_model_xy
        self._source_model_xy = self.source_model.to_crs("EPSG:3035")
        return self._source_model_xy

    @classmethod
    def from_xml(cls, asm_source_file: str, mmin: float, trts: Optional[List] = None,
                 msr: Optional[BaseMSR] = Leonard2014_Interplate,
                 aspect_limits: Tuple = (1.0, 1.5), default_usd: float = 0.0,
                 default_lsd: float = 25.0, strip_string: str = "",):
        """
        Instantiates the class from an OpenQuake area source file using the distributions
        contained for each source.

        Args:
            asm_source_file: Path to the area source model file
            mmin: Minimum magnitudes (for calculating total rates)
            trts: List of tectonic region types to be considered (will discard those sources
                  whose tectonic region types are not in the list)
        """
        if trts is None:
            trts = []
        source_pmfs = {}
        source_dframe = {
            "SRC_ID": [],
            "NAME": [],
            "TRT": [],
            "USD":  [],
            "LSD": [],
            "RATE": [],
            "geometry": [],
        }
        source_model_raw = list(nrml.read(asm_source_file))[0]
        for src in source_model_raw:
            if "areaSource" not in src.tag:
                logging.info("Source %s not area source - skipping" % src["id"])
                continue
            if len(trts) and (src["tectonicRegion"] not in trts):
                # TRT filter in place as TRT not in list
                logging.info("Source %s TRT %s not among those listed"
                             % (src["id"], src["tectonicRegion"]))
                continue
            # Should be an area source and a relevant TRT
            src_id, src_name = src["id"], src["name"]
            if strip_string:
                src_id = src_id.replace(strip_string, "")
                src_name = src_name.replace(strip_string, "")
            logging.info("SOURCE ID: %s (Name: %s)" % (src_id, src_name))
            source_dframe["SRC_ID"].append(src_id)
            source_dframe["NAME"].append(src_name)
            source_dframe["TRT"].append(src["tectonicRegion"])
            # Parse the depth distribution
            hdd = []
            for subnode in src.hypoDepthDist:
                hdd.append((subnode["probability"], subnode["depth"]))
            hdd = PMF(hdd)
            # Parse the nodal plane distribution
            npd = []
            for subnode in src.nodalPlaneDist:
                npl = NodalPlane(subnode["strike"], subnode["dip"], subnode["rake"])
                npd.append((subnode["probability"], npl))
            npd = PMF(npd)
            source_pmfs[src_id] = {"hdd": hdd, "npd": npd}
            # Parse the area geometries
            source_dframe["USD"].append(src.areaGeometry.upperSeismoDepth.text)
            source_dframe["LSD"].append(src.areaGeometry.lowerSeismoDepth.text)
            raw_crds = src.areaGeometry.Polygon.exterior.LinearRing.posList.text
            source_dframe["geometry"].append(
                geometry.Polygon(
                    [(raw_crds[i], raw_crds[i + 1])
                     for i in range(0, len(raw_crds), 2)]
                    )
                )
            total_rate = 1.0
            for node in src:
                if node.tag.split("}")[1] in ["incrementalMFD", "truncatedGutenbergRichterMFD"]:
                    total_rate = cls.get_rate_mmin(node, mmin)
            source_dframe["RATE"].append(total_rate)
        source_dframe["geometry"] = gpd.GeoSeries(source_dframe["geometry"],
                                                  index=source_dframe['SRC_ID'])
        source_dframe = gpd.GeoDataFrame(source_dframe,
                                         geometry="geometry",
                                         crs="EPSG:4326",
                                         index=source_dframe['SRC_ID'])

        return cls(source_dframe, source_pmfs, msr, aspect_limits,
                   default_usd, default_lsd)

    @staticmethod
    def get_rate_mmin(mfd_node: Node, mthresh: float) -> float:
        """
        Get the rate of seismicity above a minimum magnitude

        Args:
            mfd_node: Node of the source model corresponding to the MFD
            mthresh: Threshold magnitude
        Returns:
            Total rate above mthresh
        """
        if "incrementalMFD" in mfd_node.tag:
            mmin = mfd_node["minMag"]
            bin_width = mfd_node["binWidth"]
            rates = mfd_node.occurRates.text
            if mmin < mthresh:
                mfd_src = mfd.EvenlyDiscretizedMFD(mmin, bin_width, rates)
                mags, rates = map(np.array, zip(*mfd_src.get_annual_occurrence_rates()))
                rates = rates[mags >= mthresh]
        elif "truncGutenbergRichterMFD" in mfd_node.tag:
            aval = mfd_node["aValue"]
            bval = mfd_node["bValue"]
            mmin = mfd_node["minMag"]
            mmax = mfd_node["maxMag"]
            bin_width = mfd_node["binWidth"]
            if mmax < mthresh:
                rates = [0]
            else:
                mfd_src = mfd.TruncatedGRMFD(mmin, mmax, bin_width, aval, bval)
                mags, rates = map(np.array, zip(*mfd_src.get_annual_occurrence_rates()))
                rates = rates[mags >= mthresh]
        else:
            raise ValueError("MFD %s not yet supported" % mfd_node.tag)
        return sum(rates)

    def find_source_for_event(self, longitude, latitude, depth: Optional[float] = None,
                              transformer: Transformer = EUROPE_TRANSFORMER) ->\
            Union[str, None]:
        """
        For a single event with location given by longitude and latitude this
        determines the corresponding source zone. Note that this is a single
        event function and that for full catalogue association the full geopandas
        approach is more efficient.

        Args:
            longitude: Epicentre longitude (degrees east)
            latitude: Epicentre latitude (degrees north)
            depth: Hypocentral depth (km)
            transformer: Pyproj Transformer object for transforming the coordinates to
                         cartesian
        Returns:
            ID of source
        """
        src_ids = []
        point_xy = geometry.Point(*transformer.transform(longitude, latitude))
        for src_id, src in self.source_model_xy.iterrows():
            if src.geometry.contains(point_xy):
                src_ids.append(src.SRC_ID)
        if not len(src_ids):
            # Event not in any source
            return None
        elif len(src_ids) == 1:
            # Unique source found
            return src_ids[0]
        else:
            pass
        # Multiple sources found
        if depth is not None:
            # If the depth is constrained then use that to distinguish
            for src_id in src_ids:
                usd = self.source_model.USD.loc[src_id]
                lsd = self.source_model.LSD.loc[src_id]
                if (depth >= usd) and (depth <= lsd):
                    return src_id
            # Event is outside the depth range of any source, so return None
            return None
        else:
            # No information to specify the source, so sample randomly weighted
            # by the total rate
            src_mod = self.source_model[self.source_model.SRC_ID.isin(src_ids)]
            sample = src_mod.SRC_ID.sample(n=1,
                                           weights=src_mod.RATE / src_mod.RATE.sum(),
                                           replace=True)
            return sample.iloc[0]

    def add_source_info_for_event(self, event: pd.Series) -> pd.Series:
        """
        For a event represented by a pandas DataSeries containing attributes 'longitude' and
        'latitude' this function returns the

        """
        if ('depth' in event.index) and not pd.isna(event["depth"]):
            depth = event["depth"]
        else:
            depth = None
        # Get the source information for the location
        src_id = self.find_source_for_event(event.longitude, event.latitude, depth)
        # Return a pd.Series concatenating the event information and the source information
        if src_id:
            return pd.concat([event, self.source_model.loc[src_id]])
        else:
            if "EQID" in event.index:
                ev_id = event["EQID"]
            else:
                ev_id = ""
            logger.info("Event %s (%.4fE,%.4fN) not in any zone"
                        % (ev_id, event.longitude, event.latitude))
            # Event doesn't occur within any source - return a set of null values
            return pd.concat([event, NULL_SOURCE])

    def generate_ruptures(self, catalogue: pd.DataFrame,
                          export_file: str = "", export_type: str = "xml",
                          event_id_stem: str = "EQ"):
        """
        For a input earthquake catalogue and builds a corresponding set of ruptures according
        to the distribution of depths and ruptures implied by the source model. These can be
        optionally exported to file (xml or shp).

        Args:
            catalogue: Earthquake catalogue containing fields ["longitude", "latitude",
                       "magnitude", "time_string", "catalog_id", "event_id"]
            export_file: File to export the ruptures
            export_type: Type of file for export (from ['xml', 'shp'])
            event_id_stem: Common step for all event IDs

        Returns:
            ruptures: Dictionary of ruptures indexed by ID
        """
        # Assign the catalogue to zones
        catalogue_geom = gpd.points_from_xy(catalogue["longitude"], catalogue["latitude"])
        catalogue = gpd.GeoDataFrame(catalogue, geometry=catalogue_geom, crs='EPSG:4326')
        # Set a unique ID for each event by joining the catalog ID and the event ID
        catalogue["EQID"] = pd.Series([
            "{:g}-{:g}".format(cat_id, ev_id)
            for (cat_id, ev_id) in zip(catalogue["catalog_id"], catalogue["event_id"])])
        catalogue.set_index("EQID", drop=True, inplace=True)
        catalogue_cart = catalogue.to_crs("EPSG:3035")
        catalogue_cart = gpd.sjoin(catalogue_cart,
                                   self.source_model.to_crs("EPSG:3035"),
                                   how="left")
        # Build the rupture planes
        ruptures = self.catalogue_to_planes(catalogue_cart)
        if export_file:
            # If export folder is specified then export according to the specified filetype
            RUPTURE_SET_EXPORTER[export_type](ruptures, export_file)
        return ruptures

    def event_to_plane(self, event: pd.Series, aspect_ratio: float = 1.25) -> Dict:
        """
        Builds the rupture for a given event

        Args:
            event: The event as a row of a pandas Series (or any other class with
                   attributes longitude, latitude, magnitude, time_string, USD, LSD)
            aspect_ratio: Rupture aspect ratio (length / width) for the event (note that this
                          will be re-scaled if the rupture width exceeds the available
                          seismogenic thickness
        Returns:
            Set of rupture attributes
        """
        # Take single sample of the HDD and NPD
        if ("SRC_ID" in event.index) and (event.SRC_ID in list(self.pmfs)):
            hypo_depth = self.pmfs[event.SRC_ID]["hdd"].sample(1)[0]
            nodal_plane = self.pmfs[event.SRC_ID]["npd"].sample(1)[0]
            usd = event.USD if event.USD else self.default_usd
            lsd = event.LSD if event.LSD else self.default_lsd
        else:
            if "EQID" in event.index:
                ev_id = event["EQID"]
            else:
                ev_id = ""
            logger.info("Event %s (%.4fE,%.4fN) not in any zone"
                        % (ev_id, event.longitude, event.latitude))
            hypo_depth = self.DEFAULT_PMF["hdd"].sample(1)[0]
            nodal_plane = self.DEFAULT_PMF["npd"].sample(1)[0]
            usd = self.default_usd
            lsd = self.default_lsd
        # Get the area
        area = self.msr.get_median_area(event.magnitude, nodal_plane.rake)
        # Get the rupture dimensions for the given area, magnitude and
        # configuration
        rupture_dims = get_rupdims(area,
                                   nodal_plane.dip,
                                   lsd - usd,
                                   aspect_ratio)
        top_left, top_right, bottom_left, bottom_right = build_rupture(
            usd, lsd, event.magnitude, rupture_dims, nodal_plane.strike,
            nodal_plane.dip, nodal_plane.rake, event.longitude,
            event.latitude, hypo_depth
        )
        return {
            "time": event.time_string,
            "magnitude": event.magnitude,
            "rake": nodal_plane.rake,
            "hypocenter": {"lon": event.longitude,
                           "lat": event.latitude,
                           "depth": hypo_depth},
            "topLeft": {"lon": top_left[0],
                        "lat": top_left[1],
                        "depth": top_left[2]},
            "topRight": {"lon": top_right[0],
                         "lat": top_right[1],
                         "depth": top_right[2]},
            "bottomLeft": {"lon": bottom_left[0],
                           "lat": bottom_left[1],
                           "depth": bottom_left[2]},
            "bottomRight": {"lon": bottom_right[0],
                            "lat": bottom_right[1],
                            "depth": bottom_right[2]}
        }

    def catalogue_to_planes(self, catalogue: pd.DataFrame) -> Dict:
        """
        Generates the rupture planes for each event in the catalogue

        Args:
            catalogue: Earthquake catalogue containing fields ["longitude", "latitude",
                       "magnitude", "time_string"]

        Returns:
            ruptures: Dictionary of ruptures indexed by ID
        """
        ruptures = {}
        nevents = catalogue.shape[0]
        event_count = catalogue.index.value_counts(sort=False)
        aspect_ratios = np.random.uniform(*self.aspect_limits, size=nevents)
        for i, eq_id in enumerate(event_count.index):
            if event_count.loc[eq_id] > 1:
                # Event is attributed to multiple sources - select by rate
                subcat = catalogue.loc[eq_id]
                probs = subcat["RATE"].to_numpy() / subcat["RATE"].sum()
                mfd_pmf = PMF([(probs[j], str(subcat["SRC_ID"].iloc[j]))
                              for j in range(subcat.shape[0])])
                sample_id = mfd_pmf.sample(1)[0]
                event = subcat[subcat["SRC_ID"] == sample_id].iloc[0]
            else:
                event = catalogue.loc[eq_id]
            ruptures[eq_id] = self.event_to_plane(event, aspect_ratios[i])
        return ruptures
