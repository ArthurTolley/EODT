# app/data_prep/osm_utils.py
import os
import xml.etree.ElementTree as ET
from subprocess import Popen
from time import sleep
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from src.road_det_data_gen.data_prep.geo_utils import graphInsert, graphDensify, graph2RegionCoordinate, graphGroundTruthPreProcess, graphVis2048Segmentation
import pickle
import json
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OSMLoader:
    """
    Loads and parses OpenStreetMap (OSM) data for a specified geographic region.

    Downloads OSM data if not provided, parses nodes and ways, and builds
    a graph representation including edges and properties. Supports filtering
    underground roads and inclusion of service roads.

    Attributes:
        nodedict: Dict mapping node IDs to node info including lat, lon, and connectivity.
        waydict: Dict mapping way IDs to lists of node references.
        roadlist: List of roads (unused here, reserved for extensions).
        roaddict: Dictionary of roads (unused here, reserved for extensions).
        edge2edgeid: Dict mapping edge tuples (node1, node2) to unique edge IDs.
        edgeid2edge: Dict mapping edge IDs back to node tuples.
        edgeProperty: Dict mapping edge IDs to properties like width, lane count, layer, etc.
        edgeId: Counter for assigning new edge IDs.
        buildings: List of building polygons and heights.
        way_c: Counter for assigning way IDs.
        minlat, maxlat, minlon, maxlon: Bounding box coordinates from OSM data.
    """

    def __init__(
        self,
        region: list[float],
        noUnderground: bool = False,
        osmfile: str | None = None,
        includeServiceRoad: bool = False
    ) -> None:
        """
        Initialize the OSMLoader, downloading OSM data if needed, and parsing its contents.

        Args:
            region: [lat_start, lat_end, lon_start, lon_end] bounding box coordinates.
            noUnderground: If True, exclude roads with layer < 0.
            osmfile: Optional path to a local OSM file. If None, downloads OSM data.
            includeServiceRoad: If True, includes service roads in the graph.
        """
        bbox = f"{region[2]},{region[0]},{region[3]},{region[1]}"  # OSM uses lon,lat order
        os.makedirs("tmp", exist_ok=True)
        filename = osmfile or self._download_osm_file(bbox)

        self._init_state()

        road_blacklist = {
            'None', 'pedestrian', 'footway', 'bridleway', 'steps', 'path',
            'sidewalk', 'cycleway', 'proposed', 'construction', 'bus_stop',
            'crossing', 'elevator', 'emergency_access_point', 'escape', 'give_way'
        }

        mapxml = ET.parse(filename).getroot()
        self._parse_bounds(mapxml)
        self._parse_nodes(mapxml)

        for way in mapxml.findall('way'):
            self._parse_way(way, road_blacklist, noUnderground, includeServiceRoad)

    def _download_osm_file(self, bbox: str) -> str:
        """
        Download OSM data for the given bounding box using Overpass API.

        Retries every 60 seconds if download fails.

        Args:
            bbox: Bounding box string "minlon,minlat,maxlon,maxlat".

        Returns:
            Path to the downloaded OSM file.
        """
        path = f"tmp/map?bbox={bbox}"
        while not os.path.exists(path):
            Popen(f"wget http://overpass-api.de/api/map?bbox={bbox}", shell=True).wait()
            Popen(f"mv 'map?bbox={bbox}' tmp/", shell=True).wait()
            if not os.path.exists(path):
                logging.warning("Download failed. Retrying in 60 seconds...")
                sleep(60)
        return path

    def _init_state(self) -> None:
        """Initialize all internal data structures."""
        self.nodedict: dict[str, dict] = {}
        self.waydict: dict[int, list[str]] = {}
        self.roadlist: list = []
        self.roaddict: dict = {}
        self.edge2edgeid: dict[tuple[str, str], int] = {}
        self.edgeid2edge: dict[int, tuple[str, str]] = {}
        self.edgeProperty: dict[int, dict] = {}
        self.edgeId: int = 0
        self.buildings: list = []
        self.way_c: int = 0

    def _parse_bounds(self, root: ET.Element) -> None:
        """
        Parse map bounds from OSM XML root.

        Args:
            root: XML root element of the OSM file.
        """
        bounds = root.find('bounds')
        self.minlat = float(bounds.get('minlat'))
        self.maxlat = float(bounds.get('maxlat'))
        self.minlon = float(bounds.get('minlon'))
        self.maxlon = float(bounds.get('maxlon'))

    def _parse_nodes(self, root: ET.Element) -> None:
        """
        Parse all nodes and store them in nodedict.

        Args:
            root: XML root element of the OSM file.
        """
        for node in root.findall('node'):
            self.nodedict[node.get('id')] = {
                'node': node,
                'lat': float(node.get('lat')),
                'lon': float(node.get('lon')),
                'to': {},
                'from': {}
            }

    def _parse_way(self, way: ET.Element, road_blacklist: set[str], noUnderground: bool, includeServiceRoad: bool) -> None:
        """
        Parse a single OSM way and add edges or buildings as appropriate.

        Args:
            way: XML element for the way.
            road_blacklist: Set of highway types to ignore.
            noUnderground: If True, skip ways with layer < 0.
            includeServiceRoad: If False, exclude service roads marked as parking.
        """
        nd_refs = [nd.get('ref') for nd in way.findall('nd')]
        tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}

        highway = tags.get('highway', 'None')
        is_building = 'building' in tags
        parking = tags.get('amenity') == 'parking' or tags.get('service') in {'parking_aisle', 'driveway'}
        lanes = self._try_float(tags.get('lanes'))
        width = self._parse_width(tags.get('width'), lanes)
        layer = self._try_int(tags.get('layer'), default=0)
        cycleway = tags.get('cycleway', 'none')
        oneway = {'yes': 1, '1': 1, '-1': -1}.get(tags.get('oneway'), 0)
        building_height = self._estimate_height(tags)

        if noUnderground and layer < 0:
            return

        if is_building:
            shape = [(self.nodedict[x]['lat'], self.nodedict[x]['lon']) for x in nd_refs]
            self.buildings.append([shape, building_height])
            return

        if highway in road_blacklist or (not includeServiceRoad and parking):
            return

        self._add_edges(nd_refs, highway, width, lanes, layer, cycleway, tags, oneway)

    def _parse_width(self, width_str: str | None, lanes: float | None) -> float:
        """
        Parse width of the way, with fallback defaults.

        Args:
            width_str: Raw width tag string.
            lanes: Number of lanes parsed.

        Returns:
            Calculated width.
        """
        width = self._try_float(width_str)
        if width is None:
            width = 6.6 if lanes in (None, 1) else (lanes * 3.7 if lanes is not None else 6.6)
        elif lanes is not None and width > lanes * 3.7 * 2:
            width = width / 2
        return width

    def _try_float(self, val: str | None) -> float | None:
        """
        Try to parse a float from a possibly complex tag string.

        Args:
            val: String to parse.

        Returns:
            Parsed float or None if not parsable.
        """
        try:
            return float(val.split(';')[0].split()[0]) if val else None
        except Exception:
            return None

    def _try_int(self, val: str | None, default: int = 0) -> int:
        """
        Try to parse an integer from a string.

        Args:
            val: String to parse.
            default: Default value if parsing fails.

        Returns:
            Parsed integer or default.
        """
        try:
            return int(val)
        except Exception:
            return default

    def _estimate_height(self, tags: dict[str, str]) -> float:
        """
        Estimate building height from tags.

        Args:
            tags: Dictionary of OSM tags.

        Returns:
            Estimated height, default 6 if no info.
        """
        try:
            if 'height' in tags:
                return float(tags['height'].split()[0])
            elif 'ele' in tags:
                return float(tags['ele'].split()[0]) * 3
        except Exception:
            pass
        return 6

    def _add_edges(
        self, 
        nd_refs: list[str], 
        highway: str, 
        width: float, 
        lanes: float | None, 
        layer: int, 
        cycleway: str, 
        tags: dict[str, str], 
        oneway: int
    ) -> None:
        """
        Add edges between nodes of a way, recording edge properties and connectivity.

        Args:
            nd_refs: List of node IDs for the way.
            highway: Highway type string.
            width: Width of the way.
            lanes: Number of lanes.
            layer: Layer index (e.g., underground).
            cycleway: Cycleway type.
            tags: All tags for the way.
            oneway: Directionality of the way: 1 for oneway, -1 for reverse oneway, 0 for both directions.
        """
        def record_edge(n1: str, n2: str) -> None:
            if (n1, n2) not in self.edge2edgeid:
                eid = self.edgeId
                self.edge2edgeid[(n1, n2)] = eid
                self.edgeid2edge[eid] = (n1, n2)
                self.edgeProperty[eid] = {
                    "width": width,
                    "lane": lanes,
                    "layer": layer,
                    "roadtype": highway,
                    "cycleway": cycleway,
                    "info": dict(tags)
                }
                self.edgeId += 1

        def connect_graph(nodes: list[str]) -> None:
            for i in range(len(nodes) - 1):
                a, b = nodes[i], nodes[i + 1]
                self.nodedict[a]['to'][b] = 1
                self.nodedict[b]['from'][a] = 1
                record_edge(a, b)
                record_edge(b, a)

        if oneway >= 0:
            connect_graph(nd_refs)
            self.waydict[self.way_c] = nd_refs
            self.way_c += 1

        if oneway <= 0:
            reversed_refs = list(reversed(nd_refs))
            connect_graph(reversed_refs)
            self.waydict[self.way_c] = reversed_refs
            self.way_c += 1

def process_osm_graph(
    data_geo_bounds: Tuple[float, float, float, float],
    tile_geo_bounds: Tuple[float, float, float, float],
    output_prefix: str,
    img_pixel_width: int = 512,
    img_pixel_height: int = 512,
    optional_outputs: bool = False
) -> None:
    """
    Load OSM data for the given bounding box, build and process a graph of roads,
    densify the graph, convert coordinates, and save results and visualizations.

    Args:
        data_geo_bounds (Tuple[float, float, float, float]): Geographic bounds of the data region
            in the format (lat_start, lat_end, lon_start, lon_end).
        tile_geo_bounds (Tuple[float, float, float, float]): Geographic bounds of the tile region  
        output_prefix (str): Prefix string for output files.

    Returns:
        None: Saves processed graph and sample points to disk and generates visualization image.
    """
    data_lat_st, data_lat_ed, data_lon_st, data_lon_ed = data_geo_bounds
    tile_lat_st, tile_lat_ed, tile_lon_st, tile_lon_ed = tile_geo_bounds
    osm_map = OSMLoader([tile_lat_st, tile_lat_ed, tile_lon_st, tile_lon_ed], False, includeServiceRoad=False)
    # osm_map = OSMLoader([data_lat_st, data_lat_ed, data_lon_st, data_lon_ed], False, includeServiceRoad=False)
    node_neighbor = {}

    nodedict = osm_map.nodedict

    for node_id, node_info in nodedict.items():
        n1key = (node_info["lat"], node_info["lon"])
        neighbors = set(node_info["to"]) | set(node_info["from"])
        for nid in neighbors:
            neighbor_info = nodedict[nid]
            n2key = (neighbor_info["lat"], neighbor_info["lon"])
            node_neighbor = graphInsert(node_neighbor, n1key, n2key)

    # Filter the dictionary to remove nodes outside bounding box
    def within_geo_bounds(point: Tuple[float, float], bounds: Tuple[float, float, float, float]) -> bool:
        lat, lon = point
        lat_st, lat_ed, lon_st, lon_ed = bounds
        return lat_st <= lat <= lat_ed and lon_st <= lon <= lon_ed

    # Function to check if a point is within bounds
    def within_pixel_bounds(point, width, height):
        return point[0] <= width and point[1] <= height

    nodes_to_visualise = {}
    for key, values in node_neighbor.items():
        if within_geo_bounds(key, data_geo_bounds):
            filtered_values = [v for v in values if within_geo_bounds(v, data_geo_bounds)]
            if filtered_values:  # Only add key if it has at least one valid value
                nodes_to_visualise[key] = filtered_values

    # Filter the dictionary
    node_neighbor = graphDensify(node_neighbor)

    dense_nodes_to_visualise = {}
    for key, values in node_neighbor.items():
        if within_geo_bounds(key, data_geo_bounds):
            filtered_values = [v for v in values if within_geo_bounds(v, data_geo_bounds)]
            if filtered_values:  # Only add key if it has at least one valid value
                dense_nodes_to_visualise[key] = filtered_values

    node_neighbor_region = graph2RegionCoordinate(node_neighbor, [data_lat_st, data_lon_st, data_lat_ed, data_lon_ed])

    # Filter the dictionary
    node_neighbor_region_filter = {}
    for key, values in node_neighbor_region.items():
        if within_pixel_bounds(key, img_pixel_width, img_pixel_height):
            filtered_values = [v for v in values if within_pixel_bounds(v, img_pixel_width, img_pixel_height)]
            if filtered_values:  # Only add key if it has at least one valid value
                node_neighbor_region_filter[key] = filtered_values
    node_neighbor_refine, sample_points = graphGroundTruthPreProcess(node_neighbor_region_filter)

    graphVis2048Segmentation(
        dense_nodes_to_visualise, [tile_lat_st, tile_lon_st, tile_lat_ed, tile_lon_ed],
        f"{output_prefix}_gt_dense.png", draw_nodes=True
    )

    graphVis2048Segmentation(
        nodes_to_visualise, [tile_lat_st, tile_lon_st, tile_lat_ed, tile_lon_ed],
        f"{output_prefix}_gt.png", draw_nodes=True
    )

    logging.info("Graph saved and visualized.")

    with open(f"{output_prefix}_refine_gt_graph.p", "wb") as f:
        pickle.dump(node_neighbor_refine, f)

    if optional_outputs:
        with open(f"{output_prefix}_refine_gt_graph_samplepoints.json", "w") as f:
            json.dump(sample_points, f, indent=2)

        with open(f"{output_prefix}_graph_gt.pickle", "wb") as f:
            pickle.dump(node_neighbor_region_filter, f)

