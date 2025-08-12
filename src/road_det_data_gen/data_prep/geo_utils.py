# app/data_prep/geo_utils.py
import math
import numpy as np
from rtree import index
import cv2
import sys
from collections import defaultdict
from pyproj import Transformer, CRS
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Point = tuple[float, float]
Edge = tuple[Point, Point]
Graph = dict[Point, list[Point]]

def GPSDistance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """
    Calculate the approximate distance between two GPS coordinates.

    This function computes the distance between two points on the Earth's surface
    using a simplified formula that assumes a flat Earth. It does not account for
    the Earth's curvature and is therefore less accurate for long distances.

    Args:
        p1 (tuple): A tuple representing the first GPS coordinate (latitude, longitude).
        p2 (tuple): A tuple representing the second GPS coordinate (latitude, longitude).

    Returns:
        float: The approximate distance between the two GPS coordinates.

    Note:
        - Latitude and longitude are expected to be in decimal degrees.
        - The distance is calculated in the same units as the input coordinates.
        - For more accurate distance calculations, consider using the Haversine formula
          or geodesic distance methods.
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    delta_lat = lat1 - lat2
    delta_lon = (lon1 - lon2) * math.cos(math.radians(lat1))
    return math.hypot(delta_lat, delta_lon)


def graphInsert(
    node_neighbor: dict[tuple[int, int], list[tuple[int, int]]],
    n1key: tuple[int, int],
    n2key: tuple[int, int]
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """
    Insert an undirected edge between two nodes represented by pixel coordinates
    into the graph's adjacency list.

    Args:
        node_neighbor: A dictionary representing the graph, where keys are pixel
            coordinate tuples (x, y), and values are lists of neighboring node tuples.
        n1key: The first node as a tuple of (x, y) coordinates.
        n2key: The second node as a tuple of (x, y) coordinates.

    Returns:
        The updated graph dictionary with the edge added between n1key and n2key.
        Duplicate edges are prevented. Self-loops (edges from a node to itself)
        are ignored.
    """
    if n1key == n2key:
        return node_neighbor  # Avoid self-loop if not desired

    neighbors1 = node_neighbor.setdefault(n1key, [])
    if n2key not in neighbors1:
        neighbors1.append(n2key)

    neighbors2 = node_neighbor.setdefault(n2key, [])
    if n1key not in neighbors2:
        neighbors2.append(n1key)

    return node_neighbor


def graphDensify(node_neighbor: Graph, density: float = 0.00020) -> Graph:
    """
    Densify the graph by interpolating points along chains of nodes.
    This function identifies chains of nodes that are either isolated or have
    exactly two neighbors, and interpolates additional points along these chains
    to ensure a minimum distance between points.

    Args:
        node_neighbor: A dictionary representing the graph, where keys are node
            coordinates and values are lists of neighboring node coordinates.
        density: The minimum distance between interpolated points along a chain.

    Returns:
        A new graph dictionary with additional interpolated points along chains.
    """
    visited = set()
    new_node_neighbor = {}

    for node, neighbors in node_neighbor.items():
        if len(neighbors) == 1 or len(neighbors) > 2:
            if node in visited:
                continue

            for next_node in neighbors:
                if next_node in visited:
                    continue

                chain = [node, next_node]
                current = next_node

                while True:
                    next_neighbors = node_neighbor.get(current, [])
                    if len(next_neighbors) != 2:
                        break
                    prev = chain[-2]
                    next_candidate = next_neighbors[0] if next_neighbors[1] == prev else next_neighbors[1]
                    chain.append(next_candidate)
                    current = next_candidate

                visited.update(chain[:-1])

                # Precompute distances along the chain
                pd = [0.0]
                for a, b in zip(chain, chain[1:]):
                    pd.append(pd[-1] + GPSDistance(a, b))

                total_length = pd[-1]
                num_interp = int(total_length / density)

                if num_interp == 0:
                    graphInsert(new_node_neighbor, chain[0], chain[-1])
                    continue

                step = total_length / (num_interp + 1)
                last_pt = chain[0]
                d_cursor = step
                j = 0

                for _ in range(num_interp):
                    while pd[j + 1] < d_cursor:
                        j += 1

                    seg_len = pd[j + 1] - pd[j]
                    a = (d_cursor - pd[j]) / seg_len if seg_len > 0 else 0.0

                    x1, y1 = chain[j]
                    x2, y2 = chain[j + 1]
                    pt = ((1 - a) * x1 + a * x2, (1 - a) * y1 + a * y2)

                    graphInsert(new_node_neighbor, last_pt, pt)
                    last_pt = pt
                    d_cursor += step

                graphInsert(new_node_neighbor, last_pt, chain[-1])

    return new_node_neighbor


def _latlon_to_pixel(lat: float, lon: float, region: tuple[float, float, float, float], scale: int) -> tuple[float, float]:
    """
    Convert GPS coordinates to pixel coordinates based on the bounding region and scale.

    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        region (tuple): Bounding box (min_lat, min_lon, max_lat, max_lon).
        scale (int): Target resolution (e.g., 512 or 2048).

    Returns:
        Tuple[float, float]: Pixel coordinates (y, x) in image space.
    """
    min_lat, min_lon, max_lat, max_lon = region
    inv_width = scale / (max_lon - min_lon)
    inv_height = scale / (max_lat - min_lat)

    x = (lon - min_lon) * inv_width
    y = (max_lat - lat) * inv_height  # Y is flipped due to image coordinate system
    return y, x


def graph2RegionCoordinate(
    node_neighbor: Graph,
    region: tuple[float, float, float, float],
    scale: int = 512
) -> dict[Point, list[Point]]:
    """
    Convert a GPS-based graph to pixel coordinates within a given region.

    Args:
        node_neighbor: Dictionary representing the graph with GPS coordinates.
        region: Tuple (min_lat, min_lon, max_lat, max_lon).
        scale: Image resolution for coordinate mapping (default is 512).

    Returns:
        A new graph dictionary with pixel-based node coordinates.
    """
    new_node_neighbor: dict[Point, list[Point]] = {}

    for node, neighbors in node_neighbor.items():
        y0, x0 = _latlon_to_pixel(node[0], node[1], region, scale)

        for neighbor in neighbors:
            y1, x1 = _latlon_to_pixel(neighbor[0], neighbor[1], region, scale)

            n1key = (y0, x0)
            n2key = (y1, x1)

            graphInsert(new_node_neighbor, n1key, n2key)

    return new_node_neighbor


def graphVis2048Segmentation(
    node_neighbor: dict[tuple[float, float], list[tuple[float, float]]],
    region: tuple[float, float, float, float],
    filename: str,
    size: int = 512,
    draw_nodes: bool = True
) -> None:
    """
    Render a visual representation of a graph onto an image with:
    - white lines for edges
    - optionally, red dots for nodes

    Args:
        node_neighbor: Graph with pixel-based coordinates.
        region: Tuple (min_lat, min_lon, max_lat, max_lon).
        filename: Output filename for the visualization image.
        size: Size of the square output image (default is 512).
        draw_nodes: If True, draw red circles at each node location (default: True).
    """
    # Create a 3-channel image for colored drawing
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Draw edges (white)
    for node, neighbors in node_neighbor.items():
        x0 = int((node[1] - region[1]) / (region[3] - region[1]) * size)
        y0 = int((region[2] - node[0]) / (region[2] - region[0]) * size)

        for neighbor in neighbors:
            x1 = int((neighbor[1] - region[1]) / (region[3] - region[1]) * size)
            y1 = int((region[2] - neighbor[0]) / (region[2] - region[0]) * size)

            cv2.line(img, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=2)

    # Optionally draw nodes (red)
    if draw_nodes:
        for node in node_neighbor:
            x = int((node[1] - region[1]) / (region[3] - region[1]) * size)
            y = int((region[2] - node[0]) / (region[2] - region[0]) * size)
            cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(filename, img)


def locate_stacking_road(graph: dict[Point, list[Point]]) -> tuple[dict[tuple[Edge, Edge], Point], dict[Point, list[Point]]]:
    """
    Detect and adjust overlapping (stacked) road segments in a graph.

    Args:
        graph: A dictionary representing the road network as an adjacency list.

    Returns:
        - A dictionary of crossing points between intersecting edges.
        - An adjustment dictionary mapping points to displacement vectors to reduce stacking.
    """
    idx = index.Index()
    seen_edges = set()
    edges: list[Edge] = []

    # Collect unique undirected edges and index them
    for n1, neighbors in graph.items():
        for n2 in neighbors:
            edge_key = tuple(sorted((n1, n2)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append((n1, n2))

            x1, y1 = n1
            x2, y2 = n2
            xmin, xmax = sorted((x1, x2))
            ymin, ymax = sorted((y1, y2))
            idx.insert(len(edges) - 1, (xmin, ymin, xmax, ymax))

    crossing_point: dict[tuple[Edge, Edge], Point] = {}
    adjustment: dict[Point, list[Point]] = defaultdict(list)
    threshold = 9.5

    for i, (n1, n2) in enumerate(edges):
        x1, y1 = n1
        x2, y2 = n2
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))

        candidates = idx.intersection((xmin, ymin, xmax, ymax))
        for c_idx in candidates:
            if c_idx == i:
                continue
            c1, c2 = edges[c_idx]

            # Skip if edges share endpoints
            if set((n1, n2)) & set((c1, c2)):
                continue

            if intersect(n1, n2, c1, c2):
                ip = intersectPoint(n1, n2, c1, c2)
                crossing_point[( (n1, n2), (c1, c2) )] = ip

                for p1, p2 in [(n1, n2), (n2, n1), (c1, c2), (c2, c1)]:
                    d = distance(ip, p1)
                    if d < threshold:
                        norm = neighbors_norm(p1, p2)
                        weight = (threshold - d) / threshold
                        adjustment[p1].append((float(norm[0] * weight), float(norm[1] * weight)))
    return crossing_point, dict(adjustment)


def locate_parallel_road(graph: Graph) -> list[Point]:
    """
    Identify nodes that are part of road segments running parallel to each other within close proximity.

    Args:
        graph: A dictionary representing a road network, where each node maps to a list of connected nodes.

    Returns:
        A list of points (nodes) that are part of parallel road segments.
    """
    idx = index.Index()
    seen_edges = set()
    edges: list[Edge] = []

    for n1, neighbors in graph.items():
        for n2 in neighbors:
            edge_key = tuple(sorted((n1, n2)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append((n1, n2))

            x1, y1 = n1
            x2, y2 = n2
            xmin, xmax = sorted((x1, x2))
            ymin, ymax = sorted((y1, y2))
            idx.insert(len(edges) - 1, (xmin, ymin, xmax, ymax))

    parallel_road: list[Point] = []
    offset = 20
    min_dist = 10
    cos_sim_threshold = 0.985

    for i, (n1, n2) in enumerate(edges):
        if distance(n1, n2) < min_dist:
            continue

        x1, y1 = n1
        x2, y2 = n2
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        bbox = (xmin - offset, ymin - offset, xmax + offset, ymax + offset)

        candidates = idx.intersection(bbox)

        for c_idx in candidates:
            if c_idx == i:
                continue

            c1, c2 = edges[c_idx]
            if len({n1, n2, c1, c2}) < 4:  # Shared nodes
                continue

            # Skip if n1 or n2 is indirectly connected to c1 or c2
            try:
                if any(c in graph.get(nei, []) for node in (n1, n2) for nei in graph[node] for c in (c1, c2)):
                    continue
            except KeyError:
                # If a node was removed or not found, skip this candidate
                logging.warning("Node %s or %s not found in graph during parallel road detection.", n1, n2)
                continue

            vec1 = (n2[0] - n1[0], n2[1] - n1[1])
            vec2 = (c2[0] - c1[0], c2[1] - c1[1])

            if abs(neighbors_cos((0, 0), vec1, vec2)) > cos_sim_threshold:
                parallel_road.append(n1)
                break  # Avoid adding n1 multiple times

    return parallel_road



def apply_adjustment(graph: Graph, adjustment: dict[Point, list[tuple[float, float]]]) -> tuple[Graph, int]:
    """
    Apply vector-based positional adjustments to nodes in a graph.

    Args:
        graph: A dictionary representing the graph where keys are nodes (coordinates),
               and values are lists of connected nodes.
        adjustment: A dictionary mapping each node to a list of adjustment vectors.

    Returns:
        A tuple containing:
            - The adjusted graph.
            - The number of successfully adjusted nodes.
    """
    current_graph = graph.copy()
    moved_nodes = 0

    for node, vectors in adjustment.items():
        # Compute average adjustment vector
        vec = np.sum(np.array(vectors), axis=0)
        norm = np.linalg.norm(vec)

        if norm == 0:
            continue

        # Normalize if needed
        if norm > 1.0:
            vec /= norm

        for scale in [1.5, 1.0]:  # Try larger movement first
            new_node = (float(node[0] + vec[0] * scale), float(node[1] + vec[1] * scale))

            if new_node == node:
                continue

            if new_node not in current_graph:
                neighbors = current_graph.pop(node, [])

                current_graph[new_node] = neighbors

                for neighbor in neighbors:
                    if neighbor == node:
                        continue  # Skip self-loop to the removed node
                    try:
                        current_graph[neighbor] = [
                            new_node if n == node else n for n in current_graph[neighbor]
                        ]
                    except KeyError:
                        logging.warning("Node %s not found in graph during adjustment.", node)
                        logging.warning("Warning: Node %s not found in graph during adjustment.", neighbor)
                        sys.exit()
                        continue

                moved_nodes += 1
                break  # Move applied, stop trying lower scale

    logging.info("Adjusted %d nodes", moved_nodes)
    return current_graph, moved_nodes


def graph_move_node(graph: Graph, old_n: Point, new_n: Point) -> Graph:
    """
    Move a node in the graph to a new coordinate while preserving connections.

    Args:
        graph: The adjacency list representing the graph.
        old_n: The current coordinate of the node to move.
        new_n: The new coordinate to move the node to.

    Returns:
        The updated graph with the node moved.
    """
    if old_n == new_n or old_n not in graph:
        return graph  # No change needed or invalid move

    neighbors = graph.pop(old_n)
    graph[new_n] = neighbors

    for neighbor in neighbors:
        graph[neighbor] = [new_n if n == old_n else n for n in graph[neighbor]]

    return graph


def apply_adjustment_delete_closeby_nodes(graph: Graph, adjustment: dict[Point, list[tuple[float, float]]]) -> Graph:
    """
    Delete nodes with dense and redundant adjustment vectors and reconnect neighbors.

    If a node has at least 4 adjustment vectors that suggest close-by overlapping,
    and it has exactly two neighbors, the node is removed and its neighbors are
    connected directly and nudged closer to the deleted node.

    Args:
        graph: The graph as an adjacency list.
        adjustment: Adjustment vectors for certain nodes.

    Returns:
        The updated graph with select nodes removed and their neighbors adjusted.
    """
    threshold = 9.5

    nodes_to_delete = [
        k for k, v in adjustment.items()
        if len(v) >= 4 and len(graph.get(k, [])) == 2
    ]

    deleted = set()
    for k in nodes_to_delete:
        if k in deleted:
            continue
        neighbors = graph.get(k, [])
        if len(neighbors) != 2:
            continue
        nei1, nei2 = neighbors
        if nei1 in deleted or nei2 in deleted:
            continue  # One of them already changed

        dists = [(1.0 - distance(vv, (0.0, 0.0))) * threshold for vv in adjustment[k]]
        dists.sort()
        gap = sum(dists[:4]) / 2.0

        if gap >= 12.0:
            continue

        # proceed
        del graph[k]
        deleted.add(k)
        logging.info("Deleted node: %s", k)

        # Replace old node in neighbors' lists
        graph[nei1] = [n for n in graph[nei1] if n != k]
        if nei2 not in graph[nei1]:
            graph[nei1].append(nei2)

        graph[nei2] = [n for n in graph[nei2] if n != k]
        if nei1 not in graph[nei2]:
            graph[nei2].append(nei1)

        # Push nei1 closer to nei2 if not already in adjustment
        if nei1 not in adjustment:
            vec = neighbors_norm(k, nei1)
            new_nei1 = (nei1[0] + vec[0] * 5.0, nei1[1] + vec[1] * 5.0)
            graph = graph_move_node(graph, nei1, new_nei1)

        if nei2 not in adjustment:
            vec = neighbors_norm(k, nei2)
            new_nei2 = (nei2[0] + vec[0] * 5.0, nei2[1] + vec[1] * 5.0)
            graph = graph_move_node(graph, nei2, new_nei2)

    return graph



def graphGroundTruthPreProcess(graph: Graph) -> tuple[Graph, dict[str, list]]:
    """
    Refines the input graph by repeatedly resolving stacked roads and pruning redundant nodes.
    Also collects sample points for various structural graph features.

    Args:
        graph: A spatial graph represented as an adjacency list.

    Returns:
        A tuple containing the refined graph and a dictionary of feature samples.
    """
    for iteration in range(40):  # Empirical max iterations
        crossing_points, adjustments = locate_stacking_road(graph)

        if iteration % 5 == 0 and iteration != 0:
            graph  = apply_adjustment_delete_closeby_nodes(graph, adjustments)
        else:
            graph, count = apply_adjustment(graph, adjustments)
            if count == 0:
                break  # Converged

    # Collect structural feature samples
    sample_points = {
        'parallel_road': locate_parallel_road(graph),
        'complicated_intersections': [node for node, neighbors in graph.items() if len(neighbors) > 4],
        'overpass': [(int(p[0]), int(p[1])) for p in crossing_points.values()]
    }

    return graph, sample_points


def neighbors_norm(k1: tuple[float, float], k2: tuple[float, float]) -> tuple[float, float]:
    """
    Compute the normalized direction vector from k2 to k1.

    Args:
        k1: Target point.
        k2: Origin point.

    Returns:
        A 2D unit vector (dx, dy) from k2 to k1.
    """
    vec = np.array(k1) - np.array(k2)
    length = np.hypot(*vec)
    if length == 0:
        return (0.0, 0.0)
    return (float((vec / length)[0]), float((vec / length)[1]))


def neighbors_cos(k1: tuple[float, float], k2: tuple[float, float], k3: tuple[float, float]) -> float:
    """
    Compute the cosine of the angle between vectors (k1 → k2) and (k1 → k3).

    Args:
        k1: The shared origin point.
        k2: First direction endpoint.
        k3: Second direction endpoint.

    Returns:
        Cosine of the angle between the two vectors.
    """
    vec1 = np.array(neighbors_norm(k2, k1))
    vec2 = np.array(neighbors_norm(k3, k1))
    return float(np.dot(vec1, vec2))


def ccw(A: tuple[float,float], B: tuple[float,float] ,C: tuple[float,float]) -> bool:
    """
    Check if the points A, B, C are in counter-clockwise order.

    Args:
        A, B, C: Points as tuples (x, y).

    Returns:
        bool: True if A, B, C are in counter-clockwise order, False otherwise.
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A: tuple[float,float], B: tuple[float,float] ,C: tuple[float,float],D: tuple[float,float]) -> bool:
    """
    Check if line segments AB and CD intersect.

    Args:
        A, B: Endpoints of the first segment as tuples (x, y).
        C, D: Endpoints of the second segment as tuples (x, y).

    Returns:
        bool: True if segments AB and CD intersect, False otherwise.
    """
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points in 2D space.

    Args:
        p1 (tuple): First point as a tuple (x, y).
        p2 (tuple): Second point as a tuple (x, y).

    Returns:
        float: The Euclidean distance between the two points.
    """
    diff = np.array(p1) - np.array(p2)
    return np.hypot(*diff)


def intersectPoint(
    A: tuple[float, float],
    B: tuple[float, float],
    C: tuple[float, float],
    D: tuple[float, float],
    resolution: int = 200
) -> tuple[float, float]:
    """
    Vectorized brute-force method to find the point on segment AB that minimizes
    the sum of distances to points C and D.

    Args:
        A, B: Endpoints of the segment as (x, y) tuples.
        C, D: Target points to minimize distance sum to.
        resolution: Number of points sampled along the line segment.

    Returns:
        The optimal (x, y) point on AB minimizing distance(P, C) + distance(P, D).
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    # Sample `resolution` points between A and B
    t = np.linspace(0, 1, resolution)
    points = (1 - t)[:, None] * A + t[:, None] * B  # Shape: (resolution, 2)

    # Compute distances from all points to C and D
    dists_to_C = np.linalg.norm(points - C, axis=1)
    dists_to_D = np.linalg.norm(points - D, axis=1)
    total_dists = dists_to_C + dists_to_D

    # Find the point with the minimum total distance
    min_index = np.argmin(total_dists)
    return (float(points[min_index][0]), float(points[min_index][1]))




def latlon_bbox_from_center(
    lat: float,
    lon: float,
    pixels: int,
    scale: float
) -> tuple[float, float, float, float]:
    """
    Create a bounding box around a lat/lon point with given pixel size and resolution.

    Args:
        lat (float): Latitude of center point.
        lon (float): Longitude of center point.
        pixels (int): Width and height of image in pixels (assumes square).
        scale (float): Ground sampling distance (meters per pixel).

    Returns:
        (lat_min, lon_min, lat_max, lon_max): Bounding box in WGS84.
    """

    # Total box size in meters
    half_size_m = (pixels * scale) / 2.0

    # Determine UTM zone from center point
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0

    utm_crs = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs{' +north' if is_northern else ' +south'}")
    wgs84 = CRS.from_epsg(4326)

    # Create transformers
    to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    to_latlon = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    # Convert center to UTM
    center_x, center_y = to_utm.transform(lon, lat)

    # Calculate bounding box in UTM
    min_x = center_x - half_size_m
    max_x = center_x + half_size_m
    min_y = center_y - half_size_m
    max_y = center_y + half_size_m

    # Convert back to lat/lon
    lon_min, lat_min = to_latlon.transform(min_x, min_y)
    lon_max, lat_max = to_latlon.transform(max_x, max_y)

    tiles = []
    tiles.append((lat_min, lon_min, lat_max, lon_max))

    return tiles

def tile_bbox_from_latlon_bbox(cfg: dict) -> list[tuple[float, float, float, float]]:
    scale = cfg["scale"]
    pixels = cfg["pixels"]
    overlap_pixels = cfg.get("overlap_pixels", 0)

    tile_size_m = pixels * scale
    overlap_m = scale * overlap_pixels
    stride_m = tile_size_m - overlap_m

    # Define center and UTM zone
    center_lat = (cfg["start_lat"] + cfg["end_lat"]) / 2
    center_lon = (cfg["start_lon"] + cfg["end_lon"]) / 2

    utm_crs = CRS.from_proj4(f"+proj=utm +zone={(int((center_lon + 180) / 6) + 1)} +datum=WGS84 +units=m +no_defs")
    wgs84 = CRS.from_epsg(4326)

    # Transformer
    to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    to_latlon = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    # Project bounding box to UTM
    min_x, min_y = to_utm.transform(cfg["start_lon"], cfg["start_lat"])
    max_x, max_y = to_utm.transform(cfg["end_lon"], cfg["end_lat"])

    # Generate tiles
    tiles = []
    y = min_y
    while y + tile_size_m <= max_y + 1:
        x = min_x
        while x + tile_size_m <= max_x + 1:
            # Tile corners in UTM
            tile_bounds_utm = (x, y, x + tile_size_m, y + tile_size_m)

            # Convert to lat-lon
            lon_min, lat_min = to_latlon.transform(x, y)
            lon_max, lat_max = to_latlon.transform(x + tile_size_m, y + tile_size_m)

            tiles.append((lat_min, lon_min, lat_max, lon_max))
            x += stride_m
        y += stride_m

    return tiles

