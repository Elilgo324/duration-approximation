"""Fixed road locations, directed landmarks, and the three-layer DP baseline."""
from __future__ import annotations

import base64
import json
import math
import struct
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h3
import numpy as np
import requests

from .runtime import save_json


def directed_segment_ids(hint):
    """Read the two SegmentID bitfields in pinned OSRM 5.27.1 x86_64 hints.

    This only inspects an OSRM-issued hint, never fabricates one. Fail closed if
    its layout changes. See upstream PhantomNode and util::SegmentID definitions.
    """
    data = base64.urlsafe_b64decode(hint)
    if len(data) < 8:
        raise ValueError("Unexpected OSRM hint")
    fields = struct.unpack_from("<II",data)
    enabled = tuple(v & 0x7fffffff for v in fields if v & 0x80000000)
    if len(enabled) not in (1,2):
        raise ValueError("OSRM hint has no valid directed segment")
    return enabled


def bearing(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    return math.degrees(math.atan2(math.sin(lon2-lon1)*math.cos(lat2),
           math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1))) % 360


def offsets_m(points, origin):
    points = np.asarray(points)
    return (points - np.asarray(origin)) * np.array([111195*math.cos(math.radians(origin[1])), 111195])


class Router:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        # Avoid the pinned server's ~40 ms keep-alive response delay on loopback.
        # Every query includes its own local HTTP/TCP setup in reported latency.
        self.session.headers["Connection"] = "close"

    def call(self, service, points, **options):
        coordinates = ";".join(f"{p['input'][0]:.6f},{p['input'][1]:.6f}" for p in points)
        if all("hint" in p for p in points):
            options["hints"] = ";".join(p["hint"] for p in points)
        response = self.session.get(f"{self.url}/{service}/v1/driving/{coordinates}", params=options, timeout=120)
        data = response.json()
        if data.get("code") in ("NoSegment", "NoRoute"):
            return None
        response.raise_for_status()
        if data.get("code") != "Ok":
            raise RuntimeError(data)
        return data

    def snap_candidates(self, coordinate, radius=None, heading=None, number=1):
        coordinate = [round(float(x), 6) for x in coordinate]
        options = {"number": number}
        if radius is not None:
            options["radiuses"] = radius
        if heading is not None:
            options["bearings"] = f"{heading},5"
        response = self.call("nearest", [{"input": coordinate}], **options)
        if response is None:
            return []
        for point in response["waypoints"]:
            point["input"] = coordinate
            if heading is not None:
                point["heading"] = heading
        return response["waypoints"]

    def snap(self, coordinate, radius=None, heading=None):
        points = self.snap_candidates(coordinate, radius, heading)
        return points[0] if points else None

    def landmark(self, coordinate, nodes):
        nearest = self.snap(coordinate)
        if nearest is None:
            raise ValueError(f"No driving road near landmark {coordinate}")
        return self._directed_landmark(nearest,nodes)

    def _directed_landmark(self, nearest, nodes):
        coordinate = nearest["input"]
        segments = directed_segment_ids(nearest["hint"])
        if len(segments) == 1:
            nearest["directed_segment"] = segments[0]
            return nearest
        a, b = nearest["nodes"]
        if a not in nodes or b not in nodes or a == b:
            raise ValueError("Landmark segment endpoints unavailable in map extract")
        # Choose a deterministic orientation independent of the trip, trying the reverse
        # only when that orientation is not drivable on this exact nearest segment.
        a, b = sorted((a, b))
        heading = round(bearing(nodes[a], nodes[b])) % 360
        attempts = []

        def matching_point(points, direction):
            for point in points:
                candidate = directed_segment_ids(point["hint"])
                displacement = float(np.linalg.norm(offsets_m([point["location"]], nearest["location"])))
                attempts.append(dict(heading=direction, segments=candidate, displacement_m=displacement))
                if len(candidate) == 1 and candidate[0] in segments and displacement < 0.2:
                    point["directed_segment"] = candidate[0]
                    return point
            return None

        for direction in (heading, (heading+180) % 360):
            point = matching_point(self.snap_candidates(coordinate, heading=direction), direction)
            if point is not None:
                return point
        # OSRM stores coordinates at 1e-6 degree precision. Raw OSM coordinates
        # can give a different bearing on very short segments. At junctions, the
        # first bearing-filtered result can also be another equally close road.
        def osrm_coordinate(values):
            return [math.copysign(math.floor(abs(x)*1e6+0.5)/1e6, x) for x in values]

        fixed_heading = round(bearing(osrm_coordinate(nodes[a]), osrm_coordinate(nodes[b]))) % 360
        for direction in (fixed_heading, (fixed_heading+180) % 360):
            point = matching_point(self.snap_candidates(coordinate, heading=direction, number=8,
                                   radius=nearest["distance"]+1), direction)
            if point is not None:
                print("LANDMARK_SNAP_RECOVERY " + json.dumps(dict(coordinate=list(coordinate),
                      original_heading=heading, fixed_heading=fixed_heading, attempts=attempts)), flush=True)
                return point
        raise ValueError("Could not preserve nearest segment while fixing its legal direction: " +
                         json.dumps(dict(coordinate=list(coordinate), nearest=nearest, attempts=attempts)))

    def endpoint_directions(self, point, nodes):
        """Keep every legal direction at this exact snapped endpoint position.

        OSRM's ambiguous two-point query can miss the better arrival direction
        on the same compressed edge. Explicit states avoid that case without
        restricting the coordinate's legal departure/arrival choices.
        """
        segments = directed_segment_ids(point["hint"])
        first = self._directed_landmark(dict(point),nodes)
        if len(segments) == 1:
            return [first]
        other = next(segment for segment in segments if segment != first["directed_segment"])
        heading = (first["heading"]+180) % 360
        for candidate in self.snap_candidates(point["input"], heading=heading, number=8, radius=point["distance"]+1):
            if (directed_segment_ids(candidate["hint"]) == (other,) and
                    np.linalg.norm(offsets_m([candidate["location"]],point["location"])) < .2):
                candidate["directed_segment"] = other
                return [first,candidate]
        raise ValueError(f"Could not preserve both endpoint directions at {point['input']}")

    def table(self, sources, destinations):
        source_groups = [p.get("directions",[p]) for p in sources]
        destination_groups = [p.get("directions",[p]) for p in destinations]
        source_points = [p for group in source_groups for p in group]
        destination_points = [p for group in destination_groups for p in group]
        points = source_points + destination_points
        response = self.call("table", points, sources=";".join(map(str, range(len(source_points)))),
                             destinations=";".join(map(str, range(len(source_points), len(points)))),
                             annotations="duration", generate_hints="false")
        if response is None:
            return np.full((len(sources), len(destinations)), np.inf)
        durations = np.array([[np.inf if x is None else x for x in row] for row in response["durations"]])
        source_starts = np.cumsum([0]+[len(group) for group in source_groups[:-1]])
        destination_starts = np.cumsum([0]+[len(group) for group in destination_groups[:-1]])
        return np.minimum.reduceat(np.minimum.reduceat(durations,source_starts,axis=0),destination_starts,axis=1)

    def route(self, points):
        if "directions" in points[0] or "directions" in points[-1]:
            # Joined validation must use the same explicit endpoint states as
            # the tables; ambiguous hints can miss the shorter departure too.
            return min(self.route([source,*points[1:-1],destination])
                       for source in points[0].get("directions",[points[0]])
                       for destination in points[-1].get("directions",[points[-1]]))
        response = self.call("route", points, overview="false", steps="false", continue_straight="true")
        return np.inf if response is None else response["routes"][0]["duration"]


def road_geometry(osm: Path, bbox):
    nodes = {}
    segments = []
    allowed = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
               "secondary", "secondary_link", "tertiary", "tertiary_link", "residential",
               "unclassified", "living_street", "service"}
    west, south, east, north = bbox
    for _, elem in ET.iterparse(osm, events=("end",)):
        if elem.tag == "node":
            nodes[int(elem.attrib["id"])] = [float(elem.attrib["lon"]), float(elem.attrib["lat"])]
            elem.clear()
        elif elem.tag == "way":
            tags = {e.attrib["k"]: e.attrib["v"] for e in elem.findall("tag")}
            if tags.get("highway") in allowed and tags.get("access") not in {"no", "private"}:
                refs = [int(e.attrib["ref"]) for e in elem.findall("nd")]
                for a, b in zip(refs, refs[1:]):
                    if a in nodes and b in nodes:
                        p, q = nodes[a], nodes[b]
                        if all(west <= x[0] <= east and south <= x[1] <= north for x in (p, q)):
                            segments.append([*p, *q])
            elem.clear()
        elif elem.tag == "relation":
            elem.clear()
    if not segments:
        raise ValueError("No road segments inside the study rectangle")
    return nodes, np.array(segments)


def road_cells(segments, resolutions):
    cells = {r: set() for r in resolutions}
    # Cover road geometry at 100 m spacing, then expand one ring to include every
    # cell that a sampled endpoint can enter after its <=100 m road snap.
    for segment in segments:
        a, b = segment[:2], segment[2:]
        length = np.linalg.norm(offsets_m([b], a))
        for fraction in np.linspace(0, 1, max(2, math.ceil(length/100)+1)):
            lon, lat = a + fraction*(b-a)
            for r in resolutions:
                cells[r].add(h3.latlng_to_cell(lat, lon, r))
    return {r: sorted({neighbor for cell in group for neighbor in h3.grid_disk(cell, 1)})
            for r, group in cells.items()}


def build_landmarks(router, cells, nodes, output):
    started = time.perf_counter()
    vertex_points = {}
    landmarks = []
    landmark_ids = {}
    cell_landmarks = {}
    for index, cell in enumerate(cells):
        vertices = h3.cell_to_vertexes(cell)
        if len(vertices) != 6:
            raise ValueError("Protocol requires six-vertex H3 cells")
        ids = []
        for vertex in vertices:
            if vertex not in vertex_points:
                lat, lon = h3.vertex_to_latlng(vertex)
                point = router.landmark([lon, lat], nodes)
                identity = (point["directed_segment"], *point["location"])
                if identity not in landmark_ids:
                    landmark_ids[identity] = len(landmarks)
                    landmarks.append(point)
                vertex_points[vertex] = dict(vertex=[lon, lat], landmark=landmark_ids[identity], snap_m=point["distance"])
            ids.append(vertex_points[vertex]["landmark"])
        cell_landmarks[cell] = ids
        if index % 100 == 0:
            print(f"Landmarks: {index}/{len(cells)} cells; {len(landmarks)} unique", flush=True)
    save_json(output / "landmarks.json", dict(landmarks=landmarks, vertices=vertex_points, cells=cell_landmarks))
    return landmarks, cell_landmarks, time.perf_counter()-started


def landmark_matrix(router, landmarks, output, block_size):
    n = len(landmarks)
    matrix = np.lib.format.open_memmap(output / "landmark_durations.npy", mode="w+", dtype="float32", shape=(n,n))
    started = time.perf_counter()
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            matrix[i:i+block_size, j:j+block_size] = router.table(landmarks[i:i+block_size], landmarks[j:j+block_size])
        matrix.flush()
        if i % (block_size*8) == 0:
            print(f"Landmark matrix: {min(i+block_size,n)}/{n} rows", flush=True)
    return matrix, time.perf_counter()-started


def dp_duration(first_leg, middle, last_leg):
    candidates = np.asarray(first_leg)[:,None] + np.asarray(middle) + np.asarray(last_leg)[None,:]
    i, j = np.unravel_index(np.argmin(candidates), candidates.shape)
    return float(candidates[i,j]), int(i), int(j)


def feature_row(pickup, dropoff, pickup_landmarks, dropoff_landmarks, middle):
    p, q = pickup["location"], dropoff["location"]
    delta = offsets_m([q], p)[0]
    # No endpoint routing durations or ground-truth-derived information enters ML.
    geometry = np.r_[offsets_m([a["location"] for a in pickup_landmarks], p).ravel(),
                     offsets_m([a["location"] for a in dropoff_landmarks], q).ravel(),
                     delta, np.linalg.norm(delta)]
    return np.r_[geometry, np.asarray(middle).ravel()]
