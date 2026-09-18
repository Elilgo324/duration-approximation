"""Pinned OSRM installation, historical OSM extracts, and local routing lifecycle."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager

import requests

OSRM_IMAGE = "ghcr.io/project-osrm/osrm-backend@sha256:b1ca5d72da456e82f81b8732a095c2357b00710099efa844040d1eb40b4b5386"
# Official v5.27.1 Bullseye image: executable/TBB layer and Boost/Lua layer.
# Its Boost 1.74 build supports the worker's older kernel syscall interface.
OSRM_LAYERS = ("fefad04b76fafc77824e8ad1af40395aba8da794e39b7480b25dab97bdba201a",
               "1f5dd719c1297eb4d0ebb7709b15d0d8d9c0d1d1700efd1cdacc4928373c7e58")
SOURCE_URL = "https://codeload.github.com/Project-OSRM/osrm-backend/tar.gz/refs/tags/v5.27.1"
SOURCE_SHA = "52391580e0f92663dd7b21cbcc7b9064d6704470e2601bf3ec5c5170b471629a"
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter")
MAP_HEADERS = {"User-Agent": "DurationApproximationResearch/1.0 (OSM academic benchmark)",
               "Accept": "application/xml"}


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_checked(url: str, path: Path, expected: str, headers=None):
    with requests.get(url, stream=True, timeout=(30, 180), headers=headers) as response:
        response.raise_for_status()
        with path.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    if digest(path) != expected:
        raise ValueError(f"Checksum mismatch for {url}")


def install_osrm(work: Path, output: Path) -> tuple[Path, Path]:
    work.mkdir(parents=True, exist_ok=True)
    token_response = requests.get("https://ghcr.io/token", params=dict(service="ghcr.io",
        scope="repository:project-osrm/osrm-backend:pull"), timeout=30)
    token_response.raise_for_status()
    headers = {"Authorization": "Bearer " + token_response.json()["token"]}
    manifest = work/"image_manifest.json"
    download_checked("https://ghcr.io/v2/project-osrm/osrm-backend/manifests/" + OSRM_IMAGE.split("@")[1],
        manifest, OSRM_IMAGE.split("sha256:")[1], dict(headers, Accept="application/vnd.docker.distribution.manifest.v2+json"))
    manifest_layers = {layer["digest"] for layer in json.loads(manifest.read_text())["layers"]}
    rootfs = work/"runtime"
    for sha in OSRM_LAYERS:
        if "sha256:" + sha not in manifest_layers:
            raise ValueError("Runtime layer is not part of pinned official image")
        archive = work/f"{sha}.tar.gz"
        download_checked("https://ghcr.io/v2/project-osrm/osrm-backend/blobs/sha256:"+sha, archive, sha, headers)
        with tarfile.open(archive) as tar:
            members = [m for m in tar if m.name.startswith("usr/local/bin/osrm-") or
                       (m.name.startswith(("usr/local/lib/", "usr/lib/x86_64-linux-gnu/", "lib/x86_64-linux-gnu/"))
                        and ".so" in Path(m.name).name)]
            tar.extractall(rootfs, members=members, filter="data")
    shutil.copyfile(manifest, output/"osrm_image_manifest.json")
    archive = work/"source.tar.gz"
    download_checked(SOURCE_URL, archive, SOURCE_SHA)
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    binaries = rootfs / "usr/local/bin"
    os.environ["LD_LIBRARY_PATH"] = ":".join(str(rootfs/p) for p in
        ("usr/local/lib", "usr/lib/x86_64-linux-gnu", "lib/x86_64-linux-gnu"))
    profiles = work / "osrm-backend-5.27.1" / "profiles"
    profile = profiles / "car.lua"
    original = profile.read_text()
    assert "weight_name                     = 'routability'" in original
    profile.write_text(original.replace("weight_name                     = 'routability'", "weight_name                     = 'duration'", 1))
    shutil.copytree(profiles, output / "routing_profiles")
    version = subprocess.check_output([str(binaries / "osrm-extract"), "--version"], text=True).strip()
    if "5.27.1" not in version:
        raise ValueError(f"Unexpected OSRM version: {version}")
    save_json(output / "routing_runtime.json", dict(version=version, binary_image=OSRM_IMAGE,
              binary_layer_sha256=OSRM_LAYERS, profile_source_url=SOURCE_URL, source_sha256=SOURCE_SHA,
              profile_sha256=digest(profile), weight="duration", traffic=False))
    # Exercise file access and Lua/profile initialization before any large map request.
    fixture = work / "preflight.osm"
    fixture.write_text('''<osm version="0.6" generator="duration-research-preflight">
<node id="1" lat="32.00" lon="34.80" version="1"/>
<node id="2" lat="32.01" lon="34.80" version="1"/>
<node id="3" lat="32.01" lon="34.81" version="1"/>
<node id="4" lat="32.00" lon="34.81" version="1"/>
<way id="1" version="1"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
<tag k="highway" v="residential"/></way></osm>''')
    print("Routing runtime preflight", flush=True)
    preflight = subprocess.run([str(binaries / "osrm-extract"), str(fixture), "-p", str(profile), "-t", "1"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / "routing_preflight.log").write_text(preflight.stdout)
    if preflight.returncode:
        print(preflight.stdout[-12000:], flush=True)
        preflight.check_returncode()
    return binaries, profile


def download_map(city: dict, timestamp: str, buffer: float, output: Path, work: Path) -> Path:
    west, south, east, north = city["bbox"]
    bbox = (south-buffer, west-buffer, north+buffer, east+buffer)
    # Include restrictions and all their referenced ways/nodes, including via-way restrictions.
    query = (f'[out:xml][timeout:240][maxsize:536870912][date:"{timestamp}"];'
             f'way["highway"]({",".join(map(str, bbox))})->.roads;'
             'rel(bw.roads)["type"="restriction"]->.restrictions;'
             '(.roads;.restrictions;);(._;>>;);out meta;')
    (output / "overpass_query.txt").write_text(query)
    path = work / "map.osm"
    started = time.perf_counter()
    reference_dir = os.environ.get("EXPERIMENT_REFERENCE_DIR")
    if reference_dir:
        reference = Path(reference_dir)/city["name"]
        archived_map = reference/"map.osm.gz"
        if archived_map.is_file():
            identity = json.loads((reference/"map_identity.json").read_text())
            if identity["query_sha256"] != hashlib.sha256(query.encode()).hexdigest():
                raise ValueError("Archived map query does not match the frozen protocol")
            with gzip.open(archived_map,"rb") as source, path.open("wb") as target:
                shutil.copyfileobj(source,target)
            if digest(path) != identity["osm_sha256"]:
                raise ValueError("Archived map checksum mismatch")
            shutil.copyfile(archived_map,output/"map.osm.gz")
            identity.update(reference_directory=str(reference), original_download_seconds=identity["download_seconds"],
                            download_seconds=0, reuse_seconds=time.perf_counter()-started)
            save_json(output/"map_identity.json",identity)
            print("MAP_REUSED " + json.dumps(dict(city=city["name"],sha256=identity["osm_sha256"],
                  reference=str(reference))),flush=True)
            return path
    for attempt in range(4):
        endpoint = OVERPASS[attempt % len(OVERPASS)]
        try:
            print(json.dumps(dict(stage="map_download", city=city["name"], attempt=attempt+1)), flush=True)
            with requests.post(endpoint, data={"data": query}, headers=MAP_HEADERS,
                               stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with path.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            nodes = ways = restrictions = 0
            for _, elem in ET.iterparse(path):
                if elem.tag == "remark":
                    raise ValueError(f"Incomplete OSM response: {elem.text}")
                nodes += elem.tag == "node"
                ways += elem.tag == "way"
                restrictions += elem.tag == "relation"
                elem.clear()
            if nodes < 100 or ways < 20:
                raise ValueError("Map extract unexpectedly empty")
            with path.open("rb") as source, gzip.open(output / "map.osm.gz", "wb") as target:
                shutil.copyfileobj(source, target)
            save_json(output / "map_identity.json", dict(timestamp=timestamp, endpoint=endpoint,
                      query_sha256=hashlib.sha256(query.encode()).hexdigest(), osm_sha256=digest(path),
                      bytes=path.stat().st_size, nodes=nodes, ways=ways, restriction_relations=restrictions,
                      bbox_south_west_north_east=bbox, download_seconds=time.perf_counter()-started,
                      attribution="© OpenStreetMap contributors; ODbL 1.0", source="https://www.openstreetmap.org/copyright"))
            return path
        except (requests.RequestException, ValueError, ET.ParseError) as error:
            print(f"Map download attempt failed: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(30)
    raise AssertionError("unreachable")


@contextmanager
def routing_server(binaries: Path, profile: Path, osm: Path, output: Path, port: int = 5000):
    started = time.perf_counter()
    with (output / "osrm_preprocessing.log").open("w") as log:
        commands = [
            [str(binaries / "osrm-extract"), str(osm), "-p", str(profile), "-t", "1"],
            [str(binaries / "osrm-contract"), str(osm.with_suffix(".osrm")), "-t", "1"],
        ]
        for command in commands:
            print(f"Routing preprocessing: {Path(command[0]).name}", flush=True)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                log.flush()
                print((output / "osrm_preprocessing.log").read_text()[-12000:], flush=True)
                result.check_returncode()
    save_json(output / "routing_preprocessing.json", dict(seconds=time.perf_counter()-started,
              dataset_bytes=sum(p.stat().st_size for p in osm.parent.glob("map.osrm*"))))
    with (output / "osrm_server.log").open("w") as log:
        process = subprocess.Popen([str(binaries / "osrm-routed"), str(osm.with_suffix(".osrm")),
                                    "--algorithm", "ch", "-t", "1", "-p", str(port),
                                    "--max-table-size", "256"], stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError("OSRM server exited; see osrm_server.log")
                try:
                    requests.get(f"http://127.0.0.1:{port}/", timeout=1)
                    break
                except requests.ConnectionError:
                    time.sleep(0.25)
            else:
                raise TimeoutError("OSRM server did not become ready")
            yield f"http://127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
