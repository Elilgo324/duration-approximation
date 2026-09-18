"""Fetch public OSM geometry for the article's separate road-pattern illustrations.

Existing checked extracts are reused. The bundled extracts reproduce the figures;
new provider responses need not be byte-identical, even at the same map timestamp.
"""
import gzip
import hashlib
import json
import time
from pathlib import Path

import requests

CLASSES = ('motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|'
           'secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service')
ENDPOINTS = ('https://overpass-api.de/api/interpreter', 'https://overpass.private.coffee/api/interpreter')


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'article/map-data'
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / 'duration_approximation/protocol.json').read_text())
    for city in config['cities']:
        path = output / f'{city["name"]}.json.gz'
        identity_path = output / f'{city["name"]}_identity.json'
        if path.exists():
            identity = json.loads(identity_path.read_text())
            assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == identity['geometry_sha256']
            print('Reusing checked geometry:', city['name'], flush=True)
            continue
        west, south, east, north = city['bbox']
        query = (f'[out:json][timeout:180][date:"{config["map_timestamp"]}"];'
                 f'way["highway"~"^({CLASSES})$"]({south},{west},{north},{east});out geom;')
        errors = []
        for endpoint in ENDPOINTS:
            started = time.time()
            print('Fetching', city['name'], endpoint, flush=True)
            try:
                response = requests.post(endpoint, data={'data': query},
                    headers={'User-Agent': 'OriFogler-DurationResearch-Figures/1.0'}, timeout=(30, 210))
                response.raise_for_status()
                data = response.json()
                if data.get('remark'):
                    raise ValueError(data['remark'])
                if len(data.get('elements', [])) < 100:
                    raise ValueError('Unexpectedly small road extract')
                roads = [{'type': 'way', 'id': r['id'], 'tags': {'highway': r['tags']['highway']},
                          'geometry': r['geometry']} for r in data['elements']
                         if r['tags']['highway'] != 'service' and len(r.get('geometry', [])) > 1]
                raw = json.dumps({'elements': roads}, separators=(',', ':')).encode()
                record = {'city': city['name'],
                    'purpose': 'Road-pattern illustration; not the frozen experimental routing extract',
                    'requested_timestamp': config['map_timestamp'], 'endpoint': endpoint, 'query': query,
                    'bbox_wsen': city['bbox'], 'response_sha256': hashlib.sha256(response.content).hexdigest(),
                    'response_bytes': len(response.content), 'ways': len(data['elements']),
                    'osm3s': data.get('osm3s', {}), 'attribution': '© OpenStreetMap contributors, ODbL 1.0',
                    'source': 'https://www.openstreetmap.org/copyright',
                    'download_seconds': round(time.time() - started, 2),
                    'geometry_sha256': hashlib.sha256(raw).hexdigest(),
                    'display_ways': len(roads), 'geometry_bytes': len(raw),
                    'transformation': 'Retain way ID, highway class and geometry; omit service roads and other tags; serialize compact JSON. No coordinate simplification.'}
                with path.open('wb') as stream:
                    with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as archive:
                        archive.write(raw)
                identity_path.write_text(json.dumps(record, indent=2) + '\n')
                print('Saved', city['name'], len(roads), 'displayed ways', flush=True)
                break
            except (requests.RequestException, ValueError) as error:
                errors.append(str(error))
                print(type(error).__name__, str(error)[:250], flush=True)
        else:
            raise RuntimeError(errors)


if __name__ == '__main__':
    main()
