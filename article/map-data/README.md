# Road-map illustration data

These OpenStreetMap extracts are **cartographic context**, separate from the frozen OSRM input maps that produced the experiment's labels. They cover the same three endpoint study rectangles and request the protocol timestamp `2026-09-17T00:00:00Z`. No experiment was rerun to make the maps.

Each `*_identity.json` records the exact Overpass query and provider, response checksum, OSM provider metadata, source licensing, and geometry checksum. The compressed JSON retains way IDs, highway class and original geometry; service roads and unrelated tags are omitted. Coordinates are not simplified. The full provider responses are not bundled.

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). The extracted geometry is distributed under ODbL 1.0. These notices and source identity records accompany the derived data. This data license does not change the license of third-party software.

From the repository root, with `requirements-figures.txt` installed:

```sh
python -m duration_approximation.road_figures
```

This reproduces the six individual maps and `article/figures/city-road-comparison.png` without network access. All geometry hashes are checked before plotting. `figure_manifest.json` records centers, extents, projection and color classes.

The main panels show matched 12 × 12 km windows, north up, in local equirectangular coordinates. Overview maps cover the full endpoint rectangles with an orange outline around the matched window; overview scales differ. Motorway, trunk, primary and secondary roads (including links) are blue. Tertiary/link, residential, unclassified and living streets are gray. Directions, turn restrictions, access rules and water polygons are not shown. Blank space is not a water classification. These are road-geometry illustrations, not an exact display of the directed routing graph.

`python -m duration_approximation.fetch_road_maps` checks and reuses existing geometry, or fetches an extract if its file is absent. Historical timestamps do not guarantee byte-identical responses from a different server or retrieval date; retain the bundled files to reproduce these particular figures.
