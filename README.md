# From Hexagons to Driving Times

A reproducible experiment in static driving-duration approximation: H3 resolutions **6, 7, 8**, three separately trained cities (**Tel Aviv–Gush Dan, Amsterdam, NYC**), and four methods (**landmark DP, ridge regression, histogram gradient boosting, MLP**).

[Read the article](https://orifogler.com/blog/posts/from-hexagons-to-driving-times.html).

## The question

How much prediction accuracy does a coarse, precomputed representation of a road network retain? Compare error with the cost of storing and computing a dense directed landmark matrix. The DP route is deliberately conservative: concatenating legal paths through fixed directed landmarks produces an upper bound on the shortest duration under the same static routing model. The ML estimates have no such guarantee.

## Reproduce the experiment

The experiment downloads public historical OpenStreetMap extracts and checksum-verified OSRM 5.27.1 binaries and profile sources. It runs OSRM locally; it does not use a public routing demo server. No proprietary service, account, or credentials are needed.

Use Docker with Linux amd64 support (Apple Silicon requires emulation). The core experiment completed successfully on Linux and its 15 standalone tests pass locally. This extracted Docker recipe has not yet been built end-to-end; the local Docker daemon was unavailable during publication.

```sh
docker build --platform linux/amd64 -t duration-approximation .
mkdir -p outputs
docker run --rm --platform linux/amd64 --cpus=1 --memory=2g \
  -v "$PWD/outputs:/results" duration-approximation \
  --mode smoke --output /results/smoke
```

The full experiment uses `online` as its historical mode name; this is an **offline static benchmark**, not live traffic:

```sh
docker run --rm --platform linux/amd64 --cpus=1 --memory=2g \
  -v "$PWD/outputs:/results" duration-approximation \
  --mode online --output /results/full
```

Use a new output directory for every execution. The published full run took about 47 minutes on one CPU and 2 GiB RAM. Download availability and machine speed affect runtime. Allow several GB of disk for maps, OSRM preprocessing, and outputs. The smoke test covers Tel Aviv and Amsterdam with small datasets; Amsterdam still builds the full landmark coverage.

For Python-only analysis and tests (no OSRM execution required):

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-figures.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
python -m duration_approximation.figures --results results --output article/figures
```

## Frozen protocol

See [`protocol.json`](duration_approximation/protocol.json). OSM timestamp: `2026-09-17T00:00:00Z`; seed: `20260918`. Bounding boxes are `[west, south, east, north]`; the NYC rectangle includes neighboring urban areas. A 0.12-degree routing buffer permits paths outside the endpoint rectangle, but can still truncate long detours.

Per city, sample 3,000 locations approximately uniformly along eligible road length; choose positions within the middle 90% of each sampled segment. Snap within 100 m, retaining both legal endpoint directions. Split locations into 2,250 training, 375 validation, and 375 test endpoints. Draw 12,000/2,000/2,000 distinct directed pairs within those pools. Reject pairs sharing a cell at **any** tested resolution, then exclude unreachable or zero-duration labels. No endpoint appears in two splits; endpoints are reused among pairs within a split. All resolutions use the same retained pairs.

For each city/resolution, cover eligible road geometry sampled every 100 m and add a one-cell ring. Snap each H3 vertex to its nearest eligible road position with one fixed legal direction. Shared vertices and identical directed snapped positions are deduplicated. Precompute all directed landmark durations, including diagonal and within-cell entries, in a float32 matrix.

The 63 base ML features are 36 landmark-to-landmark durations; 24 east/north endpoint-to-landmark offsets; and pickup-to-drop-off east/north offset and straight-line distance. Training-only median imputation adds missingness flags; training-only standardization is applied. Exact endpoint routing legs are exclusive to DP. Each model has two fixed hyperparameter candidates, selected by validation MAE. MLP epoch selection uses validation only. No additional seeds, models, or hyperparameters were searched after inspecting test performance.

## Ground truth and validation

Ground truth is shortest driving duration in the pinned OSRM duration-weighted car profile, **not observed driving time**. Traffic is absent. Core experiment modules are unchanged from the completed run; their hashes are in [`source_manifest.json`](source_manifest.json).

The router expands ambiguous endpoints into explicit legal directions and minimizes over them. Internal landmarks remain fixed directed states, so independently calculated legs can be concatenated. Tests cover direction handling, snap consistency, disjoint splits, feature construction, cached-map identity, and the DP bound. The full experiment additionally checks every DP test prediction against ground truth and compares 50 selected DP paths per city/resolution against a joined waypoint route (450 checks total). Tolerances account for OSRM decisecond rounding and float32 matrix storage.

## Results and artifacts

The `results/` directory contains the small, machine-readable result records used for the article. Figures are generated from those records, not manually entered chart values. Generated experiment output contains endpoints, pair IDs, split IDs, ground-truth labels, features, predictions, models, all landmark matrices, routing profiles, map identities, timing records and the report. Raw matrices and maps are excluded from Git history.

Exact reruns require identical map bytes: a historical timestamp alone is not a guarantee of byte-identical responses from different Overpass providers. To reuse an existing run's map extracts, mount that run read-only and set `EXPERIMENT_REFERENCE_DIR` to its root. The runner validates the query and decompressed map checksums before use:

```sh
docker run --rm --platform linux/amd64 --cpus=1 --memory=2g \
  -v "$PWD/outputs:/results" -v "$PWD/frozen-run:/reference:ro" \
  -e EXPERIMENT_REFERENCE_DIR=/reference duration-approximation \
  --mode online --output /results/repeated
```

## Interpretation

This is a cross-cell, road-length-sampled benchmark, not a model of ride demand or all short trips. The comparison is within each city, not cross-city transfer. One fixed split and training seed do not establish statistical superiority for small model differences. Query timings include feature construction or local HTTP routing calls and exclude common snapping; they are individual-query deployment measurements, not a batched routing benchmark. Matrix storage excludes model checkpoints, landmark metadata, and the routing graph. A static upper bound does not guarantee on-time arrival in real traffic.

## Data and software attribution

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under ODbL 1.0. OSRM, its car profile and H3 are separate upstream projects with their own licenses. Third-party binaries are downloaded from immutable official artifacts and are not vendored here.
