"""Run the frozen three-city, three-resolution duration approximation protocol."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h3
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from .learning import metrics, predict, train_models
from .routing import Router, build_landmarks, dp_duration, feature_row, landmark_matrix, offsets_m, road_cells, road_geometry
from .runtime import download_map, install_osrm, routing_server, save_json


def sample_endpoints(router, segments, count, rng, radius, bbox):
    centers = (segments[:,:2] + segments[:,2:])/2
    delta = segments[:,2:]-segments[:,:2]
    lengths = np.linalg.norm(delta*np.c_[111195*np.cos(np.deg2rad(centers[:,1])),np.full(len(segments),111195)], axis=1)
    probabilities = lengths/lengths.sum()
    endpoints, seen = [], set()
    attempts = 0
    west,south,east,north = bbox
    while len(endpoints) < count and attempts < count*10:
        attempts += 1
        segment = segments[rng.choice(len(segments), p=probabilities)]
        point = router.snap(segment[:2]+rng.uniform(.05,.95)*(segment[2:]-segment[:2]), radius=radius)
        if point is None:
            continue
        lon,lat = point["location"]
        identity = (*point["location"], *point["nodes"])
        if identity not in seen and west<=lon<=east and south<=lat<=north:
            seen.add(identity)
            endpoints.append(point)
        if len(endpoints) and len(endpoints)%500 == 0:
            print(f"Sampled {len(endpoints)}/{count} endpoints",flush=True)
    if len(endpoints) != count:
        raise RuntimeError("Could not sample the required distinct routable endpoints")
    return endpoints, attempts


def sample_pairs(endpoints, config, rng):
    n = len(endpoints)
    permutation = rng.permutation(n)
    pools = np.split(permutation,[int(.75*n),int(.875*n)])
    pairs, splits, rejected = [], [], {}
    cell_ids = np.array([[h3.latlng_to_cell(p["location"][1],p["location"][0],r)
                        for r in config["resolutions"]] for p in endpoints])
    for name,pool,key in zip(("train","validation","test"),pools,("train_pairs","validation_pairs","test_pairs")):
        selected = set()
        attempts = same_cell = 0
        while len(selected) < config[key] and attempts < config[key]*100:
            attempts += 1
            p,q = map(int,rng.choice(pool,2,replace=False))
            if np.any(cell_ids[p] == cell_ids[q]):
                same_cell += 1
                continue
            selected.add((p,q))
        if len(selected) < config[key]:
            raise RuntimeError(f"Insufficient eligible pairs in {name} endpoint pool")
        chosen = sorted(selected)
        pairs.extend(chosen)
        splits.extend([name]*len(chosen))
        rejected[name] = dict(attempts=attempts,same_cell_rejected=same_cell,endpoint_ids=pool.tolist())
    return np.array(pairs),np.array(splits),cell_ids,rejected


def label_pairs(router, endpoints, pairs):
    truth = np.empty(len(pairs))
    started = time.perf_counter()
    for index,p in enumerate(np.unique(pairs[:,0])):
        indices = np.flatnonzero(pairs[:,0] == p)
        for start in range(0,len(indices),64):
            batch = indices[start:start+64]
            truth[batch] = router.table([endpoints[p]],[endpoints[q] for q in pairs[batch,1]])[0]
        if index%300 == 0:
            print(f"Ground truth: {index} pickup locations processed",flush=True)
    return truth,time.perf_counter()-started


def get_landmark_pair(p,q,resolution,endpoints,cells,landmarks):
    pc = h3.latlng_to_cell(endpoints[p]["location"][1],endpoints[p]["location"][0],resolution)
    qc = h3.latlng_to_cell(endpoints[q]["location"][1],endpoints[q]["location"][0],resolution)
    pi,qi = cells[pc],cells[qc]
    return pi,qi,[landmarks[i] for i in pi],[landmarks[i] for i in qi]


def evaluate_resolution(router,endpoints,pairs,splits,truth,resolution,cells,landmarks,matrix,config,output,smoke):
    features = []
    for p,q in pairs:
        pi,qi,pl,ql = get_landmark_pair(p,q,resolution,endpoints,cells,landmarks)
        features.append(feature_row(endpoints[p],endpoints[q],pl,ql,matrix[np.ix_(pi,qi)]))
    x = np.array(features)
    np.savez_compressed(output / "features.npz",features=x)
    rows,models = train_models(x,truth,splits,config["models"],config["seed"],output,smoke)
    test = np.flatnonzero(splits == "test")
    dp_predictions,dp_ms,route_ms,ml_ms = [],[],[],{name:[] for name in models}
    checks = []
    for iteration,k in enumerate(test):
        p,q = pairs[k]
        started = time.perf_counter()
        pi,qi,pl,ql = get_landmark_pair(p,q,resolution,endpoints,cells,landmarks)
        middle = matrix[np.ix_(pi,qi)]
        first = router.table([endpoints[p]],pl)[0]
        last = router.table(ql,[endpoints[q]])[:,0]
        duration,i,j = dp_duration(first,middle,last)
        dp_ms.append((time.perf_counter()-started)*1000)
        dp_predictions.append(duration)
        if duration < truth[k]-.31:
            raise AssertionError(f"Upper-bound violation: DP {duration}, truth {truth[k]}")
        if iteration < (20 if smoke else 50) and np.isfinite(duration):
            joined = router.route([endpoints[p],pl[i],ql[j],endpoints[q]])
            if not np.isfinite(joined) or abs(joined-duration)>.31:
                points = [endpoints[p],pl[i],ql[j],endpoints[q]]
                routes = [router.call("route", leg, overview="false", steps="false",
                          annotations="nodes", continue_straight="true")
                          for leg in (points, points[:2], points[1:3], points[2:])]
                diagnostic = dict(pair_index=int(k),points=points,dp=duration,truth=float(truth[k]),
                    table_legs=[float(first[i]),float(middle[i,j]),float(last[j])],routes=routes)
                save_json(output/"route_mismatch.json",diagnostic)
                print("ROUTE_MISMATCH " + json.dumps(dict(diagnostic, routes=[dict(
                    waypoints=r["waypoints"],routes=[{key:value for key,value in route.items() if key!="legs"}
                    for route in r["routes"]]) if r else None for r in routes])),flush=True)
                raise AssertionError(f"DP legs do not join: sum={duration}, waypoint route={joined}")
            checks.append(dict(pair_index=int(k),dp=duration,joined=joined))
        if iteration < (10 if smoke else 200):
            started = time.perf_counter()
            direct = router.route([endpoints[p],endpoints[q]])
            route_ms.append((time.perf_counter()-started)*1000)
            if abs(direct-truth[k])>.11:
                raise AssertionError("Route and table disagree on ground truth")
            for name,model in models.items():
                started = time.perf_counter()
                pi,qi,pl,ql = get_landmark_pair(p,q,resolution,endpoints,cells,landmarks)
                row = feature_row(endpoints[p],endpoints[q],pl,ql,matrix[np.ix_(pi,qi)])
                predict(model,row[None,:])
                ml_ms[name].append((time.perf_counter()-started)*1000)
        if iteration%500 == 0:
            print(f"DP resolution {resolution}: {iteration}/{len(test)} test trips",flush=True)
    save_json(output / "dp_route_checks.json",checks)
    np.save(output / "dp_test_predictions.npy",dp_predictions)
    dp_row = dict(method="dp",**metrics(truth[test],dp_predictions),query_ms_mean=float(np.mean(dp_ms)),query_ms_p95=float(np.quantile(dp_ms,.95)))
    rows.append(dp_row)
    for row in rows:
        if row["method"] in ml_ms:
            latencies = ml_ms[row["method"]]
            row.update(query_ms_mean=float(np.mean(latencies)),query_ms_p95=float(np.quantile(latencies,.95)))
        row["resolution"] = resolution
    save_json(output / "query_timings.json",dict(dp_ms=dp_ms,direct_route_ms=route_ms,ml_ms=ml_ms,
        scope="Already-snapped endpoints. ML includes H3 lookup, matrix gather, geometry features and model inference. DP includes two local HTTP table calls and minimum. Direct route minimizes up to four local HTTP calls over explicit endpoint directions. Common snapping and direction resolution excluded."))
    for row in rows:
        prediction = np.array(dp_predictions) if row["method"] == "dp" else np.load(output/f"{row['method']}_test_predictions.npy")
        row["by_truth_duration"] = {}
        for label,low,high in (("under_10_min",0,600),("10_to_20_min",600,1200),("over_20_min",1200,np.inf)):
            mask = (truth[test]>=low)&(truth[test]<high)
            if mask.any():
                row["by_truth_duration"][label] = metrics(truth[test][mask],prediction[mask])
    save_json(output / "metrics.json",rows)
    return rows


def write_report(output,rows,smoke):
    text = ["# Duration approximation " + ("SMOKE TEST — not research results" if smoke else "experiment v6"), "",
        "Ground truth: shortest driving duration under the archived OSRM 5.27.1 duration profile and historical OSM extract. No traffic.",
        "Cities trained independently; identical endpoint-disjoint train/validation/test splits across resolutions 6/7/8. Only trips in different cells at all resolutions are evaluated.",
        "Landmarks use the nearest eligible segment and one fixed legal direction. Models receive 36 precomputed durations plus 27 geometry features; no exact endpoint-leg durations.",
        "", "| City | H3 | Method | MAE (s) | P95 absolute (s) | Bias (s) | Coverage | Query mean (ms) |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        def fmt(key):
            value=row.get(key)
            return "n/a" if value is None else f"{value:.2f}"
        text.append(f"| {row['city']} | {row['resolution']} | {row['method']} | {fmt('mae_seconds')} | {fmt('p95_absolute_seconds')} | {fmt('bias_seconds')} | {fmt('coverage')} | {fmt('query_ms_mean')} |")
    text += ["", "## Interpretation limits", "",
        "Evaluation rectangles are explicit study regions, not administrative city boundaries. NYC's rectangle includes nearby urban roads. Sampling is proportional to eligible road-segment length, not real demand.",
        "Exact durations are exact only for this frozen routing model and buffered extract; map edges can truncate longer detours. Same-cell trips and unreachable ground-truth trips are excluded and counted.",
        "DP errors are conditional on finite landmark connections; coverage is reported. ML missing landmark durations are imputed from training data with indicators.",
        "Latency excludes common endpoint snapping. It includes feature construction for ML and endpoint table requests for DP. Direct routing minimizes up to four HTTP requests over endpoint direction states; DP uses two expanded table requests. Local HTTP overhead is included, so these are deployment measurements, not pure algorithm timings.",
        "Two predefined hyperparameter candidates per model; validation MAE selects candidates and MLP epoch. Test data never selects settings. One fixed training seed; no claim of universal model superiority.",
        "All preprocessing and training costs are separate artifacts. Dense matrices include within-cell/diagonal entries for simple indexing; count their complete storage cost.",
        "", "Map data © OpenStreetMap contributors, ODbL 1.0. https://www.openstreetmap.org/copyright", ""]
    (output/"REPORT.md").write_text("\n".join(text))
    save_json(output/"metrics.json",rows)


def run_city(city,config,output,work,binaries,profile,smoke):
    city_output=output/city["name"]
    city_output.mkdir()
    city_work=work/city["name"]
    city_work.mkdir()
    osm=download_map(city,config["map_timestamp"],config["routing_buffer_deg"],city_output,city_work)
    nodes,segments=road_geometry(osm,city["bbox"])
    with routing_server(binaries,profile,osm,city_output) as url:
        router=Router(url)
        if city["name"] == "tel_aviv":
            regression_points = [router.snap([34.790688,31.967638]),
                router.landmark([34.798917,31.970306],nodes),
                router.landmark([34.814597,32.006706],nodes),
                router.snap([34.814739,32.006740])]
            for endpoint in (regression_points[0],regression_points[-1]):
                endpoint["directions"] = router.endpoint_directions(endpoint,nodes)
            separate = [router.route(regression_points[k:k+2]) for k in range(3)]
            table_legs = [float(router.table([regression_points[k]],[regression_points[k+1]])[0,0]) for k in range(3)]
            joined = router.route(regression_points)
            regression = dict(points=regression_points,separate=separate,table_legs=table_legs,joined=joined)
            save_json(city_output/"same_segment_regression.json",regression)
            print("SAME_SEGMENT_REGRESSION " + json.dumps(regression),flush=True)
            if (not np.all(np.isfinite(separate)) or not np.isfinite(joined) or abs(sum(separate)-joined)>.31
                    or not np.allclose(separate,table_legs,rtol=0,atol=.11)):
                raise AssertionError("Known same-segment routing regression remains unresolved")
        rng=np.random.default_rng(config["seed"])
        endpoints,attempts=sample_endpoints(router,segments,config["endpoint_pool"],rng,config["max_endpoint_snap_m"],city["bbox"])
        for endpoint in endpoints:
            endpoint["directions"] = router.endpoint_directions(endpoint,nodes)
        pairs,splits,cell_ids,rejections=sample_pairs(endpoints,config,rng)
        truth,label_seconds=label_pairs(router,endpoints,pairs)
        finite=np.isfinite(truth)&(truth>0)
        save_json(city_output/"endpoints.json",endpoints)
        save_json(city_output/"sampling.json",dict(endpoint_attempts=attempts,splits=rejections,
                  unreachable_or_zero_pairs=int((~finite).sum()),requested_pairs=len(pairs),label_seconds=label_seconds,
                  retained_by_split={s:int(((splits==s)&finite).sum()) for s in ("train","validation","test")}))
        pairs,splits,truth=pairs[finite],splits[finite],truth[finite]
        if any(np.count_nonzero(splits==s)<10 for s in ("train","validation","test")):
            raise ValueError("Too few reachable trips in a split")
        np.savez_compressed(city_output/"dataset.npz",pairs=pairs,splits=splits,truth_seconds=truth,endpoint_cells=cell_ids)
        if smoke and city["name"] != "amsterdam":
            cells_by_resolution={r:sorted(set(cell_ids[:,i])) for i,r in enumerate(config["resolutions"])}
        else:
            cells_by_resolution=road_cells(segments,config["resolutions"])
            for i,r in enumerate(config["resolutions"]):
                cells_by_resolution[r]=sorted(set(cells_by_resolution[r])|set(cell_ids[:,i]))
        rows=[]
        for resolution in config["resolutions"]:
            print(f"City {city['name']} resolution {resolution}",flush=True)
            resolution_output=city_output/f"h3_{resolution}"
            resolution_output.mkdir()
            landmarks,cells,snap_seconds=build_landmarks(router,cells_by_resolution[resolution],nodes,resolution_output)
            matrix,matrix_seconds=landmark_matrix(router,landmarks,resolution_output,config["table_block_size"])
            save_json(resolution_output/"precomputation.json",dict(cells=len(cells),landmarks=len(landmarks),
                matrix_entries=int(matrix.size),matrix_payload_bytes=int(matrix.nbytes),landmark_seconds=snap_seconds,
                matrix_seconds=matrix_seconds,unreachable_entries=int(np.isinf(matrix).sum())))
            city_rows=evaluate_resolution(router,endpoints,pairs,splits,truth,resolution,cells,landmarks,matrix,config,resolution_output,smoke)
            for row in city_rows:
                row["city"]=city["name"]
                row["landmarks"]=len(landmarks)
                row["matrix_bytes"]=int(matrix.nbytes)
            rows.extend(city_rows)
            print("RESOLUTION_AUDIT " + json.dumps(dict(city=city["name"], resolution=resolution,
                  precomputation=json.loads((resolution_output/"precomputation.json").read_text()),
                  joined_route_checks=len(json.loads((resolution_output/"dp_route_checks.json").read_text())),
                  metrics=city_rows), allow_nan=False), flush=True)
            del matrix
        return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("smoke","online"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(__file__).with_name("protocol.json").read_text())
    smoke=args.mode=="smoke"
    if smoke:
        config.update(train_pairs=120,validation_pairs=20,test_pairs=20,endpoint_pool=120)
        # Amsterdam uses the full road-derived landmark coverage even in smoke:
        # endpoint-only coverage missed the junction failure in build 19.
        config["cities"]=config["cities"][:2]
    save_json(output/"protocol.json",config)
    save_json(output/"runtime.json",dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
              sklearn=sklearn.__version__,h3=h3.__version__,mode=args.mode))
    rows=[]
    smoke_failures=[]
    started=time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="duration_approximation_") as temporary, threadpool_limits(limits=1):
            work=Path(temporary)
            binaries,profile=install_osrm(work/"osrm",output)
            for city in config["cities"]:
                try:
                    rows.extend(run_city(city,config,output,work,binaries,profile,smoke))
                except AssertionError as error:
                    if not smoke:
                        raise
                    smoke_failures.append(dict(city=city["name"],error=str(error)))
                    print("SMOKE_CITY_FAILURE " + json.dumps(smoke_failures[-1]),flush=True)
                write_report(output,rows,smoke)
                shutil.rmtree(work/city["name"])
            if smoke_failures:
                save_json(output/"smoke_failures.json",smoke_failures)
                raise AssertionError("Smoke validation failed; see smoke_failures.json and route_mismatch.json")
        completion=dict(complete=True,mode=args.mode,rows=len(rows),seconds=time.perf_counter()-started)
        save_json(output/"completion.json",completion)
        print((output/"REPORT.md").read_text(), flush=True)
        print("COMPLETION " + json.dumps(completion), flush=True)
    except Exception as error:
        save_json(output/"completion.json",dict(complete=False,mode=args.mode,error=repr(error),seconds=time.perf_counter()-started))
        raise


if __name__ == "__main__":
    main()
