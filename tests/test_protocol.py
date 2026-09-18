import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import floyd_warshall

from duration_approximation.experiment import sample_pairs
from duration_approximation.learning import metrics
from duration_approximation.routing import directed_segment_ids, dp_duration, feature_row


def test_directed_landmark_hint_distinguishes_road_directions():
    import base64
    import struct
    for a,b,expected in ((0x80000005,0x80000009,(5,9)),(0x80000005,9,(5,)),(5,0x80000009,(9,))):
        hint=base64.urlsafe_b64encode(struct.pack("<II",a,b)).decode()
        assert directed_segment_ids(hint)==expected


def test_dp_is_a_directed_upper_bound_and_matches_layer_enumeration():
    rng = np.random.default_rng(42)
    graph = rng.uniform(1,100,(16,16))
    np.fill_diagonal(graph,0)
    distances = floyd_warshall(graph,directed=True)
    a,b = np.arange(1,7),np.arange(8,14)
    actual,i,j = dp_duration(distances[0,a],distances[np.ix_(a,b)],distances[b,15])
    expected = min(distances[0,u]+distances[u,v]+distances[v,15] for u in a for v in b)
    assert actual == expected
    assert actual >= distances[0,15]
    assert actual == distances[0,a[i]]+distances[a[i],b[j]]+distances[b[j],15]


def test_dp_unreachable_pairs_stay_unreachable():
    duration,_,_ = dp_duration([1,2],np.full((2,2),np.inf),[3,4])
    assert np.isinf(duration)
    result = metrics([10,20],[np.inf,25])
    assert result["coverage"] == .5
    assert result["mae_seconds"] == 5


def test_endpoint_disjoint_splits_and_same_cell_filter():
    rng = np.random.default_rng(21)
    endpoints = [{"location":[34.72+rng.random()*.23,31.95+rng.random()*.27]} for _ in range(240)]
    config = dict(resolutions=[6,7,8],train_pairs=100,validation_pairs=20,test_pairs=20)
    pairs,splits,cells,_ = sample_pairs(endpoints,config,rng)
    pools = [set(pairs[splits==s].ravel()) for s in ("train","validation","test")]
    assert all(not pools[i]&pools[j] for i in range(3) for j in range(i))
    assert len(set(map(tuple,pairs))) == len(pairs)
    assert np.all(cells[pairs[:,0]] != cells[pairs[:,1]])


def test_features_use_geometry_and_precomputed_matrix_only():
    p,q = {"location":[34.8,32.0]}, {"location":[34.9,32.1]}
    a=[{"location":[34.8+i*.001,32.01]} for i in range(6)]
    b=[{"location":[34.9+i*.001,32.11]} for i in range(6)]
    matrix=np.arange(36).reshape(6,6)
    row=feature_row(p,q,a,b,matrix)
    assert row.shape == (63,)
    np.testing.assert_array_equal(row[27:],matrix.ravel())
    assert row[26]>0


def test_fixed_experiment_scope():
    root=Path(__file__).resolve().parents[1]/"duration_approximation"
    config=json.loads((root/"protocol.json").read_text())
    assert [c["name"] for c in config["cities"]] == ["tel_aviv","amsterdam","nyc"]
    assert config["resolutions"] == [6,7,8]
    assert set(config["models"]) == {"ridge","boosted_trees","mlp"}


def landmark_waypoint(segments, location=(4.9, 52.3)):
    import base64
    import struct
    fields = [value | 0x80000000 for value in segments]
    fields += [0] * (2-len(fields))
    return dict(hint=base64.urlsafe_b64encode(struct.pack('<II', *fields)).decode(),
                location=list(location), nodes=[10,20], distance=2.0)


def test_landmark_recovers_original_segment_from_junction_tie(monkeypatch):
    from duration_approximation.routing import Router
    router = Router('unused')
    original = landmark_waypoint([5,9])
    wrong_road = landmark_waypoint([100])
    wrong_position = landmark_waypoint([5], (4.901,52.3))
    valid = landmark_waypoint([5])
    calls = []

    def nearest(service, points, **options):
        calls.append(options)
        if 'bearings' not in options:
            return dict(waypoints=[original])
        if options['number'] == 1:
            return dict(waypoints=[wrong_road])
        return dict(waypoints=[wrong_road, wrong_position, valid])

    monkeypatch.setattr(router, 'call', nearest)
    point = router.landmark([4.9,52.3], {10:[4.9,52.3],20:[4.901,52.3]})
    assert point['directed_segment'] == 5
    assert point['location'] == original['location']
    assert point['hint'] == valid['hint']
    assert calls[-1]['number'] == 8
    assert calls[-1]['radiuses'] == 3.0


def test_landmark_uses_osrm_coordinate_precision_on_short_segments(monkeypatch):
    from duration_approximation.routing import Router, bearing
    router = Router('unused')
    nodes = {10:[4.9,52.3],20:[4.9000014,52.3000004]}
    assert abs(bearing(nodes[10],nodes[20])-90) > 5

    def nearest(service, points, **options):
        if 'bearings' not in options:
            return dict(waypoints=[landmark_waypoint([5,9])])
        if options['bearings'] == '90,5':
            return dict(waypoints=[landmark_waypoint([5])])
        return dict(waypoints=[landmark_waypoint([100])])

    monkeypatch.setattr(router, 'call', nearest)
    assert router.landmark([4.9,52.3], nodes)['directed_segment'] == 5


def test_landmark_rejects_nearby_different_road_and_reports_coordinate(monkeypatch):
    import pytest
    from duration_approximation.routing import Router
    router = Router('unused')
    def nearest(service, points, **options):
        return dict(waypoints=[landmark_waypoint([100] if 'bearings' in options else [5,9])])
    monkeypatch.setattr(router, 'call', nearest)
    with pytest.raises(ValueError, match=r'coordinate.*4.9.*52.3'):
        router.landmark([4.9,52.3], {10:[4.9,52.3],20:[4.901,52.3]})


def test_one_way_landmark_preserves_osrm_hint_without_resnapping(monkeypatch):
    from duration_approximation.routing import Router
    router = Router('unused')
    original = landmark_waypoint([9])
    calls = []
    def nearest(service, points, **options):
        calls.append(options)
        return dict(waypoints=[original])
    monkeypatch.setattr(router, 'call', nearest)
    assert router.landmark([4.9,52.3], {})['hint'] == original['hint']
    assert len(calls) == 1


def test_archived_map_reuse_checks_protocol_and_content(tmp_path, monkeypatch):
    import gzip
    import hashlib
    import pytest
    import requests
    from duration_approximation.runtime import download_map
    city = dict(name='amsterdam',bbox=[4.72,52.28,5.05,52.44])
    work,output = tmp_path/'work',tmp_path/'output'
    work.mkdir(); output.mkdir()
    monkeypatch.delenv('EXPERIMENT_REFERENCE_DIR',raising=False)
    # Capture the actual frozen query without issuing a network request.
    def no_network(*args,**kwargs):
        raise RuntimeError('network disabled in test')
    monkeypatch.setattr(requests,'post',no_network)
    with pytest.raises(RuntimeError,match='network disabled'):
        download_map(city,'2026-09-17T00:00:00Z',.12,output,work)
    archive = tmp_path/'reference'/'amsterdam'
    archive.mkdir(parents=True)
    payload=b'<osm version="0.6"/>'
    (archive/'map.osm.gz').write_bytes(gzip.compress(payload))
    identity=dict(query_sha256=hashlib.sha256((output/'overpass_query.txt').read_bytes()).hexdigest(),
                  osm_sha256=hashlib.sha256(payload).hexdigest(),download_seconds=12)
    (archive/'map_identity.json').write_text(json.dumps(identity))
    monkeypatch.setenv('EXPERIMENT_REFERENCE_DIR',str(archive.parent))
    assert download_map(city,'2026-09-17T00:00:00Z',.12,output,work).read_bytes()==payload
    (archive/'map.osm.gz').write_bytes(gzip.compress(b'wrong map'))
    with pytest.raises(ValueError,match='checksum mismatch'):
        download_map(city,'2026-09-17T00:00:00Z',.12,output,work)
    with pytest.raises(ValueError,match='query does not match'):
        download_map(city,'2026-09-16T00:00:00Z',.12,output,work)


def test_table_minimizes_over_both_legal_endpoint_directions(monkeypatch):
    from duration_approximation.routing import Router
    router=Router('unused')
    points=[dict(input=[i,0]) for i in range(6)]
    monkeypatch.setattr(router,'call',lambda *args,**kwargs: dict(durations=[[107,64.8,20],[40,None,30],[9,8,None]]))
    result=router.table([dict(directions=points[:2]),points[2]],
                        [dict(directions=points[3:5]),points[5]])
    np.testing.assert_array_equal(result,[[40,20],[8,np.inf]])


def test_endpoint_direction_split_keeps_original_position_and_both_states(monkeypatch):
    from duration_approximation.routing import Router
    router=Router('unused')
    original=landmark_waypoint([5,9]);original['input']=original['location'][:]
    def nearest(service,points,**options):
        return dict(waypoints=[landmark_waypoint([5] if options['bearings']=='90,5' else [9])])
    monkeypatch.setattr(router,'call',nearest)
    states=router.endpoint_directions(original,{10:[4.9,52.3],20:[4.901,52.3]})
    assert {s['directed_segment'] for s in states}=={5,9}
    assert all(s['location']==original['location'] for s in states)
    assert all(s is not original for s in states)
    one_way=landmark_waypoint([5]);one_way['input']=one_way['location'][:]
    one_state=router.endpoint_directions(one_way,{})
    assert len(one_state)==1 and one_state[0] is not one_way


def test_route_uses_best_explicit_endpoint_direction(monkeypatch):
    from duration_approximation.routing import Router
    router=Router('unused')
    source=dict(input=[0,0]);destinations=[dict(input=[1,0]),dict(input=[2,0])]
    monkeypatch.setattr(router,'call',lambda service,points,**kwargs:
                        dict(routes=[dict(duration=107 if points[-1]['input'][0]==1 else 64.8)]))
    assert router.route([source,dict(directions=destinations)])==64.8


def test_joined_route_expands_endpoints_and_preserves_fixed_landmarks(monkeypatch):
    from duration_approximation.routing import Router
    router=Router('unused')
    sources=[dict(input=[0,0]),dict(input=[1,0])]
    destinations=[dict(input=[2,0]),dict(input=[3,0])]
    landmarks=[dict(input=[4,0]),dict(input=[5,0])]
    calls=[]
    def route(service,points,**options):
        assert points[1:-1]==landmarks
        calls.append((points[0],points[-1]))
        duration=1126.1 if points[0]==sources[1] and points[-1]==destinations[0] else 1239.5
        return dict(routes=[dict(duration=duration)])
    monkeypatch.setattr(router,'call',route)
    assert router.route([dict(directions=sources),*landmarks,dict(directions=destinations)])==1126.1
    assert len(calls)==4
