"""Tests for the pano-position check and the reposition tool (SidewalkWebpage#5361):
offsets against a synthetic street grid, the per-sequence verdict, the position-field
switch in the Mapillary record builder, and the run-dir binding to one field."""
import json

import pytest

import geo
import main
import position_check
import reposition
import send_to_ps
from sources import mapillary
from test_sources_mapillary import make_meta

LAT0, LNG0 = 42.85, -94.85


def _frame():
    return geo.LocalFrame(LAT0, LNG0)


def _grid_osm(frame, spacing_m=100.0, count=3):
    """A count x count grid of named N-S and E-W streets around the frame origin."""
    elements = []
    lo, hi = -spacing_m * (count - 1) / 2, spacing_m * (count - 1) / 2
    for i in range(count):
        c = lo + i * spacing_m
        (lat_a, lng_a), (lat_b, lng_b) = frame.to_latlng(c, lo), frame.to_latlng(c, hi)
        elements.append({"type": "way", "tags": {"name": f"NS{i}"},
                         "geometry": [{"lat": lat_a, "lon": lng_a}, {"lat": lat_b, "lon": lng_b}]})
        (lat_a, lng_a), (lat_b, lng_b) = frame.to_latlng(lo, c), frame.to_latlng(hi, c)
        elements.append({"type": "way", "tags": {"name": f"EW{i}"},
                         "geometry": [{"lat": lat_a, "lon": lng_a}, {"lat": lat_b, "lon": lng_b}]})
    return {"elements": elements}


def test_measure_point_signs_and_axes():
    frame = _frame()
    index = position_check.StreetIndex(position_check.street_segments(_grid_osm(frame), frame))
    # A cardinal grid: family a is N-S (positive east), family b is E-W (positive north).
    assert index.theta0 == pytest.approx(0.0, abs=1e-6)
    assert index.axes()["a"] == {"streets": "N-S streets", "pos": "east", "neg": "west"}
    # 5 m east of the central N-S street (x=0), far from any E-W street.
    m = position_check.measure_point(index, 5.0, 40.0)
    assert m["street"] == "NS1"
    assert m["a"] == pytest.approx(5.0) and m["b"] is None and m["cross"] == pytest.approx(5.0)
    # 3 m south of the central E-W street (y=0).
    m = position_check.measure_point(index, 40.0, -3.0)
    assert m["street"] == "EW1"
    assert m["b"] == pytest.approx(-3.0) and m["a"] is None
    # Nothing within 30 m of a street: not scored.
    assert position_check.measure_point(index, 1000.0, 1000.0) is None


def _mapillary_record(pano_id, seq, raw_en, sfm_en, frame):
    raw_lat, raw_lng = frame.to_latlng(*raw_en)
    sfm_lat, sfm_lng = frame.to_latlng(*sfm_en)
    meta = make_meta(sequence=seq,
                     geometry={"type": "Point", "coordinates": [raw_lng, raw_lat]},
                     computed_geometry={"type": "Point", "coordinates": [sfm_lng, sfm_lat]})
    return {"detections": [], "pano": mapillary.build_pano_record(pano_id, 0.0, 0.0, meta)}


MANIFEST = {"run_name": "synthetic", "imagery_source": "mapillary"}
AREA = {"type": "Polygon", "coordinates": [[[LNG0 - 0.01, LAT0 - 0.01], [LNG0 + 0.01, LAT0 - 0.01],
                                             [LNG0 + 0.01, LAT0 + 0.01], [LNG0 - 0.01, LAT0 + 0.01],
                                             [LNG0 - 0.01, LAT0 - 0.01]]]}


def _northbound(seq, raw_x, sfm_x, frame, n=25):
    """A sequence driving north along NS1 (x=0) with each field at a fixed east offset."""
    return [_mapillary_record(f"{seq}{k}", seq, (raw_x, -80 + k * 6), (sfm_x, -80 + k * 6), frame)
            for k in range(n)]


def _offsets(seq, raw_xs, sfm_xs, frame):
    """A sequence driving north on NS1 (x=0) with a per-pano east offset for each field, on
    the block between EW1 (y=0) and EW2 (y=100) so no pano snaps to a cross street."""
    return [_mapillary_record(f"{seq}{k}", seq, (rx, 8 + 3 * k), (sx, 8 + 3 * k), frame)
            for k, (rx, sx) in enumerate(zip(raw_xs, sfm_xs))]


def _check(records, frame, **kw):
    return position_check.check_run(MANIFEST, AREA, records, _grid_osm(frame), **kw)


def test_check_run_flags_the_drifted_sequence_and_recommends_raw():
    frame = _frame()
    # Sequence A drives north on NS1: raw GPS on the line, SfM 8 m west of it.
    records = _northbound("A", 0.5, -7.5, frame)
    # Sequence B drives east on EW1: both fields within a lane of the line.
    for k in range(30):
        x = -90 + k * 6
        records.append(_mapillary_record(f"b{k}", "B", (x, 1.0), (x, -1.0), frame))
    result = _check(records, frame)

    assert result["submitted_field"] == "sfm"  # build_pano_record wrote computed_geometry
    assert result["flagged_sequences"] == ["A"]
    by_id = {s["sequence_id"]: s for s in result["sequences"]}
    assert by_id["A"]["recommended"] == "raw" and by_id["A"]["flagged"]
    assert by_id["A"]["sfm_minus_raw_east_m"] == pytest.approx(-8.0, abs=0.05)
    assert not by_id["B"]["flagged"]
    assert by_id["A"]["bias_m"]["sfm"]["a"] == pytest.approx(-7.5, abs=0.05)
    # Whole-run across-N-S summary sees A's SfM positions 7.5 m west of the line.
    assert result["fields"]["sfm"]["across_a"]["median"] == pytest.approx(-7.5, abs=0.05)
    assert result["fields"]["raw"]["across_a"]["median"] == pytest.approx(0.5, abs=0.05)
    assert result["fields"]["raw"]["across_a"]["neg_share"] == 0.0
    assert by_id["A"]["submitted_field"] == "sfm" and not by_id["A"]["beyond_snap"]

    # The check must be able to confirm its own fix: after reposition.py moves A to raw
    # the file is mixed by design (A on raw, B still on SfM), and the verdict has to be
    # on what each sequence actually submitted — not on the run-wide majority field.
    fixed = [{**rec, "pano": reposition.reposition_pano(rec["pano"], "raw")} if rec["pano"]["sequence_id"] == "A"
             else rec for rec in records]
    again = _check(fixed, frame)
    assert again["flagged_sequences"] == [] and again["both_off_sequences"] == []
    again_by_id = {s["sequence_id"]: s for s in again["sequences"]}
    assert again_by_id["A"]["submitted_field"] == "raw" and again_by_id["A"]["recommended"] == "raw"
    assert again_by_id["B"]["submitted_field"] == "sfm"
    assert again["submitted_field"] == "sfm"  # the run-wide majority, kept for the report


def test_alternating_lane_offsets_flag_on_the_paired_metric_not_the_signed_bias():
    """SfM 6 m off on every pano, alternately east and west (a lane offset on a street
    driven both ways): the signed bias cancels to ~0 and #60's rule saw nothing, while every
    label sits 6 m off. Raw is uniformly 0.5 m off. The paired metric sees 5.5 m per pano."""
    frame = _frame()
    sfm = [6.0 if k % 2 else -6.0 for k in range(24)]
    result = _check(_offsets("A", [0.5] * 24, sfm, frame), frame)
    row = result["sequences"][0]
    assert row["bias_m"]["sfm"]["max_abs"] < position_check.RESOLUTION_FLOOR_M  # the old rule's blind spot
    assert row["cross_track_median_m"]["sfm"] == pytest.approx(6.0, abs=0.05)
    assert row["paired_median_m"] == pytest.approx(5.5, abs=0.05) and row["raw_closer_share"] == 1.0
    assert row["paired_verdict"] == "raw" and row["flagged"] and row["recommended"] == "raw"
    assert result["flagged_sequences"] == ["A"]


def test_a_closer_but_twice_as_scattered_field_is_not_a_fix():
    """Jitter is per-pano error too: raw is 3 m closer at the median but its cross-track IQR
    is 2x SfM's (> MAX_IQR_RATIO), so the sequence is reported off the street, not flagged."""
    frame = _frame()
    sfm = [5.0 if k % 2 else 7.0 for k in range(24)]      # median 6, IQR 2
    raw = [1.0 if k % 2 else 5.0 for k in range(24)]      # median 3, IQR 4
    row = _check(_offsets("A", raw, sfm, frame), frame)["sequences"][0]
    assert row["cross_track_iqr_m"] == {"submitted": 2.0, "sfm": 2.0, "raw": 4.0}
    assert row["paired_verdict"] == "raw"                  # closer, pano by pano...
    assert row["both_off"] and not row["flagged"]          # ...but not a fix
    assert row["recommended"] == "sfm"


def test_an_alternative_that_is_worse_is_never_recommended():
    """Richmond-shaped (jKtaJMek7wQl5AOH28qdcm): the submitted field is grossly off, the
    other is further off still. Reported both_off; never flagged toward the worse field."""
    frame = _frame()
    result = _check(_offsets("A", [9.0] * 24, [6.0] * 24, frame), frame)
    row = result["sequences"][0]
    assert row["paired_median_m"] == pytest.approx(-3.0, abs=0.05) and row["paired_verdict"] == "sfm"
    assert row["both_off"] and not row["flagged"] and row["recommended"] == "sfm"
    assert result["flagged_sequences"] == [] and result["both_off_sequences"] == ["A"]


def test_a_difference_inside_the_resolution_floor_is_undecidable_and_never_gates():
    """1.5 m per pano is inside the ~1.75 m the OSM reference floors at: whichever field
    is 'closer', the check cannot see it, so it reports and does not act."""
    frame = _frame()
    # Submitted 3 m off: not off the street at all; the 1.5 m difference is just reported.
    row = _check(_offsets("A", [1.5] * 24, [3.0] * 24, frame), frame)["sequences"][0]
    assert row["undecidable"] and row["paired_verdict"] == "undecidable"
    assert not row["off_street"] and not row["flagged"] and not row["both_off"]
    # Submitted 6 m off: off the street, but a 1.5 m "fix" is not one the check can vouch for.
    result = _check(_offsets("A", [4.5] * 24, [6.0] * 24, frame), frame)
    row = result["sequences"][0]
    assert row["undecidable"] and row["off_street"] and row["both_off"] and not row["flagged"]
    assert result["paired"]["undecidable"] == 1 and result["paired"]["share_below_floor"] == 1.0
    # ...while the same 6 m with raw 2.8 m closer (> the floor) is a fix.
    assert _check(_offsets("A", [3.2] * 24, [6.0] * 24, frame), frame)["flagged_sequences"] == ["A"]


def test_a_sequence_beyond_the_snap_cap_is_flagged_not_passed():
    frame = _frame()
    # SfM 45 m east of NS1 (and 55 m from NS2): nothing within MAX_SNAP_M, so it has no
    # bias at all — the worst drift must not read as clean. Raw sits on the line.
    result = _check(_northbound("A", 0.5, 45.0, frame), frame)
    row = result["sequences"][0]
    assert row["beyond_snap"] and row["flagged"] and row["recommended"] == "raw"
    # The 12 panos passing the E-W cross streets still snap (to them), so the sequence does
    # have a bias; it is the 13 that snap to nothing that carry the signal.
    assert row["not_near_a_street"] == {"submitted": 13, "sfm": 13, "raw": 0}
    assert result["flagged_sequences"] == ["A"]
    # Both fields beyond the cap: reported as both_off, no recommendation to act on.
    result = _check(_northbound("A", 45.0, 45.0, frame), frame)
    assert result["flagged_sequences"] == [] and result["both_off_sequences"] == ["A"]


def test_partial_overpass_answers_are_refused_and_never_cached(tmp_path, monkeypatch):
    good = {"elements": [{"type": "way", "geometry": [{"lat": LAT0, "lon": LNG0}], "tags": {}}]}
    assert position_check.validate_osm_payload(good) is None
    # Overpass reports a timeout or memory exhaustion as HTTP 200 + `remark`, with whatever
    # elements it had produced so far.
    assert "remark" in position_check.validate_osm_payload({**good, "remark": "runtime error: Query timed out"})
    assert position_check.validate_osm_payload({"elements": []}) == "zero streets returned"
    assert position_check.validate_osm_payload({"version": 0.6}) is not None

    # A cached bad answer is refetched and replaced, not reused.
    cache = tmp_path / "osm_streets.json"
    cache.write_text(json.dumps({"elements": [], "remark": "runtime error"}), encoding="utf-8")
    monkeypatch.setattr(position_check, "fetch_osm_streets", lambda bbox: dict(good))
    payload, path, cached = position_check.load_or_fetch_streets(tmp_path, AREA)
    assert not cached and path == cache
    assert position_check.validate_osm_payload(json.load(open(cache, encoding="utf-8"))) is None
    # A good cache built with the current highway filter is reused as-is.
    payload, path, cached = position_check.load_or_fetch_streets(tmp_path, AREA)
    assert cached and payload["_query"]["highway"] == position_check.STREET_HIGHWAY_RE
    # An explicit --osm file the user supplied is refused rather than overwritten.
    theirs = tmp_path / "theirs.json"
    theirs.write_text(json.dumps({"elements": []}), encoding="utf-8")
    with pytest.raises(SystemExit):
        position_check.load_or_fetch_streets(tmp_path, AREA, theirs)


def test_report_geometry_helpers():
    w = (0.0, 0.0, 800.0, 400.0)
    # A long straight crossing the window with both endpoints far outside is still drawn.
    assert position_check._segment_hits_window((-500.0, 200.0), (1300.0, 200.0), w, 60)
    assert not position_check._segment_hits_window((-500.0, 600.0), (1300.0, 600.0), w, 60)
    assert position_check._segment_hits_window((100.0, 100.0), (100.0, 100.0), w, 60)  # degenerate, inside
    # The densest 800 x 400 box may be anchored on an empty cell: two clusters 700 m apart
    # fit in one window only when its SW corner sits in the empty ground before them.
    pts = [(50.0, 350.0)] * 5 + [(750.0, 50.0)] * 5
    x0, y0, x1, y1 = position_check.densest_window(pts)
    assert x0 <= 50 and x1 >= 750 and y0 <= 50 and y1 >= 350


def test_reposition_pano_switches_field_and_stays_submittable():
    meta = make_meta(geometry={"type": "Point", "coordinates": [-77.4300, 37.5400]})
    pano = mapillary.build_pano_record("img-1", 0.0, 0.0, meta)
    assert (pano["lng"], pano["lat"]) == (-77.4360, 37.5407)  # SfM by default

    moved = reposition.reposition_pano(pano, "raw")
    assert (moved["lng"], moved["lat"]) == (-77.4300, 37.5400)
    assert moved["position_field"] == "geometry"
    assert pano["lat"] == 37.5407  # input untouched
    assert reposition.reposition_pano(moved, "sfm")["lat"] == 37.5407  # round trip

    # The extra key rides along to PS like every other provenance field.
    payload = send_to_ps.transform_record({"detections": [], "pano": moved})
    assert payload["pano"]["position_field"] == "geometry"
    assert payload["pano"]["lat"] == 37.5400

    # Non-Mapillary blocks are refused rather than guessed at.
    assert reposition.reposition_pano({"source": "gsv", "lat": 1, "lng": 2}, "raw") is None


def test_build_pano_record_honours_position_field(monkeypatch):
    meta = make_meta(geometry={"type": "Point", "coordinates": [-77.4300, 37.5400]})
    monkeypatch.setattr(mapillary, "POSITION_FIELD", "raw")
    pano = mapillary.build_pano_record("img-1", 0.0, 0.0, meta)
    assert (pano["lng"], pano["lat"]) == (-77.4300, 37.5400)
    # An unreconstructed image (no computed_geometry) falls back to the GPS fix.
    monkeypatch.setattr(mapillary, "POSITION_FIELD", "sfm")
    pano = mapillary.build_pano_record("img-2", 0.0, 0.0, make_meta(computed_geometry=None,
                                        geometry={"type": "Point", "coordinates": [-77.4300, 37.5400]}))
    assert (pano["lng"], pano["lat"]) == (-77.4300, 37.5400)


def test_run_dir_is_bound_to_one_position_field(tmp_path):
    geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    run_dir = tmp_path / "city"
    manifest = main.load_or_init_run_dir(run_dir, "city.geojson", geom, "hash", "mapillary", position_field="raw")
    assert manifest["mapillary_position"] == "raw"
    assert json.load(open(run_dir / "manifest.json"))["mapillary_position"] == "raw"
    main.load_or_init_run_dir(run_dir, "city.geojson", geom, "hash", "mapillary", position_field="raw")
    with pytest.raises(SystemExit):
        main.load_or_init_run_dir(run_dir, "city.geojson", geom, "hash", "mapillary", position_field="sfm")
    # A manifest from before the flag is an SfM run.
    del manifest["mapillary_position"]
    main.save_manifest(run_dir / "manifest.json", manifest)
    main.load_or_init_run_dir(run_dir, "city.geojson", geom, "hash", "mapillary", position_field="sfm")


def test_beyond_snap_with_an_off_street_alternative_still_flags():
    frame = _frame()
    # SfM beyond the cap for most of the sequence, raw 3.5 m off (over the threshold but
    # tens of metres better): that is a fix, not "both off".
    result = _check(_northbound("A", 3.5, 45.0, frame), frame)
    row = result["sequences"][0]
    assert row["beyond_snap"] and row["flagged"] and row["recommended"] == "raw"


def test_truncated_street_cache_is_refetched(tmp_path, monkeypatch):
    good = {"elements": [{"type": "way", "geometry": [{"lat": LAT0, "lon": LNG0}], "tags": {}}]}
    cache = tmp_path / "osm_streets.json"
    cache.write_text('{"elements": [{"type": "way", "geo', encoding="utf-8")  # Ctrl-C mid-write
    monkeypatch.setattr(position_check, "fetch_osm_streets", lambda bbox: dict(good))
    _payload, _path, cached = position_check.load_or_fetch_streets(tmp_path, AREA)
    assert not cached and json.load(open(cache, encoding="utf-8"))["_query"]["highway"]
    assert not list(tmp_path.glob("*.tmp"))


def _write_run(run_dir, records, frame):
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({**MANIFEST, "run_name": run_dir.name}), encoding="utf-8")
    (run_dir / "area.geojson").write_text(json.dumps(AREA), encoding="utf-8")
    (run_dir / "osm_streets.json").write_text(json.dumps({**_grid_osm(frame), "_query": {
        "highway": position_check.STREET_HIGHWAY_RE}}), encoding="utf-8")
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_cli_loop_check_reposition_recheck(tmp_path):
    """The whole offline loop: flag -> reposition -> confirm the output in place, with the
    outputs of the confirmation written beside the .check file, never over the run's own."""
    frame = _frame()
    run = tmp_path / "city"
    _write_run(run, _northbound("A", 0.5, -7.5, frame), frame)
    assert position_check.main([str(run)]) == 1
    verdict = (run / "position_check.json").read_bytes()

    assert reposition.main([str(run / "results.jsonl"), "--from-check"]) == 0
    check = run / "results.check.jsonl"
    assert check.exists()
    assert position_check.main([str(run), "--results", str(check)]) == 0
    assert (run / "results.check.position_check.json").exists()
    assert (run / "position_check.json").read_bytes() == verdict  # the run's own verdict is untouched

    # A bare --from-check on the .check file reads the verdict written beside it.
    assert reposition.main([str(check), "--from-check"]) == 0  # "nothing to do": nothing flagged there

    # reposition.py never writes a results.jsonl, whatever the source...
    with pytest.raises(SystemExit):
        reposition.main([str(check), "--field", "sfm", "--out", str(run / "results.jsonl")])
    # ...and another run's results.jsonl is refused rather than scored against this area
    # and written over that run's tracked verdict.
    other = tmp_path / "other"
    _write_run(other, _northbound("B", 0.5, 0.5, frame), frame)
    with pytest.raises(SystemExit):
        position_check.main([str(run), "--results", str(other / "results.jsonl")])
    assert not (other / "position_check.json").exists()


def test_run_check_records_the_results_hash_and_names_outputs_by_file(tmp_path):
    import hashlib
    frame = _frame()
    run = tmp_path / "city"
    _write_run(run, _northbound("A", 0.5, 0.5, frame), frame)
    result = position_check.run_check(run, report=False)
    assert result["results_sha256"] == hashlib.sha256((run / "results.jsonl").read_bytes()).hexdigest()
    assert json.load(open(run / "position_check.json"))["results_sha256"] == result["results_sha256"]
    assert not (run / "position_report.html").exists()
    assert position_check.check_path_for(run / "results.jsonl") == run / "position_check.json"
    assert position_check.check_path_for(run / "results.check.jsonl") == run / "results.check.position_check.json"
    assert position_check.report_path_for(run / "results.check.jsonl") == run / "results.check.position_report.html"
    assert position_check.load_check(run / "results.check.jsonl")[0] is None
    assert position_check.load_check(run / "results.jsonl")[0]["results_sha256"] == result["results_sha256"]


def test_reposition_refuses_a_file_that_is_already_live(tmp_path, capsys):
    """Frame consistency (issue #62): PS upserts the pano row, so repositioning a submitted
    file moves labels that are live. reposition.py refuses, naming what would move, unless
    the whole-city decision is typed out; a rewrite that moves nothing is not refused."""
    frame = _frame()
    run = tmp_path / "city"
    _write_run(run, _northbound("A", 0.5, -7.5, frame), frame)
    results = run / "results.jsonl"
    record = {"input_file": "results.jsonl", "sha256": position_check.file_sha256(results),
              "total_lines": 25, "endpoints": {"https://ps.example/ai/submitLabelsOnPano": {
                  "submitted_lines": 25, "labels_submitted": 40, "min_confidence": 0.3}}}
    position_check.submission_record_for(results).write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(SystemExit, match=r"already submitted.*25 lines, 40 labels.*move 25 pano"):
        reposition.main([str(results), "--field", "raw"])
    assert not (run / "results.raw.jsonl").exists() and not list(run.glob("*.tmp"))

    # The same file on the field it already carries moves nothing: nothing to protect.
    assert reposition.main([str(results), "--field", "sfm"]) == 0

    assert reposition.main([str(results), "--field", "raw", "--reposition-live-city"]) == 0
    assert (run / "results.raw.jsonl").exists()
    assert "WARNING (--reposition-live-city)" in capsys.readouterr().out

    # The check itself reports the live campaign beside its verdict.
    result = position_check.run_check(run, report=False)
    assert result["live_campaigns"] == [{"endpoint": "https://ps.example/ai/submitLabelsOnPano",
                                         "submitted_lines": 25, "labels_submitted": 40}]
    assert result["rule"] == position_check.RULE


def test_reposition_never_overwrites_a_submitted_campaign_file(tmp_path):
    """The default output name can be a file that already shipped (Laurens' results.check.jsonl);
    overwriting it would put other panos under the hash its submission record vouches for."""
    frame = _frame()
    run = tmp_path / "city"
    _write_run(run, _northbound("A", 0.5, -7.5, frame), frame)
    shipped = run / "results.raw.jsonl"
    shipped.write_text("{}\n", encoding="utf-8")
    position_check.submission_record_for(shipped).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="submitted campaign file"):
        reposition.main([str(run / "results.jsonl"), "--field", "raw"])
    assert shipped.read_text(encoding="utf-8") == "{}\n"
