"""send_to_ps.py over multi-type results files (issue #168): one POST per (line, type).

The golden half pins what a CurbRamp-only file produces -- the POSTed payloads, the dry-run
output, the resume sidecar's bytes and the submission record's bytes -- against fixtures
captured from origin/main BEFORE any multi-type code existed, so "single-type files are
unchanged" is a byte comparison, not a reading of the diff. Torch-free; nothing is POSTed.
"""
import io
import json
import os
import shutil
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import coverage_check
import position_check
import main
import reinfer
import send_to_ps
from conftest import make_provenance
from detectors import registry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "multitype_submit"
GOLDEN_JSONL = FIXTURES / "curbramp_only.jsonl"
PROD = "https://ps.example/ai/submitLabelsOnPano"


class _FrozenDatetime(datetime):
    """The record stamps `datetime.now(timezone.utc)`; freeze it so its bytes are comparable."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


def _golden_campaign(workdir, setattr):
    """Run the fixture through a capped then a full mocked campaign, plus a dry run.

    `setattr` is a monkeypatch.setattr, so the same function that captured the fixtures
    (on origin/main 4993105, with pytest.MonkeyPatch) checks them. Returns (posted payloads,
    dry-run stdout with the file path replaced by <FILE>, sidecar bytes, record bytes).
    """
    path = Path(workdir) / "results.jsonl"
    shutil.copyfile(GOLDEN_JSONL, path)
    sent = []
    ok = SimpleNamespace(status_code=200, ok=True, text="")
    setattr(send_to_ps, "send_to_project_sidewalk",
            lambda payload, url, key=None: sent.append(payload) or ok)
    setattr(send_to_ps.socket, "gethostname", lambda: "golden-host")
    setattr(send_to_ps, "datetime", _FrozenDatetime)

    out = io.StringIO()
    with redirect_stdout(out):
        send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True)
    dry = out.getvalue().replace(str(path), "<FILE>")
    with redirect_stdout(io.StringIO()):
        send_to_ps.process_jsonl_file(str(path), PROD, limit=2)
        send_to_ps.process_jsonl_file(str(path), PROD)
    sidecar = (Path(workdir) / "results.jsonl.submitted").read_bytes()
    record = (Path(workdir) / "results.jsonl.submission.json").read_bytes()
    return sent, dry, sidecar, record


def test_curbramp_only_campaign_is_byte_identical_to_main(tmp_path, monkeypatch):
    """Issue #168's done-when: a CurbRamp-only file submits exactly as before the multi-type
    plumbing -- same payloads, same dry run, same sidecar bytes, same record bytes."""
    sent, dry, sidecar, record = _golden_campaign(tmp_path, monkeypatch.setattr)
    assert sent == json.loads((FIXTURES / "curbramp_only.payloads.json").read_text(encoding="utf-8"))
    assert dry == (FIXTURES / "curbramp_only.dryrun.txt").read_text(encoding="utf-8")
    # The sidecar is written in text mode, so today's bytes end lines with os.linesep; the
    # fixture is stored LF (.gitattributes) and compared in the platform's own line ending.
    expected = (FIXTURES / "curbramp_only.submitted").read_bytes()
    assert sidecar == expected.replace(b"\n", os.linesep.encode())
    assert record == (FIXTURES / "curbramp_only.submission.json").read_bytes()


# --- multi-type files --------------------------------------------------------------------

# A stand-in second detector: a higher operating point than CurbRamp's and NO rig mask, so
# every test below can tell "the spec's value" from "CurbRamp's value".
XWALK = "Crosswalk"
XWALK_MODEL = {"model_id": "xwalk-model@abcdef123456", "model_training_date": "10-01-2026",
               "api_version": "1.0.0", "model_repo": "projectsidewalk/xwalk-model",
               "model_revision": "abcdef1234567890abcdef1234567890abcdef12"}


@pytest.fixture(autouse=True)
def crosswalk_spec(monkeypatch):
    monkeypatch.setitem(registry.SPECS, XWALK, registry.DetectorSpec(
        XWALK, "detectors.xwalk", "XwalkDetector", 0.1, 0.5, 0.6, False))


def _golden_line(n=0):
    return json.loads(GOLDEN_JSONL.read_text(encoding="utf-8").splitlines()[n])


def _curb_model():
    line = _golden_line()
    return {k: line[k] for k in registry.PROVENANCE_KEYS}


def _two_type_line(n, detections):
    """A two-type line written by the real writer (main.build_output_line, #168): typed
    detections, label_type null, label_types sorted (Crosswalk < CurbRamp), a models map,
    line-level model_* = CurbRamp's. The pano block is the golden fixture's, renamed."""
    pano = dict(_golden_line()["pano"], panorama_id=f"MT{n}", lat=44.05 + n * 1e-4)
    result = {"pano_id": pano["panorama_id"], "pano": pano,
              "detections": [(x, y, c, t) for t, x, y, c in detections],
              "label_types": sorted(["CurbRamp", XWALK])}
    return json.loads(json.dumps(main.build_output_line(
        result, {"CurbRamp": make_provenance(), XWALK: dict(XWALK_MODEL)})))


def _two_type_file(tmp_path, name="results.jsonl"):
    lines = [
        _two_type_line(1, [("CurbRamp", 0.5, 0.25, 0.9), (XWALK, 0.25, 0.6, 0.7),
                           ("CurbRamp", 0.3, 0.95, 0.8), (XWALK, 0.75, 0.95, 0.8),
                           (XWALK, 0.1, 0.6, 0.4)]),     # 0.4: above CurbRamp's 0.30, below 0.5
        _two_type_line(2, []),
    ]
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8", newline="\n")
    return path


def _capture(monkeypatch, fail=lambda payload: False):
    sent = []
    ok = SimpleNamespace(status_code=200, ok=True, text="")

    def post(payload, url, key=None):
        if fail(payload):
            return None
        sent.append(payload)
        return ok
    monkeypatch.setattr(send_to_ps, "send_to_project_sidewalk", post)
    return sent


def _state(tmp_path):
    record = json.loads((tmp_path / "results.jsonl.submission.json").read_text(encoding="utf-8"))
    return record["endpoints"][PROD]


def test_two_type_file_posts_once_per_line_and_type(tmp_path, monkeypatch):
    """Each (line, type) is its own AiLabelsSubmission: that type's detections only, at that
    type's threshold and mask, under that type's model; nothing of the other type rides along."""
    path = _two_type_file(tmp_path)
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert [(p["pano"]["pano_id"], p["label_type"]) for p in sent] == [
        ("MT1", XWALK), ("MT1", "CurbRamp"), ("MT2", XWALK), ("MT2", "CurbRamp")]
    xwalk, curb = sent[0], sent[1]
    # CurbRamp: 0.30, rig-masked (the y=0.95 detection is on the vehicle).
    assert curb["labels"] == [{"pano_x": 8192, "pano_y": 2048, "confidence": 0.9}]
    assert {k: curb[k] for k in registry.PROVENANCE_KEYS} == _curb_model()
    # Crosswalk: its own 0.5 floor (0.4 dropped) and no rig mask (y=0.95 kept).
    assert xwalk["labels"] == [{"pano_x": 4096, "pano_y": 4915, "confidence": 0.7},
                               {"pano_x": 12288, "pano_y": 7782, "confidence": 0.8}]
    assert {k: xwalk[k] for k in registry.PROVENANCE_KEYS} == XWALK_MODEL
    for payload in sent:
        assert not {"detections", "models", "label_types"} & set(payload)
    assert sent[2]["labels"] == [] == sent[3]["labels"]   # checked for each type, nothing found


def test_two_type_dry_run_prints_both_payloads(tmp_path, capsys):
    path = _two_type_file(tmp_path)
    send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True)
    out = capsys.readouterr().out
    assert out.count('"label_type": "CurbRamp"') == 2
    assert out.count(f'"label_type": "{XWALK}"') == 2
    assert '"model_id": "xwalk-model@abcdef123456"' in out
    assert not list(tmp_path.glob("*.submitted")) and not list(tmp_path.glob("*.submission.json"))


def test_per_type_sidecars_and_record(tmp_path, monkeypatch):
    """A multi-type file never writes the whole-line sidecar; each type has its own, and the
    record mirrors the band shape under endpoints[ep].types."""
    path = _two_type_file(tmp_path)
    _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert not (tmp_path / "results.jsonl.submitted").exists()
    for t in ("CurbRamp", XWALK):
        assert send_to_ps.load_submitted_lines(
            tmp_path / f"results.jsonl.type-{t}.submitted") == {1, 2}
    state = _state(tmp_path)
    curb, xwalk = state["types"]["CurbRamp"], state["types"][XWALK]
    assert (curb["submitted_lines"], curb["labels_submitted"], curb["min_confidence"],
            curb["rig_masked"]) == (2, 1, 0.3, True)
    assert (xwalk["submitted_lines"], xwalk["labels_submitted"], xwalk["min_confidence"],
            xwalk["rig_masked"]) == (2, 2, 0.5, False)
    assert (state["submitted_lines"], state["labels_submitted"]) == (2, 3)
    assert (state["min_confidence"], state["min_confidence_from"]) == (0.3, "per_type_default")
    # A rerun finds every (line, type) done and sends nothing.
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_explicit_min_confidence_applies_to_every_type(tmp_path, monkeypatch):
    path = _two_type_file(tmp_path)
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.35)
    assert [lab["confidence"] for lab in sent[0]["labels"]] == [0.7, 0.8, 0.4]
    state = _state(tmp_path)
    assert {t: s["min_confidence"] for t, s in state["types"].items()} == {"CurbRamp": 0.35,
                                                                         XWALK: 0.35}
    assert state["min_confidence_from"] == "flag"
    # ...and the resume guard holds each type to it, as it holds a single-type campaign.
    with pytest.raises(ValueError, match="--min-confidence 0.35"):
        send_to_ps.process_jsonl_file(str(path), PROD)


def test_resume_after_a_half_sent_line_sends_only_the_missing_type(tmp_path, monkeypatch):
    path = _two_type_file(tmp_path)
    _capture(monkeypatch,
             fail=lambda p: p["label_type"] == XWALK and p["pano"]["pano_id"] == "MT1")
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert send_to_ps.load_submitted_lines(
        tmp_path / f"results.jsonl.type-{XWALK}.submitted") == {2}
    state = _state(tmp_path)
    assert state["submitted_lines"] == 2      # MT1 is half-sent, so its pano is live
    assert state["types"][XWALK]["submitted_lines"] == 1

    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert [(p["pano"]["pano_id"], p["label_type"]) for p in sent] == [("MT1", XWALK)]


def test_a_lost_type_sidecar_is_refused_like_a_lost_sidecar(tmp_path, monkeypatch):
    path = _two_type_file(tmp_path)
    _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    (tmp_path / f"results.jsonl.type-{XWALK}.submitted").unlink()
    with pytest.raises(ValueError, match=f"type-{XWALK}.submitted accounts for only 0"):
        send_to_ps.process_jsonl_file(str(path), PROD)


def test_transform_record_without_a_type_refuses_a_multi_type_line():
    line = _two_type_line(1, [("CurbRamp", 0.5, 0.25, 0.9), (XWALK, 0.25, 0.6, 0.7)])
    with pytest.raises(ValueError, match="one label type per submission"):
        send_to_ps.transform_record(line)
    assert [t for t, _ in send_to_ps.split_record(line)] == [XWALK, "CurbRamp"]


def test_curb_ramp_decode_marker_stays_off_the_other_types_payload():
    """Line-level detection_decode describes the CurbRamp detector (#168 writer rule)."""
    line = _two_type_line(1, [("CurbRamp", 0.5, 0.25, 0.9), (XWALK, 0.25, 0.6, 0.7)])
    line["detection_decode"] = "gaussian"
    assert send_to_ps.transform_record(line, label_type="CurbRamp")["detection_decode"] == "gaussian"
    assert "detection_decode" not in send_to_ps.transform_record(line, label_type=XWALK)


def test_split_record_of_a_legacy_line_is_the_untyped_transform():
    line = _golden_line(2)
    assert send_to_ps.split_record(line) == [("CurbRamp", send_to_ps.transform_record(line))]


def test_mask_rig_follows_the_spec(monkeypatch):
    line = _two_type_line(1, [("CurbRamp", 0.5, 0.95, 0.9), (XWALK, 0.5, 0.95, 0.9)])
    assert send_to_ps.transform_record(line, label_type="CurbRamp")["labels"] == []
    assert len(send_to_ps.transform_record(line, label_type=XWALK)["labels"]) == 1
    monkeypatch.setitem(registry.SPECS, XWALK, registry.SPECS[XWALK]._replace(mask_rig=True))
    assert send_to_ps.transform_record(line, label_type=XWALK)["labels"] == []


def test_band_on_a_multi_type_file_is_refused(tmp_path, monkeypatch):
    path = _two_type_file(tmp_path)
    sent = _capture(monkeypatch)
    for dry in (False, True):
        with pytest.raises(ValueError, match="legacy CurbRamp cities"):
            send_to_ps.process_jsonl_file(str(path), PROD, dry_run=dry, min_confidence=0.3,
                                          max_confidence=0.55)
    assert sent == []


def test_write_band_file_refuses_a_multi_type_file(tmp_path):
    _two_type_file(tmp_path)
    _two_type_file(tmp_path, "results.f01.jsonl")
    with pytest.raises(SystemExit, match="multi-type"):
        reinfer.main_cli([str(tmp_path), "--verify", "--write-band-file"])


def test_undated_type_model_is_refused(tmp_path, monkeypatch):
    """Every models[T] is its own POST's provenance, so each must carry a training date."""
    line = _two_type_line(1, [])
    line["models"][XWALK]["model_training_date"] = None
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    sent = _capture(monkeypatch)
    with pytest.raises(ValueError, match=r"\(Crosswalk\)"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_live_positions_counts_a_half_sent_pano_as_placed(tmp_path, monkeypatch):
    """The pano row is upserted by the first POST that names it, so a line with only some
    types sent already puts its pano (and those labels) on the server."""
    path = _two_type_file(tmp_path)
    _capture(monkeypatch,
             fail=lambda p: p["label_type"] == XWALK or p["pano"]["pano_id"] == "MT2")
    send_to_ps.process_jsonl_file(str(path), PROD)
    live, problems = position_check.live_positions(tmp_path, PROD)
    assert problems == [] and set(live) == {"MT1"}
    campaigns, _ = position_check.campaigns_for(tmp_path, PROD)
    assert [(c["label_type"], c["line_numbers"]) for c in campaigns] == [("CurbRamp", {1})]


def test_coverage_expected_set_is_per_type(tmp_path, monkeypatch):
    path = _two_type_file(tmp_path)
    _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    expected, summaries, problems = coverage_check.expected_panos(tmp_path, PROD)
    assert problems == []
    assert expected["MT1"]["labels"] == 3
    assert expected["MT1"]["label_types"] == {"CurbRamp", XWALK}
    assert sorted((s["campaign"], s["replayed_labels"]) for s in summaries) == [
        (f"results.jsonl (type {XWALK})", 2), ("results.jsonl (type CurbRamp)", 1)]


def test_a_whole_line_sidecar_beside_a_multi_type_file_is_refused(tmp_path, monkeypatch):
    """`<file>.submitted` beside a multi-type file means it went out one POST per line, every
    type under one label type; the per-type campaign must not quietly start beside it."""
    path = _two_type_file(tmp_path)
    (tmp_path / "results.jsonl.submitted").write_text("1\n", encoding="utf-8")
    sent = _capture(monkeypatch)
    with pytest.raises(ValueError, match="one <file>.type-<T>.submitted per type"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


# --- review follow-ups (guards on the per-type record) ------------------------------------

TEST = "https://ps-test.example/ai/submitLabelsOnPano"


def _rewrite_state(tmp_path, endpoint, edit):
    path = tmp_path / "results.jsonl.submission.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    edit(record["endpoints"][endpoint])
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def test_a_single_type_file_whose_record_has_types_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    shutil.copyfile(GOLDEN_JSONL, path)
    _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    _rewrite_state(tmp_path, PROD, lambda st: st.update(types={"CurbRamp": {"submitted_lines": 3}}))
    with pytest.raises(ValueError, match="records a per-type campaign"):
        send_to_ps.process_jsonl_file(str(path), PROD)


@pytest.mark.parametrize("endpoint", [PROD, TEST])
def test_a_multi_type_file_whose_record_lacks_types_is_refused(tmp_path, monkeypatch, endpoint):
    """Checked over EVERY endpoint: a whole-line entry on a test server means the file already
    went out there with every type under one label type."""
    path = _two_type_file(tmp_path)
    _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), endpoint)
    _rewrite_state(tmp_path, endpoint, lambda st: st.pop("types"))
    for p in tmp_path.glob("*.type-*.submitted"):
        p.rename(p.with_name(p.name + ".aside"))
    sent = _capture(monkeypatch)
    with pytest.raises(ValueError, match=f"sent to {endpoint} WITHOUT per-type entries"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_a_pure_append_to_a_multi_type_file_resumes_and_a_repeat_is_refused(tmp_path, monkeypatch):
    """#59 works on line numbers, so it holds for per-type sidecars exactly as for one."""
    path = _two_type_file(tmp_path)
    _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    submitted = path.read_bytes()
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_two_type_line(3, [(XWALK, 0.5, 0.6, 0.9)])) + "\n")
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert [(p["pano"]["pano_id"], p["label_type"]) for p in sent] == [
        ("MT3", XWALK), ("MT3", "CurbRamp")]
    assert _state(tmp_path)["types"][XWALK]["submitted_lines"] == 3

    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(submitted.decode("utf-8").splitlines()[0] + "\n")    # MT1 again
    sent = _capture(monkeypatch)
    with pytest.raises(ValueError, match="the panos are not"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_test_then_prod_with_type_sidecars_left_in_place_is_refused(tmp_path, monkeypatch):
    path = _two_type_file(tmp_path)
    _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), TEST)
    sent = _capture(monkeypatch)
    with pytest.raises(ValueError, match="the rest went to " + TEST):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []
    for p in tmp_path.glob("*.type-*.submitted"):     # the documented move, per type
        p.rename(p.with_name(p.name + ".staging"))
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 4


def test_an_interrupt_between_two_types_resumes_with_the_missing_pair(tmp_path, monkeypatch):
    path = _two_type_file(tmp_path)
    ok = SimpleNamespace(status_code=200, ok=True, text="")
    calls = []

    def post(payload, url, key=None):
        calls.append((payload["pano"]["pano_id"], payload["label_type"]))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return ok
    monkeypatch.setattr(send_to_ps, "send_to_project_sidewalk", post)
    with pytest.raises(SystemExit):
        send_to_ps.process_jsonl_file(str(path), PROD)
    state = _state(tmp_path)
    assert state["types"][XWALK]["submitted_lines"] == 1 and "CurbRamp" not in state["types"]
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert [(p["pano"]["pano_id"], p["label_type"]) for p in sent] == [
        ("MT1", "CurbRamp"), ("MT2", XWALK), ("MT2", "CurbRamp")]


def test_a_single_non_curb_ramp_type_file_goes_through_the_typed_transform(tmp_path, monkeypatch):
    """A Crosswalk-only run writes label_types; its payload is what split_record would build:
    the type's threshold and mask, no models/label_types, no CurbRamp decode marker."""
    pano = dict(_golden_line()["pano"], panorama_id="X1")
    line = json.loads(json.dumps(main.build_output_line(
        {"pano_id": "X1", "pano": pano, "label_types": [XWALK],
         "detections": [(0.5, 0.6, 0.9, XWALK), (0.5, 0.95, 0.8, XWALK), (0.2, 0.6, 0.4, XWALK)]},
        {XWALK: dict(XWALK_MODEL)})))
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == [send_to_ps.split_record(line)[0][1]]
    assert [lab["confidence"] for lab in sent[0]["labels"]] == [0.9, 0.8]
    assert not {"models", "label_types", "detection_decode"} & set(sent[0])
    assert send_to_ps.load_submitted_lines(tmp_path / "results.jsonl.submitted") == {1}
    assert _state(tmp_path)["min_confidence"] == 0.5 and "types" not in _state(tmp_path)


def test_a_dry_run_band_below_the_default_floor_is_refused(tmp_path):
    path = tmp_path / "results.jsonl"
    shutil.copyfile(GOLDEN_JSONL, path)
    with pytest.raises(ValueError, match="must be below --max-confidence"):
        send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True, max_confidence=0.2)
