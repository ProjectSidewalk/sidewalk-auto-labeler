"""Drive the gallery's in-browser review logic in node (skipped if node is absent).

build_html embeds the viewer JS; gallery_state_test.js loads it against a minimal
DOM stub and walks the full review flow: judge crops, affirm/revoke the
missed-ramp check, the reviewed() gate, and the exported verdict schema. The
export it produces is then scored with score_validation.collect(), so the
viewer→scorer contract is tested end to end across the language boundary.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import score_validation as sv
import spot_check_gallery as g

HARNESS = Path(__file__).parent / "gallery_state_test.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_gallery_review_state_machine(tmp_path):
    entries = [
        {"pid": "TWO", "date": "2024-06", "group": "random", "full": "TWO_full.jpg",
         "crops": [
             {"img": "TWO_det0.jpg", "conf": 0.91, "x": 0.5, "y": 0.6, "cx": 0.5, "cy": 0.5},
             {"img": "TWO_det1.jpg", "conf": 0.63, "x": 0.9, "y": 0.1, "cx": 0.7, "cy": 0.2},
         ]},
        {"pid": "EMPTY", "date": "2024-01", "group": "empty",
         "full": "EMPTY_full.jpg", "crops": []},
    ]
    html = g.build_html(entries, tmp_path / "results.jsonl", "testhash", "testrun")
    js = html.split("<script>")[1].split("</script>")[0]
    js_path = tmp_path / "gallery.js"
    js_path.write_text(js, encoding="utf-8")

    result = subprocess.run(["node", str(HARNESS), str(js_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "gallery state machine OK" in result.stdout

    # Score the viewer's actual export with the real scorer: at the end of the
    # harness, TWO has crop0 correct + crop1 'unsure' + one missed mark, EMPTY is
    # affirmed no-missed. The unsure crop abstains, so precision/recall keep only
    # crop0's verdict; both panos are fully judged.
    export = json.loads(next(line for line in result.stdout.splitlines()
                             if line.startswith("EXPORT_JSON ")).removeprefix("EXPORT_JSON "))
    confs = {"TWO": [0.91, 0.63], "EMPTY": []}
    judged, recall_judged, missed, n_seen, n_judged, n_unconfirmed, n_unsure, missed_unsure = \
        sv.collect(export["panos"], confs)
    assert (n_seen, n_judged, n_unconfirmed) == (2, 2, 0)
    assert sorted(judged) == [(0.91, True)] == sorted(recall_judged)
    assert missed == 1 and n_unsure == 1 and missed_unsure == 0
