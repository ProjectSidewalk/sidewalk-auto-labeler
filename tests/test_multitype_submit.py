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

import send_to_ps

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
