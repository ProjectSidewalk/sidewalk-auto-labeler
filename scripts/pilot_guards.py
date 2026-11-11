"""Run send_to_ps.py's pre-send guards on a results file WITHOUT sending anything (#169 pilot).

A `send_to_ps.py --dry-run` skips every pre-send guard, so a dry run alone cannot show
whether a real send would be refused. This calls each guard a real send runs, in the order
`send_to_ps.process_jsonl_file` runs them, against the endpoint the send would use, and
prints each verdict verbatim. It reads files only: no request, no sidecar, no record.

    python scripts/pilot_guards.py <results.jsonl> --endpoint https://<server>/ai/submitLabelsOnPano
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import send_to_ps as s  # noqa: E402
from detectors.registry import spec_for  # noqa: E402


def run_guards(path, endpoint):
    """[(guard, verdict text)] for every pre-send guard, in send order; never raises for a
    guard's refusal (that is the verdict)."""
    f = Path(path)
    out = []

    def call(name, fn):
        try:
            res = fn()
            out.append((name, 'PASS' + ('' if res in (None, '') else f': {res}')))
        except ValueError as e:
            out.append((name, f'REFUSED: {e}'))
        except s.CampaignComplete as e:   # not a refusal: nothing left to send
            out.append((name, f'NOTHING TO DO: {e}'))

    call('check_endpoint_security', lambda: s.check_endpoint_security(endpoint, None))
    call('check_model_provenance', lambda: s.check_model_provenance(f))
    call('check_store_built', lambda: s.check_store_built(f))
    call('check_detection_decode', lambda: s.check_detection_decode(f, endpoint))
    call('check_detection_border', lambda: s.check_detection_border(f, endpoint))
    types = s.file_label_types(f)
    out.append(('file_label_types', f'{types} (one type: legacy sidecar, one POST per line)'
                if len(types) == 1 else f'{types} (multi-type: per-type sidecars)'))
    min_conf = spec_for(types[0]).operational_confidence if len(types) == 1 else None
    digest, total_lines, total_bytes = s.hash_and_count(f)
    sidecar = s.sidecar_path_for(str(f), min_conf, None)
    record_path = s.submission_record_path(str(f))

    def resume():
        record = s.load_submission_record(record_path)
        return s.check_resume_state(record, digest, s.load_submitted_lines(sidecar), endpoint,
                                    min_conf, f, record_path, sidecar, total_lines, total_bytes,
                                    None, None)
    call(f'check_resume_state (sidecar {sidecar.name}, record {record_path.name}, '
         f'min_confidence {min_conf})', resume)
    call('check_position_state', lambda: s.check_position_state(f, digest))
    call('check_live_positions', lambda: s.check_live_positions(f, endpoint, True))
    out.append(('file', f'sha256 {digest}, {total_lines} lines, {total_bytes} bytes'))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('results')
    ap.add_argument('--endpoint', required=True)
    args = ap.parse_args(argv)
    for name, verdict in run_guards(args.results, args.endpoint):
        print(f'{name}: {verdict}')


if __name__ == '__main__':
    main()
