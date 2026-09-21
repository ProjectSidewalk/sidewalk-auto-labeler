#!/usr/bin/env python
"""Rewrite the pano positions of a Mapillary results.jsonl from the other position field.

Companion to scripts/position_check.py (SidewalkWebpage#5361). Every Mapillary record
already carries both positions in source_metadata — the raw GPS fix (`geometry`) and
the SfM-corrected one (`computed_geometry`) — so switching the submitted pano lat/lng
between them needs no re-detection: detections are stored relative to the pano, and the
server places each label from the pano position plus the pixel offset, so moving the
pano moves its labels by the same vector.

Usage:
    # whole file, one field
    python scripts/reposition.py runs/laurens/results.jsonl --field raw

    # only the sequences position_check.py flagged, each to its recommended field
    python scripts/reposition.py runs/laurens/results.jsonl --from-check

    # explicit sequences
    python scripts/reposition.py runs/laurens/results.jsonl --field raw --sequences A,B

Writes <input stem>.<field or 'check'>.jsonl beside the input (or --out). The input is
never modified, and the output is a new file with a new hash, so send_to_ps.py's
submission guard treats it as a fresh campaign — which it is: the labels already on the
server from the old positions have to be retired server-side before the new ones land.
Each rewritten pano block gains `position_field` naming the field it now carries.

The output is a SUBMISSION ARTIFACT, not a run. Never swap it into results.jsonl:
main.py binds a run dir to one position field through the manifest, but it cannot see
inside results.jsonl, so a resume after a swap would append panos on the manifest's
field to a file that is now mixed — exactly what the binding exists to prevent. Confirm
the output with `position_check.py runs/<name> --results <output>` and submit it from
where it is.

Only lat/lng move. `camera_heading` (Mapillary's `computed_compass_angle`) is SfM-derived
too, but the measured SfM-vs-GPS discrepancies are translations — one near-constant
vector per sequence (position_check.json `sfm_minus_raw_*`) — not rotations, so the
heading stays consistent with either position and is deliberately left alone.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from position_check import check_path_for  # noqa: E402

FIELD_KEYS = {'raw': 'geometry', 'sfm': 'computed_geometry'}


def reposition_pano(pano, field):
    """Return a copy of the pano block positioned from `field` ('raw' or 'sfm'), or None
    when the record has no such position (non-Mapillary, or the field is absent)."""
    meta = pano.get('source_metadata') or {}
    coords = (meta.get(FIELD_KEYS[field]) or {}).get('coordinates')
    if pano.get('source') != 'mapillary' or not coords:
        return None
    out = dict(pano)
    out['lng'], out['lat'] = float(coords[0]), float(coords[1])
    out['position_field'] = FIELD_KEYS[field]
    return out


def plan_from_check(check_path):
    """{sequence_id: field} for every flagged sequence that has a recommendation."""
    with open(check_path, encoding='utf-8') as f:
        check = json.load(f)
    plan = {s['sequence_id']: s['recommended'] for s in check['sequences']
            if s['flagged'] and s['recommended']}
    return plan, check


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', help='runs/<name>/results.jsonl')
    ap.add_argument('--field', choices=sorted(FIELD_KEYS), help="position field to write ('raw' GPS or 'sfm')")
    ap.add_argument('--sequences', metavar='ID,ID,...', help='only these sequence ids (default: all)')
    ap.add_argument('--from-check', nargs='?', const='', metavar='JSON',
                    help='take flagged sequences and their recommended field from position_check.json '
                         '(default: the one beside the results file)')
    ap.add_argument('--out', metavar='FILE', help='output path (default: <stem>.<field|check>.jsonl)')
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(errors='replace')

    src = Path(args.results)
    if args.from_check is None and not args.field:
        ap.error('give --field, or --from-check to use position_check.json')
    if args.from_check is not None and (args.field or args.sequences):
        ap.error('--from-check decides field and sequences itself; drop --field/--sequences')

    if args.from_check is not None:
        check_path = Path(args.from_check) if args.from_check else check_path_for(src)
        plan, check = plan_from_check(check_path)
        if not plan:
            print(f'nothing to do: {check_path} flags no sequence')
            return 0
        suffix = 'check'
        print(f"-> {len(plan)} flagged sequence(s) from {check_path}: "
              + ', '.join(f'{k}->{v}' for k, v in sorted(plan.items())))
    else:
        only = set(args.sequences.split(',')) if args.sequences else None
        plan = None
        suffix = args.field

    out = Path(args.out) if args.out else src.with_name(f'{src.stem}.{suffix}.jsonl')
    if out.resolve() == src.resolve():
        sys.exit('refusing to overwrite the input; use --out')
    if out.name == 'results.jsonl':
        sys.exit(f'refusing to write {out}: results.jsonl is a run file that main.py resumes into, and this '
                 f'output is a submission artifact (see the docstring); keep it under another name')

    counts = Counter()
    with open(src, encoding='utf-8') as fin, open(out, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            pano = rec['pano']
            seq = pano.get('sequence_id')
            if plan is not None:
                field = plan.get(seq)
            else:
                field = args.field if (only is None or seq in only) else None
            if field:
                new = reposition_pano(pano, field)
                if new is None:
                    counts['no_field'] += 1
                else:
                    rec['pano'] = new
                    counts[f'rewritten_{field}'] += 1
            else:
                counts['unchanged'] += 1
            fout.write(json.dumps(rec) + '\n')

    total = sum(counts.values())
    print(f"-> wrote {out}: {total} records; "
          + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))
    if counts['no_field']:
        print(f"!! {counts['no_field']} record(s) had no {FIELD_KEYS.get(args.field, 'requested')} position and were left as-is")
    print(f'   next: python scripts/position_check.py {src.parent.as_posix()} --results {out.as_posix()} '
          f'(do NOT swap it into results.jsonl), then send_to_ps.py {out.as_posix()} — the submitter '
          f'refuses a Mapillary file whose check is missing, stale or still flagged')
    return 0


if __name__ == '__main__':
    sys.exit(main())
