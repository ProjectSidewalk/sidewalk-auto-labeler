"""Paid runner for Stage 1 of the corner pre-rater plan (issue #185): Gemini on judged corners.

Reuses RampNet's Gemini harness read-only (D:\\Git\\RampNet\\scripts\\model_comparison):
`GeminiDetector`'s client (Vertex + ADC from RampNet's git-ignored .env, its retry policy and
10-minute request timeout), its usage accounting (`_record_usage`), `pricing.estimate_cost`,
and `rampnet.ledger.append_rows`. Every leg appends one ledger row, tagged
`corner-prerater-185`, both to RampNet's shared ledger (analysis_out/usage_log.jsonl in the
main checkout, which vertex_usage.py --reconcile reads) and to this worktree's
runs/vancouver/corner_prerater185/stage1/usage_log.jsonl, which is committed.

Run it with RampNet's venv, which has google-genai:

    R=/d/Git/RampNet/.venv/Scripts/python.exe
    $R scripts/corner_vlm.py run --arm argon_multi_r1 --keys smoke --budget-usd 2
    $R scripts/corner_vlm.py run --arm argon_single --keys all --views nearest --budget-usd 8

Answers are appended line by line to stage1/responses/<arm>.jsonl; a re-run skips every call
id already there, so nothing is billed twice. A call that raised is not written and is retried
on the next run; a call that answered but did not parse is written with verdict null and is
NOT retried (it was billed; temperature 0). The leg stops submitting new calls once its own
estimated spend reaches --budget-usd. Credentials are never printed.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
import corner_prerater_s1 as s1  # noqa: E402

RAMPNET = Path(os.environ.get('RAMPNET_ROOT', 'D:/Git/RampNet'))
sys.path.insert(0, str(RAMPNET))
sys.path.insert(0, str(RAMPNET / 'scripts' / 'model_comparison'))

STAGE1 = REPO_ROOT / 'runs/vancouver/corner_prerater185/stage1'
TAG = 'corner-prerater-185'
TEMPERATURE = 0.0


def load_google_env(root):
    """Only the GOOGLE_* lines of RampNet's .env go into the environment (never printed);
    nothing else in that file is read into this process."""
    path = Path(root) / '.env'
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.startswith('GOOGLE_') and k not in os.environ:
            os.environ[k] = v


def parse_answer(resp):
    """(verdict, absent_kind) from a response, or (None, None) when it does not parse."""
    data = getattr(resp, 'parsed', None)
    if not isinstance(data, dict):
        try:
            data = json.loads(getattr(resp, 'text', None) or '')
        except (TypeError, ValueError):
            return None, None
    if not isinstance(data, dict):
        return None, None
    v = data.get('verdict')
    if v not in s1.VERDICTS:
        return None, None
    k = data.get('absent_kind')
    return v, (k if k in s1.ABSENT_KINDS else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--arm', required=True, choices=sorted(s1.ARMS))
    r.add_argument('--keys', required=True, help="'smoke' (the seeded 20), 'all', or a file")
    r.add_argument('--views', default='all', choices=('nearest', 'rest', 'all'))
    r.add_argument('--budget-usd', type=float, required=True)
    r.add_argument('--concurrency', type=int, default=6)
    r.add_argument('--stage1', default=str(STAGE1))
    args = ap.parse_args(argv)

    from detectors import GeminiDetector
    from pricing import estimate_cost, price_for
    from rampnet import ledger

    stage1 = Path(args.stage1)
    reqs = s1.read_jsonl(stage1 / 'requests.jsonl')
    meta = json.loads((stage1 / 'requests.json').read_text(encoding='utf-8'))
    if meta['prompt_sha256'] != s1.prompt_sha256():
        raise SystemExit('the frozen prompt changed since export; refusing')
    by_key = {q['key']: q for q in reqs}
    if args.keys == 'smoke':
        keys = s1.smoke_keys(reqs)
    elif args.keys == 'all':
        keys = None
    else:
        keys = Path(args.keys).read_text(encoding='utf-8').split()
    model_short, mode, replicate = s1.ARMS[args.arm]
    model_id = s1.MODELS[model_short]
    todo_all = s1.call_keys(reqs, args.arm, keys, args.views)
    out_path = stage1 / 'responses' / f'{args.arm}.jsonl'
    done = set()
    if out_path.exists():
        done = {x['call_id'] for x in s1.read_jsonl(out_path)}
    todo = [c for c in todo_all if c not in done]
    print(f'[{args.arm}] {model_id} {mode} r{replicate}: {len(todo_all)} calls in scope, '
          f'{len(todo_all) - len(todo)} already answered, {len(todo)} to make, '
          f'budget ${args.budget_usd:.2f}')
    if not todo:
        return 0

    load_google_env(RAMPNET)
    det = GeminiDetector(model_id=model_id, temperature=TEMPERATURE)
    det._ensure_ready()   # fails fast (and says why) without credentials; prints none
    from google.genai import types
    config = types.GenerateContentConfig(response_mime_type='application/json',
                                         response_schema=s1.RESPONSE_SCHEMA,
                                         temperature=TEMPERATURE)
    lock = threading.Lock()
    stop = threading.Event()
    stats = {'written': 0, 'no_answer': 0, 'errors': 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def image_part(path):
        data = Path(path).read_bytes()
        mime = 'image/png' if path.lower().endswith('.png') else 'image/jpeg'
        return types.Part.from_bytes(data=data, mime_type=mime)

    def spent():
        u = det.usage
        return estimate_cost(model_id, u['input_tokens'], u['output_tokens']) or 0.0

    def one(cid):
        if stop.is_set():
            return
        key, _, crop = cid.partition('@')
        req = by_key[key]
        parts = s1.prompt_parts(req, mode, int(crop) if crop else None)
        contents = [p if kind == 'text' else image_part(p) for kind, p in parts]
        t0 = time.time()
        try:
            resp = det._client.models.generate_content(model=model_id, contents=contents,
                                                       config=config)
        except Exception as e:   # retryable: not written, re-run picks it up
            with lock:
                stats['errors'] += 1
                print(f'[{args.arm}] {cid}: {type(e).__name__}: {str(e)[:200]}')
            return
        det._record_usage(resp)
        verdict, kind = parse_answer(resp)
        um = getattr(resp, 'usage_metadata', None)
        cands = getattr(resp, 'candidates', None) or []
        fr = getattr(cands[0], 'finish_reason', None) if cands else None
        row = {'call_id': cid, 'key': key, 'arm': args.arm, 'model_id': model_id,
               'mode': mode, 'replicate': replicate, 'crop': int(crop) if crop else None,
               'verdict': verdict, 'absent_kind': kind,
               'raw_text': (getattr(resp, 'text', None) or '')[:2000],
               'finish_reason': getattr(fr, 'name', None) or (str(fr) if fr else None),
               'prompt_tokens': getattr(um, 'prompt_token_count', None) if um else None,
               'candidates_tokens': getattr(um, 'candidates_token_count', None) if um else None,
               'thoughts_tokens': getattr(um, 'thoughts_token_count', None) if um else None,
               'model_version': getattr(resp, 'model_version', None),
               'seconds': round(time.time() - t0, 2),
               'ts': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        with lock:
            with open(out_path, 'a', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps(row, sort_keys=True) + '\n')
            stats['written'] += 1
            stats['no_answer'] += verdict is None
            if spent() >= args.budget_usd and not stop.is_set():
                stop.set()
                print(f'[{args.arm}] budget ${args.budget_usd:.2f} reached '
                      f'(${spent():.2f}); no new calls')
            if stats['written'] % 25 == 0:
                print(f"[{args.arm}] {stats['written']}/{len(todo)} answered, ~${spent():.2f}")

    t_start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futs = [pool.submit(one, c) for c in todo]
            for f in as_completed(futs):
                f.result()
    finally:
        u = dict(det.usage)
        cost = estimate_cost(model_id, u['input_tokens'], u['output_tokens']) if u['calls'] \
            else None
        subset = args.keys if args.keys in ('smoke', 'all') else Path(args.keys).name
        rec = {
            'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'bundle': 'vancouver_corner_galleries_243_169',
            'label': f'{TAG}:{args.arm}:{subset}:{args.views}',
            'tag': TAG,
            'provider': 'gemini', 'model_id': model_id, 'paid': bool(u['calls']),
            'model_versions': dict(det.model_versions) if det.model_versions else None,
            'panos_scored': stats['written'], 'panos_called': stats['written'],
            'elapsed_s': round(time.time() - t_start, 1),
            'signature': {'task': 'corner_verdict_185', 'arm': args.arm, 'mode': mode,
                          'replicate': replicate, 'temperature': TEMPERATURE,
                          'prompt_sha256': s1.prompt_sha256(),
                          'requests_sha256': meta['requests_sha256']},
            'serving_path': 'vertex' if det.use_vertex else 'gemini_api',
            'api_account': None,
            'stop_reasons': dict(det.stop_reasons) or None,
            'run_options': {'concurrency': args.concurrency, 'keys': subset,
                            'views': args.views, 'budget_usd': args.budget_usd,
                            'no_answer': stats['no_answer'], 'errors': stats['errors'],
                            'stopped_on_budget': stop.is_set()},
            'pano_list': None,
            **u,
            'est_cost_usd': round(cost, 4) if cost is not None else None,
            'pricing': price_for(model_id),
        }
        if u['calls']:
            for path in (ledger.canonical_repo_root(RAMPNET) / 'analysis_out/usage_log.jsonl',
                         stage1 / 'usage_log.jsonl'):
                ledger.append_rows(path, [rec])
            print(f"[{args.arm}] {u['calls']} calls, {u['input_tokens']:,} in / "
                  f"{u['output_tokens']:,} out ({u['thoughts_tokens']:,} thinking), "
                  f"~${cost:.4f}; stop reasons {dict(det.stop_reasons)}; "
                  f"no_answer {stats['no_answer']}, errors {stats['errors']}")
    return 1 if stats['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
