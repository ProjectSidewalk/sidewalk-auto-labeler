"""Run the detector over a list of pano ids whose pixels are in a local pano store.

Why this exists (issue #56): a city's deployed AI labels can outlive the imagery they were
computed on. For Vancouver, a year after submission about a third of the labeled panos no
longer resolve by id on GSV and two thirds are gone from coverage, so neither `main.py`
(a coverage scan) nor `scripts/reinfer.py` (a by-id re-fetch) can rebuild the run. Project
Sidewalk's own pano store still holds the native-resolution JPEGs, and the server's
`/backupImage/<id>/metadata` still describes every pano. This runner joins the two:

  * pano ids   : `--ids <file>` and/or `--labels <rawLabels geojson>` (the panos carrying
                 labels), plus `--sample-unlabeled N --seed S` store panos that carry none
                 (the benchmark's empty stratum). The selection is written once to
                 `<run-dir>/store_ids.txt` + `store_selection.json` and reused on resume,
                 so a store that grows between sessions cannot change the sample.
  * pano block : GET <server>/backupImage/<id>/metadata, sequential and spaced (a 429's
                 Retry-After is honoured), cached one JSON per id under
                 `<run-dir>/store_metadata/` so a resume never re-GETs. The block has every
                 key sources/gsv.build_pano_record writes; what the server cannot supply
                 (links, history, source, source_metadata, depth) is [] / null,
                 `source_detail` is "ps_store", and `camera_height_status` is `no_depth` --
                 heights come from the store's depth artifacts via
                 `scripts/harvest_depth.py --from-store`.
  * pixels     : `<store>/<id[:2]>/<id>.jpg` (sharded) or `<store>/<id>.jpg` (flat),
                 autodetected, normalized by panorama.normalize_image -- the same clamp +
                 resize as the GSV download path.

Detection, records and resume are main.py's own (`main._process`, `main.handle_result`,
`main.build_output_line`, `already_processed.txt`), so the output is an ordinary
`results.jsonl`. The manifest is bound to the model like any run (`main.bind_model`), is
`imagery_source: gsv`, and carries a `pixels` block saying where the pixels and metadata
came from. Deterministic skips (no JPEG in the store, metadata 404, incomplete metadata,
bytes that are not an image) are cached and logged with their reason in
`store_skipped.jsonl`; network and HTTP errors are left uncached and retried next run.

    # 1. metadata only (no model, no GPU): fill the cache first, anywhere the store is
    python scripts/detect_from_store.py --run-dir runs/vancouver \\
        --store /projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa \\
        --labels runs/vancouver/provenance_gate/raw_labels.geojson --labels-user <ai uuid> \\
        --sample-unlabeled 300 --seed 56 \\
        --server https://sidewalk-vancouver.cs.washington.edu --metadata-only

    # 2. the detection run (same arguments, without --metadata-only)

Memory: a 16384x8192 JPEG decodes to ~400 MB before the resize, so --workers defaults to
8, not main.py's 50.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import depth as depthlib  # noqa: E402

SOURCE_DETAIL = 'ps_store'
API_METADATA = '/backupImage/{pano_id}/metadata'
METADATA_DIR = 'store_metadata'
SKIP_LOG = 'store_skipped.jsonl'
IDS_FILE = 'store_ids.txt'
SELECTION_FILE = 'store_selection.json'
DEFAULT_WORKERS = 8

# Same politeness as scripts/coverage_check.py's metadata probe: one request at a time,
# spaced, retrying a 5xx / 429 / connection error after a backoff.
REQUEST_SPACING_S = 0.2
REQUEST_BACKOFF_S = (2.0, 8.0)
RETRY_AFTER_CAP_S = 60.0
REQUEST_TIMEOUT_S = 30

# Deterministic skip reasons (cached, never retried).
SKIP_NO_JPEG = 'jpg_missing'
SKIP_METADATA_404 = 'metadata_404'
SKIP_METADATA_INCOMPLETE = 'metadata_incomplete'
SKIP_NOT_AN_IMAGE = 'jpg_unreadable'

# The pano-block keys the server must supply; anything else may be null.
REQUIRED_METADATA = ('lat', 'lng', 'width', 'height', 'cameraHeading')

_request_lock = threading.Lock()
_last_request = [0.0]


# ------------------------------------------------------------------------------ the store

def store_layout(store):
    """'sharded' (`<id[:2]>/<id>.jpg`, pano-tools' layout) or 'flat' (`<id>.jpg`)."""
    store = Path(store)
    if not store.is_dir():
        raise SystemExit(f'--store {store}: not a directory')
    with os.scandir(store) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.jpg'):
                return 'flat'
    return 'sharded'


def jpg_path(store, layout, pano_id):
    store = Path(store)
    return store / pano_id[:2] / f'{pano_id}.jpg' if layout == 'sharded' else store / f'{pano_id}.jpg'


def _is_pano_jpg(name):
    """A pano JPEG, not a sidecar: pano-tools writes `<id>.w<width>.jpg` display copies
    beside the originals, and a GSV pano id never contains a dot."""
    return name.endswith('.jpg') and '.' not in name[:-4]


def store_ids(store, layout):
    """Every pano id with a JPEG in the store, sorted (so a seeded sample is reproducible)."""
    store = Path(store)
    dirs = [store] if layout == 'flat' else sorted(p for p in store.iterdir() if p.is_dir())
    ids = []
    for d in dirs:
        with os.scandir(d) as it:
            ids.extend(e.name[:-4] for e in it if e.is_file() and _is_pano_jpg(e.name))
    return sorted(ids)


# -------------------------------------------------------------------------------- the ids

def read_id_file(path):
    """Pano ids, one per line; blank lines and `#` comments ignored."""
    ids = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def label_pano_ids(path, user=None):
    """Pano ids carrying labels in a /v3/api/rawLabels geojson, first-seen order. `user`
    restricts to one user_id (the AI account). Reads `pano_id` (v3) or `gsv_panorama_id`
    (older exports)."""
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    ids = []
    for ft in feats:
        q = ft.get('properties') or {}
        if user is not None and str(q.get('user_id')) != user:
            continue
        pid = q.get('pano_id') or q.get('gsv_panorama_id')
        if pid:
            ids.append(pid)
    return list(dict.fromkeys(ids))


def sample_unlabeled(all_store_ids, exclude, n, seed):
    """`n` store ids not in `exclude`, drawn with random.Random(seed) from the SORTED
    candidates, so the same store, list and seed always give the same sample."""
    pool = sorted(set(all_store_ids) - set(exclude))
    if n > len(pool):
        raise SystemExit(f'--sample-unlabeled {n}: only {len(pool)} unlabeled pano(s) in the store')
    return sorted(random.Random(seed).sample(pool, n))


def _sha256_lines(ids):
    return hashlib.sha256(''.join(f'{i}\n' for i in ids).encode('utf-8')).hexdigest()


def select_ids(run_dir, store, layout, input_ids, n_unlabeled, seed):
    """The run's pano list: the input ids followed by the unlabeled sample. Written once
    (store_ids.txt + store_selection.json) and REUSED on every later call, so a resume
    processes exactly the list the run started with. A resume with different inputs,
    sample size or seed is refused rather than silently forking the run."""
    ids_path, sel_path = run_dir / IDS_FILE, run_dir / SELECTION_FILE
    selection = {'input_ids_sha256': _sha256_lines(input_ids), 'n_input_ids': len(input_ids),
                 'sample_unlabeled': n_unlabeled, 'seed': seed}
    if sel_path.exists():
        recorded = json.loads(sel_path.read_text(encoding='utf-8'))
        for key, value in {**selection, 'store': str(store)}.items():
            if recorded.get(key) != value:
                raise SystemExit(
                    f'{run_dir} was started with a different pano selection ({key}: recorded '
                    f'{recorded.get(key)!r}, now {value!r}). Use a new --run-dir, or pass the '
                    f'arguments the run was started with.')
        ids = read_id_file(ids_path)
        if _sha256_lines(ids) != recorded['ids_sha256']:
            raise SystemExit(f'{ids_path} does not match the sha256 recorded in {sel_path.name}; '
                             f'it was edited after the run started.')
        return ids, recorded
    sample = []
    if n_unlabeled:
        sample = sample_unlabeled(store_ids(store, layout), input_ids, n_unlabeled, seed)
    ids = list(dict.fromkeys(list(input_ids) + sample))
    run_dir.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(''.join(f'{i}\n' for i in ids), encoding='utf-8', newline='\n')
    selection.update({'ids_sha256': _sha256_lines(ids), 'n_ids': len(ids),
                      'n_sampled_unlabeled': len(sample), 'store': str(store), 'layout': layout,
                      'selected_at': datetime.now(timezone.utc).isoformat(timespec='seconds')})
    sel_path.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    return ids, selection


# ---------------------------------------------------------------------------- the server

def _retry_after_s(response, default):
    try:
        return min(max(float(response.headers.get('Retry-After')), 0.0), RETRY_AFTER_CAP_S)
    except (AttributeError, TypeError, ValueError):
        return default


def _get_spaced(url):
    """One GET, serialized across worker threads and spaced REQUEST_SPACING_S from the
    previous one. Returns the response, or None when no answer came back."""
    with _request_lock:
        wait = _last_request[0] + REQUEST_SPACING_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return requests.get(url, timeout=REQUEST_TIMEOUT_S, allow_redirects=False)
        except requests.RequestException:
            return None
        finally:
            _last_request[0] = time.monotonic()


def fetch_metadata(server, pano_id, cache_dir):
    """('ok', dict) | ('skipped', reason) | ('failure', reason) for one pano's
    /backupImage/<id>/metadata, served from the cache when present.

    404 is deterministic (the server has no backup for this pano); a 5xx, a 429 or a
    connection error is retried after REQUEST_BACKOFF_S (a 429 honours Retry-After) and
    then left as a retryable failure; any other status, a redirect, or a 200 that is not
    a JSON object is a retryable failure too -- a login page served with 200 must never
    become a cached pano block."""
    cache = Path(cache_dir) / f'{pano_id}.json'
    if cache.exists():
        return 'ok', json.loads(cache.read_text(encoding='utf-8'))
    url = server.rstrip('/') + API_METADATA.format(pano_id=pano_id)
    status = None
    for attempt in range(len(REQUEST_BACKOFF_S) + 1):
        response = _get_spaced(url)
        status = None if response is None else response.status_code
        if status == 200:
            try:
                body = response.json()
            except ValueError:
                return 'failure', f'metadata: 200 but not JSON from {url}'
            if not isinstance(body, dict):
                return 'failure', f'metadata: 200 but not a JSON object from {url}'
            if body.get('panoId') not in (None, pano_id):
                return 'failure', f'metadata: asked for {pano_id}, got {body.get("panoId")}'
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(cache.name + '.part')
            tmp.write_text(json.dumps(body), encoding='utf-8')
            tmp.replace(cache)
            return 'ok', body
        if status == 404:
            return 'skipped', SKIP_METADATA_404
        if status is not None and status < 500 and status != 429:
            return 'failure', f'metadata: HTTP {status} from {url}'
        if attempt < len(REQUEST_BACKOFF_S):
            backoff = REQUEST_BACKOFF_S[attempt]
            time.sleep(_retry_after_s(response, backoff) if status == 429 else backoff)
    return 'failure', f'metadata: {"no answer" if status is None else f"HTTP {status}"} from {url}'


def _capture_date(value):
    """'YYYY-MM' from the server's captureDate, else None."""
    if isinstance(value, str) and len(value) >= 7 and value[4] == '-' \
            and value[:4].isdigit() and value[5:7].isdigit():
        return value[:7]
    return None


def _float_or_none(value):
    return None if value is None else float(value)


def pano_record_from_ps(pano_id, meta):
    """The results.jsonl pano block for a pano described only by Project Sidewalk's
    /backupImage/<id>/metadata. Same keys as sources/gsv.build_pano_record, in the same
    order; the values PS does not hold are [] / None, and `source_detail` says where the
    block came from. Raises KeyError / TypeError / ValueError on unusable metadata.

    Example:
        >>> b = pano_record_from_ps('P', {'lat': 45.6, 'lng': -122.5, 'width': 16384,
        ...     'height': 8192, 'cameraHeading': 90.0, 'captureDate': '2023-05'})
        >>> b['capture_date'], b['source_detail'], b['camera_height_status']
        ('2023-05', 'ps_store', 'no_depth')
    """
    for key in REQUIRED_METADATA:
        if meta.get(key) is None:
            raise KeyError(key)
    return {
        'panorama_id': pano_id,
        'capture_date': _capture_date(meta.get('captureDate')),
        'width': int(meta['width']),
        'height': int(meta['height']),
        'tile_width': meta.get('tileWidth'),
        'tile_height': meta.get('tileHeight'),
        'lat': float(meta['lat']),
        'lng': float(meta['lng']),
        'camera_heading': float(meta['cameraHeading']),
        'camera_pitch': _float_or_none(meta.get('cameraPitch')),
        'camera_roll': _float_or_none(meta.get('cameraRoll')),
        # The provider's own credit string when the server has it -- what sources/gsv.py
        # stores from streetlevel's copyright_message.
        'copyright': meta.get('copyright'),
        'source': None,
        'history': [],
        'links': [],
        'camera_make': None,
        'camera_model': None,
        'camera_type': 'equirectangular',
        'source_detail': SOURCE_DETAIL,
        'uploader': None,
        'source_metadata': None,
        **depthlib.camera_height_fields(None),
    }


# ---------------------------------------------------------------------------- per pano

def load_store_image(path):
    """(4096x2048 RGB PIL image, native (w, h)) for one store JPEG."""
    from PIL import Image
    import panorama
    # A 16384x8192 pano is 134 MP, past PIL's decompression-bomb warning (89 MP); these
    # are our own files, so the guard is off for this read.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        native = im.size
        image = im.convert('RGB')
    return panorama.normalize_image(image), native


def fetch_from_store(pano_id, store, layout, server, cache_dir):
    """main.py's source contract (`fetch_pano`), for a store pano: success with the pano
    block and image, or a deterministic 'skipped' / retryable 'failure' with a reason."""
    from PIL import UnidentifiedImageError
    path = jpg_path(store, layout, pano_id)
    if not path.is_file():
        return {'status': 'skipped', 'reason': SKIP_NO_JPEG}
    status, meta = fetch_metadata(server, pano_id, cache_dir)
    if status != 'ok':
        return {'status': status, 'reason': meta}
    try:
        pano = pano_record_from_ps(pano_id, meta)
    except (KeyError, TypeError, ValueError) as e:
        return {'status': 'skipped', 'reason': f'{SKIP_METADATA_INCOMPLETE}: {e}'}
    try:
        image, native = load_store_image(path)
    except UnidentifiedImageError:
        return {'status': 'skipped', 'reason': SKIP_NOT_AN_IMAGE}
    result = {'status': 'success', 'pano': pano, 'image': image}
    if native != (pano['width'], pano['height']):
        result['native_size_mismatch'] = True
    return result


def log_skip(run_dir, pano_id, reason):
    with open(run_dir / SKIP_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'pano_id': pano_id, 'reason': reason,
                            'at': datetime.now(timezone.utc).isoformat(timespec='seconds')}) + '\n')


def load_skips(run_dir):
    """{pano_id: reason} from store_skipped.jsonl (last reason wins)."""
    path = Path(run_dir) / SKIP_LOG
    skips = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                skips[row['pano_id']] = row['reason']
    return skips


# ---------------------------------------------------------------------------- manifest

def pixels_block(selection, server):
    return {'store': selection['store'], 'layout': selection['layout'],
            'server': server.rstrip('/'), 'metadata': API_METADATA,
            'ids_sha256': selection['ids_sha256'], 'n_ids': selection['n_ids'],
            'n_sampled_unlabeled': selection['n_sampled_unlabeled'],
            'seed': selection['seed'], 'source_detail': SOURCE_DETAIL}


def bind_manifest(run_dir, provenance, pixels):
    """Load or create runs/<name>/manifest.json for a store run and bind it: a GSV run,
    this code's storage floor, one model revision (main.bind_model) and one `pixels`
    block. An existing manifest from a scan (area hash, no runs) is adopted; one bound to
    another store, server or id list is refused."""
    import main
    from importlib.metadata import version as pkg_version
    path = run_dir / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('imagery_source', 'gsv') != 'gsv':
            raise SystemExit(f"{run_dir.name} is a '{manifest['imagery_source']}' run; a store "
                             f"run writes GSV records. Use a new --run-dir.")
        floor = manifest.get('detection_storage_floor', 0.55)
        if floor != main.DETECTION_STORAGE_FLOOR:
            raise SystemExit(f'{run_dir.name} stores detections down to {floor}; this code '
                             f'stores down to {main.DETECTION_STORAGE_FLOOR}. Use a new --run-dir.')
        bound = manifest.get('pixels')
        if bound is not None and bound != pixels:
            diff = sorted(k for k in set(bound) | set(pixels) if bound.get(k) != pixels.get(k))
            raise SystemExit(f'{run_dir.name} is bound to other pixels ({", ".join(diff)} '
                             f'differ). Use a new --run-dir.')
        main.bind_model(manifest, provenance, run_dir.name)
        manifest['pixels'] = pixels
        manifest.setdefault('runs', [])
    else:
        manifest = {
            'run_name': run_dir.name,
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'imagery_source': 'gsv',
            **main.manifest_model_block(provenance),
            'detection_storage_floor': main.DETECTION_STORAGE_FLOOR,
            'max_peaks_per_pano': main.MAX_PEAKS_PER_PANO,
            'streetlevel_version': pkg_version('streetlevel'),
            'pixels': pixels,
            'runs': [],
        }
    main.save_manifest(path, manifest)
    return manifest


# --------------------------------------------------------------------------------- runs

def prefetch_metadata(run_dir, ids, store, layout, server):
    """--metadata-only: fill store_metadata/ without loading a model. Deterministic skips
    are cached exactly as the detection pass would cache them."""
    import main
    done = main.load_processed_ids(run_dir / 'already_processed.txt')
    cache_dir = run_dir / METADATA_DIR
    counts = {'cached': 0, 'fetched': 0, 'skipped': 0, 'failed': 0}
    with open(run_dir / 'already_processed.txt', 'a', encoding='utf-8') as f_cache:
        for pid in ids:
            if pid in done:
                continue
            if (cache_dir / f'{pid}.json').exists():
                counts['cached'] += 1
                continue
            if not jpg_path(store, layout, pid).is_file():
                status, reason = 'skipped', SKIP_NO_JPEG
            else:
                status, reason = fetch_metadata(server, pid, cache_dir)
            if status == 'ok':
                counts['fetched'] += 1
            elif status == 'skipped':
                counts['skipped'] += 1
                log_skip(run_dir, pid, reason)
                f_cache.write(f'{pid}\n')
                f_cache.flush()
            else:
                counts['failed'] += 1
                print(f'  {pid}: {reason} (will retry)')
    print(f'-> metadata: {counts}')
    return counts


def run(run_dir, ids, store, layout, server, selection, workers, limit):
    """The detection pass. Loads the model, binds the manifest, and appends to
    results.jsonl through main.handle_result in id-list order."""
    from concurrent.futures import ThreadPoolExecutor
    import main
    from detectors import ModelProvenanceError
    if getattr(main, 'curb_ramp_detector', None) is None:
        from detectors.curb_ramp import CurbRampDetector
        try:
            main.curb_ramp_detector = CurbRampDetector()
        except ModelProvenanceError as e:
            raise SystemExit(str(e))
    provenance = main.curb_ramp_detector.provenance
    manifest = bind_manifest(run_dir, provenance, pixels_block(selection, server))
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')

    done = main.load_processed_ids(run_dir / 'already_processed.txt')
    todo = [pid for pid in ids if pid not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f'-> {len(ids)} ids, {len(set(ids) & done)} already done or skipped, '
          f'{len(todo)} to process -> {run_dir / "results.jsonl"}')

    cache_dir = run_dir / METADATA_DIR
    counts = {'success': 0, 'skipped': 0, 'failed': 0}
    size_mismatch = 0

    def work(pid):
        box = {}

        def thunk():
            fetched = fetch_from_store(pid, store, layout, server, cache_dir)
            box.update(fetched)
            return fetched
        result = main._process(pid, thunk)
        result['native_size_mismatch'] = box.get('native_size_mismatch', False)
        return result

    with open(run_dir / 'already_processed.txt', 'a', encoding='utf-8') as f_cache, \
         open(run_dir / 'results.jsonl', 'a', encoding='utf-8') as f_out, \
         ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, pid) for pid in todo]
        try:
            from tqdm import tqdm
            it = tqdm(futures, desc='Detecting from store')
        except ImportError:
            it = futures
        for fut in it:
            result = fut.result()
            if result['status'] == 'skipped':
                log_skip(run_dir, result['pano_id'], result.get('reason', 'unknown'))
            outcome = main.handle_result(result, f_cache, f_out, provenance)
            counts[outcome] += 1
            size_mismatch += outcome == 'success' and result['native_size_mismatch']

    main.record_run(run_dir / 'manifest.json', manifest, started, len(todo), counts['success'],
                    counts['skipped'], counts['failed'], phase='store', provenance=provenance)
    print(f"-> {counts['success']} written, {counts['skipped']} skipped (see {SKIP_LOG}), "
          f"{counts['failed']} failed (re-run to retry).")
    if size_mismatch:
        print(f'-> NOTE: {size_mismatch} store JPEG(s) differ in size from the server\'s '
              f'width/height; their detections are still keyed by the server\'s dimensions.')
    return counts


def main_cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-dir', type=Path, required=True, help='runs/<name>')
    ap.add_argument('--store', type=Path, required=True,
                    help='pano store root: <id[:2]>/<id>.jpg or <id>.jpg (autodetected)')
    ap.add_argument('--server', required=True,
                    help='Project Sidewalk server serving /backupImage/<id>/metadata')
    ap.add_argument('--ids', type=Path, help='file of pano ids, one per line')
    ap.add_argument('--labels', type=Path, help='a /v3/api/rawLabels geojson; its pano ids')
    ap.add_argument('--labels-user', help='with --labels: only this user_id\'s labels')
    ap.add_argument('--sample-unlabeled', type=int, default=0, metavar='N',
                    help='also process N store panos that are not in the id list')
    ap.add_argument('--seed', type=int, default=0, help='seed for --sample-unlabeled')
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--limit', type=int, help='process at most this many new panos')
    ap.add_argument('--metadata-only', action='store_true',
                    help='only fetch and cache the server metadata; no model is loaded')
    args = ap.parse_args(argv)

    if not args.ids and not args.labels:
        ap.error('give --ids and/or --labels')
    if args.labels_user and not args.labels:
        ap.error('--labels-user needs --labels')
    if args.sample_unlabeled < 0 or (args.limit is not None and args.limit < 0):
        ap.error('--sample-unlabeled and --limit must be >= 0')
    input_ids = []
    if args.ids:
        input_ids += read_id_file(args.ids)
    if args.labels:
        input_ids += label_pano_ids(args.labels, args.labels_user)
    input_ids = list(dict.fromkeys(input_ids))
    if not input_ids:
        raise SystemExit('the id list is empty')

    layout = store_layout(args.store)
    ids, selection = select_ids(args.run_dir, args.store, layout, input_ids,
                                args.sample_unlabeled, args.seed)
    print(f'-> store {args.store} ({layout}); {selection["n_ids"]} pano ids '
          f'({selection["n_sampled_unlabeled"]} sampled unlabeled, seed {selection["seed"]})')
    if args.metadata_only:
        prefetch_metadata(args.run_dir, ids, args.store, layout, args.server)
        return
    run(args.run_dir, ids, args.store, layout, args.server, selection, args.workers, args.limit)


if __name__ == '__main__':
    main_cli()
