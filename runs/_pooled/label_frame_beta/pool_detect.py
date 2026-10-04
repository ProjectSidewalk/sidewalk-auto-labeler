"""Detect on the PS pano store's JPEGs for a list of (city, pano_id) -- issue #113.

Scratch runner for the beta fit on the vouched CurbRamp pool
(sidewalk-panorama-tools reports/data/2026-09-29-tilt-jm-pool.csv.gz). Pixels only: the
pose comes from pano-tools' own scan, so no metadata is fetched and nothing is written
into any run directory. Same preprocessing as production (panorama.normalize_image) and
the same detector. Resumable: ids already in the output are skipped.

    python pool_detect.py curbramp_panos.txt pool_detections.jsonl
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = '/homes/gws/jonf/sidewalk-auto-labeler-py312'
STORE = Path('/projects/makeabilitylab/sidewalk_panos/Panoramas')
sys.path.insert(0, REPO)

from PIL import Image  # noqa: E402

import panorama  # noqa: E402
from detectors.curb_ramp import CurbRampDetector  # noqa: E402

Image.MAX_IMAGE_PIXELS = None   # 16384x8192 store JPEGs trip PIL's bomb guard


def main(ids_path, out_path):
    todo = [tuple(line.rstrip('\n').split('\t')) for line in open(ids_path) if line.strip()]
    done = set()
    if Path(out_path).exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r['city'], r['pano_id']))
    todo = [t for t in todo if t not in done]
    print(f'{len(done)} done, {len(todo)} to do', flush=True)

    det = CurbRampDetector()
    Path(out_path + '.meta.json').write_text(json.dumps(
        {'provenance': det.provenance, 'store': str(STORE), 'repo': REPO}, indent=2))
    lock = threading.Lock()
    n = [0, 0]
    t0 = time.time()

    def one(item):
        city, pid = item
        path = STORE / city / pid[:2] / f'{pid}.jpg'
        rec = {'city': city, 'pano_id': pid}
        try:
            with Image.open(path) as im:
                rec['native_w'], rec['native_h'] = im.size
                image = panorama.normalize_image(im.convert('RGB'))
            rec['detections'] = [[round(x, 6), round(y, 6), round(c, 6)]
                                 for x, y, c in det.detect(image)]
        except Exception as e:   # a missing or corrupt JPEG must not stop the pass
            rec['error'] = f'{type(e).__name__}: {e}'
        with lock:
            out.write(json.dumps(rec) + '\n')
            out.flush()
            n[0] += 1
            n[1] += 'error' in rec
            if n[0] % 200 == 0:
                rate = n[0] / (time.time() - t0)
                print(f'{n[0]}/{len(todo)}  errors {n[1]}  {rate:.2f} panos/s', flush=True)

    with open(out_path, 'a') as out, ThreadPoolExecutor(6) as pool:
        list(pool.map(one, todo))
    det.close()
    print(f'finished: {n[0]} written, {n[1]} errors, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
