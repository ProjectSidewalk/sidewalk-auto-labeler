"""
Render an HTML contact sheet of annotated detections for visual QA.

Reads a run directory produced by main.py (or a bare results JSONL file), samples
panoramas, re-downloads each at up to 4096x2048, draws the detections, and writes an
index.html with a downscaled annotated panorama plus a close-up crop per detection.

The sample always includes the densest panos (most detections), a random sample of
panos with detections, and a few zero-detection panos (to eyeball false negatives).

Usage:
    python scripts/spot_check_gallery.py runs/bend
    python scripts/spot_check_gallery.py runs/bend --sample 100 --empty-sample 10
"""
import argparse
import json
import random
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw
from streetlevel import streetview
from tqdm import tqdm

socket.setdefaulttimeout(30)

DISPLAY_WIDTH = 1600   # displayed full-pano width in the gallery
CROP_SIZE = 512        # per-detection close-up, taken at download resolution
TOP_N_BY_COUNT = 5     # the densest panos are always included
DOWNLOAD_WORKERS = 8


def load_records(path_arg):
    path = Path(path_arg)
    jsonl_path = path / "results.jsonl" if path.is_dir() else path
    if not jsonl_path.exists():
        sys.exit(f"No results file found at {jsonl_path}")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    return jsonl_path, records


def choose_panos(records, sample, empty_sample, seed):
    rng = random.Random(seed)
    with_det = [r for r in records if r['detections']]
    without_det = [r for r in records if not r['detections']]

    chosen = sorted(with_det, key=lambda r: len(r['detections']), reverse=True)[:TOP_N_BY_COUNT]
    chosen_ids = {r['pano']['panorama_id'] for r in chosen}
    rest = [r for r in with_det if r['pano']['panorama_id'] not in chosen_ids]
    if sample > len(chosen) and rest:
        chosen += rng.sample(rest, min(sample - len(chosen), len(rest)))
    if without_det and empty_sample:
        chosen += rng.sample(without_det, min(empty_sample, len(without_det)))
    return chosen


def render_pano(record, images_dir):
    """Downloads one pano, draws its detections, and saves the gallery images."""
    pano = record['pano']
    pid = pano['panorama_id']
    metadata = streetview.find_panorama_by_id(pid)
    if metadata is None:
        raise RuntimeError("metadata unavailable")
    img = streetview.get_panorama(metadata, zoom=min(3, len(metadata.image_sizes) - 1))

    # Detections are stored normalized; scale to this download's resolution.
    sx, sy = img.size[0], img.size[1]

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    crops = []
    for i, det in enumerate(record['detections']):
        px, py = det['x_normalized'] * sx, det['y_normalized'] * sy
        radius = 40
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                     outline=(255, 0, 0), width=8)

        half = CROP_SIZE // 2
        left = int(min(max(px - half, 0), img.size[0] - CROP_SIZE))
        top = int(min(max(py - half, 0), img.size[1] - CROP_SIZE))
        crop = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
        crop_draw = ImageDraw.Draw(crop)
        crop_draw.ellipse([px - left - radius, py - top - radius,
                           px - left + radius, py - top + radius],
                          outline=(255, 0, 0), width=6)
        crop_name = f"{pid}_det{i}.jpg"
        crop.convert('RGB').save(images_dir / crop_name, quality=85)
        crops.append((crop_name, det['confidence']))

    full_name = f"{pid}_full.jpg"
    annotated.resize((DISPLAY_WIDTH, DISPLAY_WIDTH // 2), Image.BILINEAR) \
             .convert('RGB').save(images_dir / full_name, quality=80)
    return {'record': record, 'full': full_name, 'crops': crops}


def build_html(entries, jsonl_path):
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Detection spot check</title>",
        "<style>"
        "body{font-family:sans-serif;margin:20px;background:#fafafa}"
        ".pano{margin-bottom:36px;padding:12px;background:#fff;border:1px solid #ddd;border-radius:8px}"
        ".crops{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}"
        ".crops figure{margin:0;text-align:center;font-size:13px}"
        ".crops img{width:256px;height:256px}"
        "h2{font-size:16px;margin:0 0 8px}a{color:#06c}.meta{color:#666;font-size:13px}"
        "</style>",
        f"<h1>Detection spot check</h1><p class='meta'>Source: {escape(str(jsonl_path))}</p>",
    ]
    for entry in entries:
        pano = entry['record']['pano']
        pid = pano['panorama_id']
        gsv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&pano={pid}"
        n = len(entry['record']['detections'])
        parts.append(
            f"<div class='pano'><h2><a href='{gsv_url}'>{escape(pid)}</a> "
            f"<span class='meta'>captured {escape(str(pano['capture_date']))} — "
            f"{n} detection{'s' if n != 1 else ''}</span></h2>"
            f"<img src='images/{entry['full']}' width='100%'>"
        )
        if entry['crops']:
            parts.append("<div class='crops'>")
            for name, conf in entry['crops']:
                parts.append(f"<figure><img src='images/{name}'>"
                             f"<figcaption>conf {conf:.2f}</figcaption></figure>")
            parts.append("</div>")
        parts.append("</div>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Render an HTML contact sheet of annotated detections for visual QA."
    )
    parser.add_argument("run", help="Run directory from main.py (or a results JSONL file).")
    parser.add_argument("--sample", type=int, default=100,
                        help="Number of panos with detections to include (default: %(default)s).")
    parser.add_argument("--empty-sample", type=int, default=10,
                        help="Number of zero-detection panos to include (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed, for a reproducible gallery (default: %(default)s).")
    parser.add_argument("--out", type=Path,
                        help="Output directory (default: <run dir>/spot_check).")
    args = parser.parse_args()

    jsonl_path, records = load_records(args.run)
    chosen = choose_panos(records, args.sample, args.empty_sample, args.seed)
    print(f"{len(records)} records; rendering {len(chosen)} panos "
          f"({sum(1 for r in chosen if r['detections'])} with detections).")

    out_dir = args.out or jsonl_path.parent / "spot_check"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(render_pano, r, images_dir): r for r in chosen}
        with tqdm(total=len(futures), desc="Rendering panoramas") as pbar:
            for future in as_completed(futures):
                pid = futures[future]['pano']['panorama_id']
                try:
                    entries.append(future.result())
                except Exception as e:
                    print(f"  skipped {pid}: {e}")
                pbar.update(1)

    entries.sort(key=lambda e: len(e['record']['detections']), reverse=True)
    index_path = out_dir / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(build_html(entries, jsonl_path))
    print(f"Gallery: {index_path}")


if __name__ == "__main__":
    main()
