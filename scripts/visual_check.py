"""Plot detections (normalized -> pixel) on the full-res pano to verify the coordinate chain."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from streetlevel import streetview

jsonl_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

record = None
for line in jsonl_path.read_text().splitlines():
    r = json.loads(line)
    if r['detections']:
        record = r
        break
if record is None:
    print("NO_DETECTIONS_FOUND")
    sys.exit(0)

pano = record['pano']
print("pano:", pano['panorama_id'], "detections:", len(record['detections']),
      "full-res:", pano['width'], "x", pano['height'])

md = streetview.find_panorama_by_id(pano['panorama_id'])
zoom = min(3, len(md.image_sizes) - 1)
img = streetview.get_panorama(md, zoom=zoom)
print("downloaded:", img.size)

# Scale full-res pixel coords (what send_to_ps submits) down to this zoom level's image.
sx = img.size[0] / pano['width']
sy = img.size[1] / pano['height']

draw = ImageDraw.Draw(img)
for det in record['detections']:
    px = det['x_normalized'] * pano['width'] * sx
    py = det['y_normalized'] * pano['height'] * sy
    r = 40
    draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0), width=8)
    draw.text((px + r + 5, py - r), f"{det['confidence']:.2f}", fill=(255, 0, 0))

img.save(out_path)
print("saved:", out_path)
