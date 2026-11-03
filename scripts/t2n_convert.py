"""Convert a Tile2Net project's GeoParquet outputs to GeoJSON + a provenance stub (#104).
Runs in the tile2net venv on makelab2. Usage: python t2n_convert.py <project_dir> <out_dir>"""
import glob, hashlib, json, sys
from pathlib import Path
import geopandas as gpd

proj, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rec = {}
for kind in ('polygons', 'network'):
    files = sorted(glob.glob(str(proj / kind / '*.parquet')))
    if not files:
        continue
    g = gpd.read_parquet(files[0]).to_crs(4326)
    keep = [c for c in ('f_type', 'geometry') if c in g.columns]
    g = g[keep]
    dest = out / f'{kind}.geojson'
    g.to_file(dest, driver='GeoJSON', COORDINATE_PRECISION=7)
    rec[f'{kind}_parquet'] = files[0]
    rec[f'{kind}_parquet_sha256'] = hashlib.sha256(Path(files[0]).read_bytes()).hexdigest()
    rec[f'{kind}_geojson_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    rec[f'{kind}_geojson_bytes'] = dest.stat().st_size
    rec[f'{kind}_features'] = int(len(g))
    if 'f_type' in g.columns:
        m = g.to_crs(g.estimate_utm_crs())
        rec[f'{kind}_by_class'] = {k: {'n': int((m.f_type == k).sum()),
                                       ('area_m2' if kind == 'polygons' else 'length_m'):
                                       round(float((m[m.f_type == k].area if kind == 'polygons'
                                                    else m[m.f_type == k].length).sum()))}
                                   for k in sorted(m.f_type.unique())}
(out / 'convert.json').write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
