"""External placement oracle for the camera-height default (issue #79).

The camera height every raycast uses (geo.DEFAULT_CAMERA_HEIGHT_M = 2.6 m) cannot be
checked against RampNet ground truth: GT marks are pixels projected through the same
raycast, so every height model moves the truth along with the prediction. A city's own
curb-ramp inventory -- one surveyed or digitized point per corner ramp, placed by people
who never saw our imagery -- does not. Bend and Gainesville publish one; Paterson and
Sao Paulo do not (the portals checked are listed in NO_INVENTORY and in
docs/placement-oracle.md).

Three camera-height arms, all GSV, all at OPERATIONAL_CONFIDENCE with the rig mask on:

  (a) `a`  2.6 m everywhere (what ships).
  (b) `b`  per-pano: each measured depth height x 1.08 (geo.PER_PANO on the scaled
           heights; unmeasured panos fall back to 2.6 m). `b1.00` (unscaled) is reported
           as a reference row, never a candidate.
  (c) `c`  per-rig: every pano of a vintage (this run, capture year) whose median measured
           depth height is < 2.1 m raycasts at 2.0 m, every other dated pano at 2.5 m --
           unmeasured panos included, since a rig is not a payload. A vintage with no
           measured pano, or an undated pano, falls back to 2.6 m.

Scoring (the #42 precondition design, eval_sites.refit_frozen):

- FROZEN association, the primary frame: fuse once under (a); keep that membership; each
  arm re-places every operational member and re-solves the site. A site counts only if
  every arm places every operational member at the 25 m cap. `frozen@b` / `frozen@c`
  repeat it with membership from (b)'s / (c)'s own fuse (decision rule 3).
- The (a)-ANCHORED POOL: inventory points matched one-to-one to a site under (a) within
  the radius. Frozen membership keeps site identity, so each arm's distance is to the SAME
  site for the SAME point: median, p90, share within 3 m. The match truncates (a)'s own
  errors at the radius and no one else's, so the pool is conservative against both
  candidates -- in frozen@b / frozen@c too. `score --pool-anchor frame` re-anchors
  frozen@X's pool on X, as a sensitivity check (docs/placement-oracle.md).
- OWN-MATCH per arm: that arm's sites matched to the inventory on their own -- coverage
  (matched / kept points) and share within 3 m; the survivorship check.
- OWN association (`own` frame): each arm fused under itself, own-match only. It favours
  each arm by construction and is reported beside, and read by rule 3.
- Along-ray offset per capture year: for single-vintage pool sites, the signed offset of
  the site from its inventory point along the mean member ray (+ = site beyond the point
  = ranges run long), and that offset over the mean member range.
- Chance floor: the kept points displaced 25 m in a seeded random direction and matched
  again (agree_rate.chance_floor), the share that still matches.

The decision rule is PRE-REGISTERED on #79 (constants below, `verdict()`), fixed before
the first scoring run. It is applied to the committed arms.csv / vintage.csv; this script
never changes the default -- a pass is a follow-up PR.

Network: `fetch` only, read-only GETs to the two ArcGIS hosts in INVENTORIES, paged with
resultOffset. The pull is cached as runs/<city>/inventory_oracle/inventory.geojson
(untracked) and described by inventory.json (tracked: url, query, fetch time, sha256,
counts, field histograms); a re-run reuses it, --refresh re-pulls. Everything else is
offline. Stdlib + shapely, like agree_rate.py and position_check.py.

Usage:
    python scripts/inventory_oracle.py fetch bend gainesville vancouver
    python scripts/inventory_oracle.py characterize bend
    python scripts/inventory_oracle.py score bend gainesville
    python scripts/inventory_oracle.py score gainesville --tier 0.55 --radius 2.5 5 8 \\
        --scale 1.06 1.10 --rig-cut 2.0 2.2 --out /tmp/sens     # sensitivity, not committed
    python scripts/inventory_oracle.py score bend gainesville --pool-anchor frame         --out /tmp/anchor      # frozen@X's pool on X: the rule-3 anchoring sensitivity
    python scripts/inventory_oracle.py verdict        # always gainesville (decides) + bend
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
# agree_rate.match_one_to_one is eval_sites.match_one_to_one grid-accelerated (identical
# output, tested there); chance_floor displaces points CHANCE_SHIFT_M (25 m), seed 31.
from agree_rate import CHANCE_SEED, CHANCE_SHIFT_M, Pt, chance_floor  # noqa: E402,F401
from agree_rate import match_one_to_one  # noqa: E402
from detectors import OPERATIONAL_CONFIDENCE  # noqa: E402
from height_qc import LOW_VINTAGE_M, OPTION_B_SCALE, OPTION_C  # noqa: E402

OUT_NAME = 'inventory_oracle'
USER_AGENT = ('sidewalk-auto-labeler inventory_oracle '
              '(+https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/79)')
PAGE_MAX = 1000          # records per request, below every layer's maxRecordCount
PAGE_PAUSE_S = 0.5       # polite gap between pages
MAX_PAGES = 200
HISTOGRAM_MAX_DISTINCT = 25

# --- the registry: where each city's inventory lives -----------------------------------

INVENTORIES = {
    'bend': {
        'status': 'ok',
        'url': 'https://services5.arcgis.com/JisFYcK2mIVg9ueP/arcgis/rest/services/'
               'sCurbRamps/FeatureServer/0',
        'portal': 'https://data.bendoregon.gov/datasets/curb-ramps',
        'item': 'aa70ed04816040ea9c6dd5de6789448c',
        'publisher': 'City of Bend, OR (CityofBendOR)',
        # installed ramps only; D / DC / PROP (demolished, decommissioned, proposed) and the
        # rest are kept in the cache and counted, but never scored
        'keep_field': 'LifeCycleStatus', 'keep_values': ('I',),
        'fields': ('LifeCycleStatus', 'Direction', 'Method', 'Owner', 'Responsibility',
                   'StreetClass', 'Compliant', 'Source'),
        'characterize_fields': ('Method', 'Direction', 'StreetClass'),
        'terms': 'public item; City disclaimer (reference purposes only, not for survey '
                 'or engineering); no explicit open licence -- cited, used for analysis',
    },
    'gainesville': {
        'status': 'ok',
        'url': 'https://services2.arcgis.com/Zzhtlau4ccHkQgTu/arcgis/rest/services/'
               'PublicWorksInfrastructure_AGO/FeatureServer/3',
        'portal': 'https://data.cityofgainesville.org/ (dataGNV "ADA Ramps", 3um4-3vb3)',
        'item': '817b0156f40a4cb8a1ccdb88510d6dce',
        'publisher': 'City of Gainesville, FL Public Works',
        'keep_field': 'LIFECYCLE', 'keep_values': ('Active',),
        'fields': ('LIFECYCLE', 'CORNER', 'OWNEDBY', 'MAINTBY', 'CROSSWALK', 'SIDEWALK'),
        'characterize_fields': ('OWNEDBY', 'CORNER', 'CROSSWALK'),
        'terms': 'Public Domain on the dataGNV Socrata mirror; informational-use '
                 'disclaimer on the service',
    },
    'vancouver': {
        # Known (issue #56: 16,960 points), but the service has answered "Service ... not
        # started" on every probe (2026-09-06, 2026-09-26). Not fetched: `fetch vancouver`
        # records the status only. TODO(#56): once it answers, set status 'ok' and fill
        # keep_field/keep_values/fields from its schema.
        'status': 'service not started (probed 2026-09-06 and 2026-09-26)',
        'url': 'https://utility.arcgis.com/usrsvcs/servers/90f910b76e3c43e0acd3cfd623a3818c/'
               'rest/services/PublicWorks/transSidewalkPUB/MapServer/0',
        'publisher': 'City of Vancouver, WA',
    },
}
# Checked, no curb-ramp inventory (2026-09-26). Recorded so the negative is reproducible.
NO_INVENTORY = {
    'paterson': (
        'Passaic County ArcGIS (gis.passaiccountynj.org, 76 hosted services: parks, '
        'facilities, parcels, one pedestrian-bridges layer)',
        'NJDOT open-data portal + ArcGIS Online org HggmsDF7UJsNN1FK (roads LRS, '
        'NJ_Sidewalks polylines, crash data; no ramp layer)',
        'NJGIN hub',
        'ArcGIS Online item search "Paterson" (nothing)',
        'the City of Paterson has no GIS portal',
    ),
    'sao_paulo': (
        'GeoSampa WFS (wfs.geosampa.prefeitura.sp.gov.br, 483 layers): `calcada` is '
        'sidewalk polygons with width/slope per segment, no ramp geometry; '
        '`acessibilidade_smped` is 972 establishments holding an accessibility seal, not '
        'ramps; no rampa/rebaixamento layer',
    ),
}
ALLOWED_HOSTS = {urllib.parse.urlparse(v['url']).hostname
                 for v in INVENTORIES.values() if v['status'] == 'ok'}

# --- arms and the pre-registered rule (issue #79; fixed before the first scoring run) ---

ARM_A, ARM_B, ARM_C, ARM_B_REF = 'a', 'b', 'c', 'b1.00'
DECISION_ARMS = (ARM_A, ARM_B, ARM_C)
CANDIDATES = (ARM_B, ARM_C)
PRIMARY_RADIUS_M = 5.0
RADII_M = (2.5, 5.0, 8.0)
WITHIN_M = 3.0
FROZEN = 'frozen@'          # frame name prefix: membership from that arm's own fuse
OWN = 'own'
POOL_FRAME = 'frame'        # score --pool-anchor frame: frozen@X's pool on X (sensitivity)
DECIDING_CITY = 'gainesville'   # 64% 2026 rig: the only city that can show the effect
GUARD_CITY = 'bend'             # 84% 2024: old-rig no-regression
DECIDING_VINTAGE = '2026'
RULE_MARGIN_M = 0.10            # rules 1, 2, 6 (the #42 tolerance)
RULE_MAX_COVERAGE_DROP = 0.010  # rule 4: own-match coverage within 1.0 pt of (a)'s
RULE_MAX_SITE_DROP = 0.05       # rule 4: sites lost to the every-arm-places filter
RULE_DIRECTION_M = 0.75         # rule 5: |2026 median along-ray offset| under X


def arm_label(arm):
    return {ARM_A: '(a) 2.6 m', ARM_B: f'(b) per-pano x {OPTION_B_SCALE:.2f}',
            ARM_C: '(c) per-rig 2.0/2.5', ARM_B_REF: 'per-pano x 1.00 (ref)'}.get(arm, arm)


# ----------------------------------------------------------------------------- fetch

def http_get_json(url, params=None, timeout=120):
    """GET url?params and parse JSON. Refuses any host outside ALLOWED_HOSTS."""
    host = urllib.parse.urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise ValueError(f'refusing to fetch from {host}: not an inventory host')
    if params:
        url = url + ('&' if '?' in url else '?') + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed hosts)
        body = json.loads(r.read().decode('utf-8'))
    if isinstance(body, dict) and 'error' in body:
        raise RuntimeError(f'{url}: ArcGIS error {body["error"]}')
    return body


def _exceeded(resp):
    """ArcGIS puts exceededTransferLimit at the top level (f=json) or under properties
    (f=geojson); either counts."""
    return bool(resp.get('exceededTransferLimit')
                or (resp.get('properties') or {}).get('exceededTransferLimit'))


def fetch_features(layer_url, bbox, get=http_get_json, pause_s=PAGE_PAUSE_S):
    """Every feature of an ArcGIS layer intersecting bbox = (minx, miny, maxx, maxy) in
    WGS84, as GeoJSON features, paged with resultOffset. Returns (features, info).

    The server count for the same envelope is read first and the pages must add up to it
    exactly, with unique object ids: a page that came back short, a layer that changed
    under the paging, or an `exceededTransferLimit` the paging never resolved (the server
    says there is more and then returns nothing) all refuse rather than cache a partial
    inventory -- a silently truncated oracle would read as missing ramps.
    """
    meta = get(layer_url, {'f': 'json'})
    oid = meta.get('objectIdField') or 'OBJECTID'
    page = min(PAGE_MAX, meta.get('maxRecordCount') or PAGE_MAX)
    envelope = {'xmin': bbox[0], 'ymin': bbox[1], 'xmax': bbox[2], 'ymax': bbox[3],
                'spatialReference': {'wkid': 4326}}
    common = {'where': '1=1', 'geometry': json.dumps(envelope),
              'geometryType': 'esriGeometryEnvelope', 'inSR': 4326,
              'spatialRel': 'esriSpatialRelIntersects'}
    query = layer_url.rstrip('/') + '/query'
    expected = get(query, {**common, 'returnCountOnly': 'true', 'f': 'json'})['count']
    layer_total = get(query, {'where': '1=1', 'returnCountOnly': 'true', 'f': 'json'})['count']
    feats, offset, pages = [], 0, 0
    while True:
        if pages >= MAX_PAGES:
            raise RuntimeError(f'{layer_url}: more than {MAX_PAGES} pages; refusing')
        resp = get(query, {**common, 'outFields': '*', 'outSR': 4326,
                           'orderByFields': f'{oid} ASC', 'resultOffset': offset,
                           'resultRecordCount': page, 'f': 'geojson'})
        pages += 1
        got = resp.get('features') or []
        more = _exceeded(resp)
        if not got:
            if more:
                raise RuntimeError(f'{layer_url}: exceededTransferLimit at offset {offset} '
                                   'but an empty page -- paging did not resolve; refusing')
            break
        feats.extend(got)
        offset += len(got)
        if not more and len(got) < page:
            break
        if pause_s:
            time.sleep(pause_s)
    if len(feats) != expected:
        raise RuntimeError(f'{layer_url}: paged {len(feats)} features but the server counts '
                           f'{expected} in the envelope; refusing a partial pull')
    ids = [f.get('id', (f.get('properties') or {}).get(oid)) for f in feats]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f'{layer_url}: duplicate object ids across pages; refusing')
    info = {'layer_name': meta.get('name'), 'max_record_count': meta.get('maxRecordCount'),
            'page_size': page, 'pages': pages, 'object_id_field': oid,
            'last_edit_date_ms': (meta.get('editingInfo') or {}).get('lastEditDate'),
            'layer_total': layer_total, 'in_bbox': len(feats)}
    return feats, info


def load_area(run_dir):
    """The run's area polygon (runs/<city>/area.geojson, a bare geometry) as shapely."""
    from shapely.geometry import shape
    with open(Path(run_dir) / 'area.geojson', encoding='utf-8') as f:
        return shape(json.load(f))


def in_area(features, area):
    """Features whose point the area polygon covers (boundary included)."""
    from shapely.geometry import shape
    from shapely.prepared import prep
    pa = prep(area)
    return [f for f in features if f.get('geometry')
            and pa.covers(shape(f['geometry']).representative_point())]


def kept(features, entry):
    """The features the scoring uses: keep_field in keep_values (e.g. installed ramps)."""
    field, values = entry['keep_field'], set(entry['keep_values'])
    return [f for f in features if (f.get('properties') or {}).get(field) in values]


def histograms(features, fields):
    """{field: {value: count}} for categorical fields, or {'distinct': n} when a field has
    more than HISTOGRAM_MAX_DISTINCT values."""
    out = {}
    for name in fields:
        counts = {}
        for f in features:
            v = (f.get('properties') or {}).get(name)
            counts[str(v)] = counts.get(str(v), 0) + 1
        out[name] = (dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
                     if len(counts) <= HISTOGRAM_MAX_DISTINCT else {'distinct': len(counts)})
    return out


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def out_dir_for(city, out=None):
    return Path(out) / city if out else REPO_ROOT / 'runs' / city / OUT_NAME


def fetch_city(city, refresh=False, get=http_get_json, runs_root=None, out=None):
    """Pull one city's inventory into out_dir_for(city) (see the module docstring)."""
    entry = INVENTORIES.get(city)
    dest = out_dir_for(city, out)
    record_path = dest / 'inventory.json'
    if entry is None:
        raise SystemExit(f'{city}: no inventory registered'
                         + (' (checked, none exists; see NO_INVENTORY)' if city in NO_INVENTORY
                            else ''))
    dest.mkdir(parents=True, exist_ok=True)
    if entry['status'] != 'ok':
        record = {'city': city, 'status': entry['status'], 'url': entry['url'],
                  'publisher': entry.get('publisher'),
                  'note': 'not fetched: the service is not answering; no scoring possible'}
        record_path.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8',
                               newline='\n')
        print(f'{city}: {entry["status"]} -- recorded in {record_path}, nothing fetched')
        return record
    geo_path = dest / 'inventory.geojson'
    run_dir = Path(runs_root or REPO_ROOT / 'runs') / city
    area = load_area(run_dir)
    if geo_path.exists() and record_path.exists() and not refresh:
        record = json.loads(record_path.read_text(encoding='utf-8'))
        check_cache(geo_path, record)
        bbox = [round(v, 7) for v in area.bounds]
        if bbox != (record.get('query') or {}).get('bbox_wgs84'):
            raise SystemExit(f'{city}: the cached snapshot was pulled for bbox '
                             f'{(record.get("query") or {}).get("bbox_wgs84")}, but '
                             f'{run_dir / "area.geojson"} now spans {bbox} -- the area '
                             'changed; --refresh re-pulls it')
        print(f'{city}: reusing cached {geo_path.name} fetched {record["fetched_utc"]} '
              '(the frozen snapshot; --refresh re-pulls it)', file=sys.stderr)
        return record
    feats, info = fetch_features(entry['url'], area.bounds, get=get)
    inside = in_area(feats, area)
    keep = kept(inside, entry)
    body = json.dumps({'type': 'FeatureCollection', 'features': inside},
                      separators=(',', ':')).encode('utf-8')
    tmp = geo_path.with_name(geo_path.name + '.part')
    tmp.write_bytes(body)
    tmp.replace(geo_path)
    record = {
        'city': city, 'status': 'ok', 'url': entry['url'], 'portal': entry.get('portal'),
        'item': entry.get('item'), 'publisher': entry.get('publisher'),
        'terms': entry.get('terms'),
        'query': {'where': '1=1', 'geometry': 'envelope of area.geojson bounds',
                  'bbox_wgs84': [round(v, 7) for v in area.bounds], 'outSR': 4326,
                  'f': 'geojson', 'paging': 'resultOffset'},
        'keep': {'field': entry['keep_field'], 'values': list(entry['keep_values']),
                 'where_equivalent': f"{entry['keep_field']} IN "
                                     f"({', '.join(repr(v) for v in entry['keep_values'])})"},
        'fetched_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'file': geo_path.name, 'sha256': sha256_bytes(body), 'bytes': len(body),
        **info, 'in_area': len(inside), 'kept': len(keep),
        'histograms_in_area': histograms(inside, entry['fields']),
    }
    record_path.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8',
                           newline='\n')
    print(f'{city}: layer {info["layer_total"]} / bbox {info["in_bbox"]} / in area '
          f'{len(inside)} / kept {len(keep)} ({record["keep"]["where_equivalent"]}); '
          f'{info["pages"]} page(s); sha256 {record["sha256"][:12]}')
    return record


def check_cache(geo_path, record):
    """Refuse a cached geojson that is not the one inventory.json describes."""
    body = Path(geo_path).read_bytes()
    if sha256_bytes(body) != record.get('sha256'):
        raise SystemExit(f'{geo_path}: sha256 does not match inventory.json -- the cache was '
                         'edited or replaced; --refresh re-pulls it')


def load_inventory(city, out=None):
    """(kept point list [(index, lat, lng, props)], record) from the cached snapshot."""
    entry = INVENTORIES.get(city)
    if entry is None or entry['status'] != 'ok':
        raise SystemExit(f'{city}: no usable inventory registered')
    dest = out_dir_for(city, out)
    geo_path, record_path = dest / 'inventory.geojson', dest / 'inventory.json'
    if not geo_path.exists() or not record_path.exists():
        # fall back to the committed location when --out redirects outputs only
        dest = out_dir_for(city)
        geo_path, record_path = dest / 'inventory.geojson', dest / 'inventory.json'
    if not geo_path.exists() or not record_path.exists():
        raise SystemExit(f'{city}: no cached inventory -- run `inventory_oracle.py fetch {city}`')
    record = json.loads(record_path.read_text(encoding='utf-8'))
    check_cache(geo_path, record)
    feats = json.loads(geo_path.read_text(encoding='utf-8'))['features']
    pts = [(k, f['geometry']['coordinates'][1], f['geometry']['coordinates'][0],
            f.get('properties') or {}) for k, f in enumerate(kept(feats, entry))]
    return pts, record


# ------------------------------------------------------------------------------ arms

def _year(pano):
    return (pano.capture_date or '')[:4] or None


def vintage_medians(panos):
    """{capture year: median measured depth height} over the run's measured panos. Any
    number of measured panos makes a median (height_qc.load_city's `vmed`, which is how
    the decision table in docs/camera-height-study.md classified the low rig)."""
    by_year = {}
    for p in panos:
        y = _year(p)
        if p.camera_height_m is not None and y:
            by_year.setdefault(y, []).append(p.camera_height_m)
    return {y: fs._median(sorted(v)) for y, v in by_year.items()}


def per_rig_panos(panos, cut=OPTION_C[0], low=OPTION_C[1], high=OPTION_C[2]):
    """Option (c): copies of the panos with camera_height_m set per vintage.

    Every dated pano of a vintage whose median measured depth height is < cut gets `low`,
    every other dated pano of a measured vintage gets `high` -- measured or not, since the
    rig, not the payload, sets the height. Undated panos and vintages with no measured
    pano get None, i.e. geo.camera_height_for's 2.6 m fallback. The spread is cleared (the
    error model's flat sigma applies) and height_group names the vintage, so
    fs.camera_height_counts reports the assignment. Returns (panos, {year: (median, h)}).
    """
    med = vintage_medians(panos)
    rig = {y: (m, low if m < cut else high) for y, m in med.items()}
    out = []
    for p in panos:
        y = _year(p)
        h = rig[y][1] if y in rig else None
        out.append(replace(p, camera_height_m=h, camera_height_spread_m=None,
                           camera_height_vintage_m=None,
                           height_group=f'{y}:{h:g}' if h is not None else None))
    return out, rig


def scaled_panos(panos, scale):
    """Option (b): copies with every measured depth height (and its vintage median, so the
    #44 QC flag reads the same) multiplied by `scale`; unmeasured panos unchanged. The
    spread is left as measured (its sigma is floored at the error model's anyway)."""
    return [p if p.camera_height_m is None else
            replace(p, camera_height_m=p.camera_height_m * scale,
                    camera_height_vintage_m=None if p.camera_height_vintage_m is None
                    else p.camera_height_vintage_m * scale)
            for p in panos]


def build_arms(panos, base, scales=(), rig_cuts=()):
    """{arm: (panos, FuseParams)} -- the three decision arms, the x1.00 reference, and any
    sensitivity arms (`b<scale>`, `c<cut>`), plus the per-rig assignment for the report."""
    rig_panos, rig = per_rig_panos(panos)
    arms = {ARM_A: (panos, replace(base, camera_height_m=geo.DEFAULT_CAMERA_HEIGHT_M)),
            ARM_B: (scaled_panos(panos, OPTION_B_SCALE),
                    replace(base, camera_height_m=geo.PER_PANO)),
            ARM_C: (rig_panos, replace(base, camera_height_m=geo.PER_RIG)),
            ARM_B_REF: (panos, replace(base, camera_height_m=geo.PER_PANO))}
    for s in scales:
        arms[f'b{s:.2f}'] = (scaled_panos(panos, s), replace(base, camera_height_m=geo.PER_PANO))
    for cut in rig_cuts:
        arms[f'c{cut:.2f}'] = (per_rig_panos(panos, cut=cut)[0],
                               replace(base, camera_height_m=geo.PER_RIG))
    return arms, rig


# ---------------------------------------------------------------------------- scoring

def placer(arms):
    """place(arm, pano_id, x, y) for eval_sites.refit_frozen, with poses cached per arm."""
    lookup = {arm: {p.pano_id: p for p in ps} for arm, (ps, _) in arms.items()}
    poses = {arm: {} for arm in arms}

    def place(arm, pano_id, x, y):
        params = arms[arm][1]
        pano = lookup[arm][pano_id]
        pose = poses[arm].get(pano_id)
        if pose is None:
            pose = poses[arm][pano_id] = fs.pano_pose(pano, params.apply_pose)
        return geo.detection_ground_point(
            pose, x, y, camera_height=params.camera_height_m,
            max_range_m=params.max_range_m, errors=geo.error_model_for(pano.source),
            apply_pose=params.rotates)
    return place


def along_ray(site_e, site_n, pt_e, pt_n, members):
    """(signed offset, mean member range) of a site from its reference point along the
    mean direction of its member rays. + = the site lies beyond the point as seen from the
    cameras, i.e. ranges run long. `members` are GroundEstimates (bearing_deg, range_m).
    None when the member rays cancel (a site seen from exactly opposite sides)."""
    ue = sum(math.sin(math.radians(g.bearing_deg)) for g in members)
    un = sum(math.cos(math.radians(g.bearing_deg)) for g in members)
    norm = math.hypot(ue, un)
    if norm < 1e-9:
        return None
    along = ((site_e - pt_e) * ue + (site_n - pt_n) * un) / norm
    return along, sum(g.range_m for g in members) / len(members)


def site_vintage(site, by_id):
    """The capture year shared by every operational member's pano, 'mixed', or 'undated'."""
    years = {(by_id[d.pano_id].capture_date or '')[:4] for d, _ in site.members
             if d.operational}
    if len(years) > 1:
        return 'mixed'
    y = years.pop() if years else ''
    return y or 'undated'


def _stats(dists):
    return {'median_m': es._pct(dists, 0.5), 'p90_m': es._pct(dists, 0.9),
            'mean_m': sum(dists) / len(dists) if dists else None,
            f'share_le_{WITHIN_M:g}m': (sum(1 for d in dists if d <= WITHIN_M) / len(dists)
                                        if dists else None)}


def score_frozen(anchor, fused, points, arm_names, arms, by_id, radii, sigma_scale,
                 pool_arm=ARM_A):
    """Rows and vintage rows for one frozen frame (membership from `anchor`'s own fuse).

    fused[anchor] = (sites, frame); points are Pt in that frame. The pool at each radius
    is anchored on `pool_arm`'s re-placed positions within this membership: arm (a) for
    every frame, as pre-registered. The one-to-one match truncates the pool arm's own
    errors at the radius and nobody else's, so an (a)-anchored pool is conservative
    against both candidates; `score --pool-anchor frame` (pool_arm = anchor) is the
    sensitivity check on that, never the verdict."""
    sites, frame = fused[anchor]
    op_sites = [s for s in sites if s.n_operational > 0]
    kept_sites, site_pos, members = es.refit_frozen(op_sites, frame, placer(arms), arm_names,
                                                    sigma_scale)
    index = {s.id: k for k, s in enumerate(kept_sites)}
    frame_name = FROZEN + anchor
    rows, vrows = [], []
    for radius in radii:
        pool = match_one_to_one(points, site_pos[pool_arm], radius)  # {pt index: _Placed}
        for arm in arm_names:
            pos = site_pos[arm]
            dists = [math.hypot(points[i].e - pos[index[s.id]].e,
                                points[i].n - pos[index[s.id]].n) for i, s in pool.items()]
            own = match_one_to_one(points, pos, radius)
            own_d = [math.hypot(points[i].e - s.e, points[i].n - s.n) for i, s in own.items()]
            chance = chance_floor(points, pos, radius)
            row = {'frame': frame_name, 'arm': arm, 'radius_m': radius,
                   'n_points': len(points), 'op_sites': len(op_sites),
                   'sites_scored': len(kept_sites),
                   'sites_dropped': len(op_sites) - len(kept_sites),
                   'n_pool': len(pool), **_stats(dists),
                   'own_matched': len(own),
                   'coverage': len(own) / len(points) if points else None,
                   **{'own_' + k: v for k, v in _stats(own_d).items()},
                   'chance_matched': chance,
                   'chance_share': chance / len(points) if points else None}
            rows.append(row)
            if radius != PRIMARY_RADIUS_M:
                continue
            by_vintage = {}
            for i, s in pool.items():
                k = index[s.id]
                got = along_ray(pos[k].e, pos[k].n, points[i].e, points[i].n, members[arm][k])
                if got is None:
                    continue
                by_vintage.setdefault(site_vintage(kept_sites[k], by_id), []).append(got)
            for vintage in sorted(by_vintage):
                vals = by_vintage[vintage]
                along = [a for a, _ in vals]
                vrows.append({'frame': frame_name, 'arm': arm, 'radius_m': radius,
                              'vintage': vintage, 'n': len(vals),
                              'median_along_m': es._pct(along, 0.5),
                              'q25_along_m': es._pct(along, 0.25),
                              'q75_along_m': es._pct(along, 0.75),
                              'median_along_over_range': es._pct(
                                  [a / r for a, r in vals if r > 0], 0.5),
                              'median_range_m': es._pct([r for _, r in vals], 0.5)})
    return rows, vrows


def score_own(arm, fused, points, radii):
    """Rows for `arm` fused under itself, matched on its own (the `own` frame)."""
    sites, _frame = fused[arm]
    ops = [Pt(s.id, s.e, s.n) for s in sites if s.n_operational > 0]
    rows = []
    for radius in radii:
        own = match_one_to_one(points, ops, radius)
        own_d = [math.hypot(points[i].e - s.e, points[i].n - s.n) for i, s in own.items()]
        chance = chance_floor(points, ops, radius)
        rows.append({'frame': OWN, 'arm': arm, 'radius_m': radius, 'n_points': len(points),
                     'op_sites': len(ops), 'own_matched': len(own),
                     'coverage': len(own) / len(points) if points else None,
                     **{'own_' + k: v for k, v in _stats(own_d).items()},
                     'chance_matched': chance,
                     'chance_share': chance / len(points) if points else None})
    return rows


def score_city(city, tier=OPERATIONAL_CONFIDENCE, radii=RADII_M, scales=(), rig_cuts=(),
               runs_root=None, out=None, limit=None, panos=None, inventory=None,
               pool_anchor=ARM_A):
    """Everything `score` computes for one city. Returns a dict (see write_outputs).

    pool_anchor: ARM_A (pre-registered: every frozen frame's pool on (a)) or 'frame'
    (frozen@X's pool on X -- the anchoring sensitivity check; not what verdict reads).

    `panos` / `inventory` ([(index, lat, lng, props)]) may be passed in (tests); by
    default they are read from runs/<city>/ and the cached snapshot."""
    t0 = time.time()
    run_dir = Path(runs_root or REPO_ROOT / 'runs') / city
    record = None
    if panos is None:
        if not (run_dir / 'depth' / 'index.csv').exists():
            raise SystemExit(f'{run_dir}/depth/index.csv is missing: (b) and (c) need the '
                             'harvested depth heights (scripts/harvest_depth.py); refusing '
                             'to score them all as 2.6 m')
        panos, _skipped = fs.load_results(run_dir / 'results.jsonl', read_heights=True)
        if limit:
            panos = panos[:limit]
    if inventory is None:
        inventory, record = load_inventory(city, out)
    base = fs.FuseParams(min_confidence=tier)
    arms, rig = build_arms(panos, base, scales, rig_cuts)
    arm_names = tuple(arms)
    fused, stats = {}, {}
    for arm, (ps, params) in arms.items():
        sites, frame, st = fs.fuse(ps, params)
        fused[arm], stats[arm] = (sites, frame), st
    frame = fused[ARM_A][1]
    # every arm's fuse sees the same pano positions, so the same LocalFrame
    assert all(abs(f.lat0 - frame.lat0) < 1e-12 and abs(f.lng0 - frame.lng0) < 1e-12
               for _s, f in fused.values())
    points = []
    for k, lat, lng, _props in inventory:
        e, n = frame.to_enu(lat, lng)
        points.append(Pt(k, e, n))
    by_id = {p.pano_id: p for p in panos}
    rows, vrows = [], []
    for anchor in DECISION_ARMS:
        r, v = score_frozen(anchor, fused, points, arm_names, arms, by_id, radii,
                            base.sigma_scale,
                            pool_arm=anchor if pool_anchor == POOL_FRAME else ARM_A)
        rows += r
        vrows += v
    for arm in arm_names:
        rows += score_own(arm, fused, points, radii)
    for r in rows + vrows:
        r['city'] = city
        r['tier'] = tier
    return {'city': city, 'tier': tier, 'rows': rows, 'vintage_rows': vrows,
            'pool_anchor': pool_anchor, 'record': record, 'rig': rig, 'n_panos': len(panos),
            'camera_heights': {a: stats[a]['camera_heights'] for a in arm_names},
            'n_op_sites': {a: stats[a]['n_operational_sites'] for a in arm_names},
            'inventory': inventory, 'fused_a': fused[ARM_A],
            'runtime_s': time.time() - t0, 'limit': limit}


# ---------------------------------------------------------------------- characterize

def nearest_site_table(inventory, sites, frame, field, cap_m=50.0):
    """[(value, n, median, p90, share <= 2 m, share <= 5 m)] of each kept inventory
    point's distance to its nearest operational site, by one categorical field."""
    grid = geo.GridIndex(cap_m)
    ops = [s for s in sites if s.n_operational > 0]
    for s in ops:
        grid.add(s.e, s.n, s)
    groups = {}
    for _k, lat, lng, props in inventory:
        e, n = frame.to_enu(lat, lng)
        best = cap_m
        for s in grid.near(e, n):
            best = min(best, math.hypot(e - s.e, n - s.n))
        groups.setdefault('(all)', []).append(best)
        if field:
            groups.setdefault(str(props.get(field)), []).append(best)
    out = []
    for value, ds in sorted(groups.items(), key=lambda kv: (kv[0] != '(all)', -len(kv[1]))):
        out.append((value, len(ds), es._pct(ds, 0.5), es._pct(ds, 0.9),
                    sum(1 for d in ds if d <= 2.0) / len(ds),
                    sum(1 for d in ds if d <= 5.0) / len(ds)))
    return out


def characterize_markdown(city, inventory, record, sites, frame):
    entry = INVENTORIES[city]
    lines = [f'## Inventory: {entry.get("publisher")}', '']
    if record:
        lines += [f'- layer `{record["url"]}` ({record.get("layer_name")}), item '
                  f'`{record.get("item")}`; portal {record.get("portal")}',
                  f'- fetched {record["fetched_utc"]}: layer total {record["layer_total"]}, '
                  f'in the area bbox {record["in_bbox"]}, in `area.geojson` '
                  f'{record["in_area"]}, kept {record["kept"]} '
                  f'(`{record["keep"]["where_equivalent"]}`)',
                  f'- snapshot `{record["file"]}` (untracked): sha256 `{record["sha256"]}`, '
                  f'{record["bytes"]} bytes',
                  f'- terms: {record.get("terms")}', '']
    lines += ['Kept point -> nearest operational site under (a) (in-memory fuse at the '
              'scoring tier, capped at 50 m), by field:', '']
    for field in entry['characterize_fields']:
        lines += [f'| {field} | n | median m | p90 m | <= 2 m | <= 5 m |',
                  '|---|---:|---:|---:|---:|---:|']
        for value, n, med, p90, s2, s5 in nearest_site_table(inventory, sites, frame, field):
            lines.append(f'| {value} | {n} | {med:.2f} | {p90:.2f} | {100 * s2:.0f}% | '
                         f'{100 * s5:.0f}% |')
        lines.append('')
    return '\n'.join(lines)


# ----------------------------------------------------------------------------- verdict

def _f(v):
    if v in (None, ''):
        return None
    return float(v)


def _row(rows, frame, arm, radius=PRIMARY_RADIUS_M):
    for r in rows:
        if r['frame'] == frame and r['arm'] == arm and abs(_f(r['radius_m']) - radius) < 1e-9:
            return r
    raise KeyError(f'no row frame={frame} arm={arm} radius={radius}')


def _vrow(vrows, frame, arm, vintage):
    for r in vrows:
        if r['frame'] == frame and r['arm'] == arm and r['vintage'] == vintage:
            return r
    return None


def verdict(rows_by_city, vintage_by_city):
    """The pre-registered #79 rule. rows_by_city / vintage_by_city: {city: [row dict]} as
    written to arms.csv / vintage.csv (strings or numbers). Returns
    (selected arm, {candidate: {rule: bool}}, reasons).

    Primary frame: frozen@a, 5 m, the (a)-anchored pool. A candidate X in {b, c} replaces
    2.6 m only if ALL hold:
      1. gainesville: X's p90 AND median improve on (a) by > 0.10 m.
      2. bend: X's p90 is not worse than (a)'s by > 0.10 m.
      3. gainesville, not by construction: X fused under itself (`own`) has an own-match
         p90 below (a)'s primary p90, AND in frozen@X (X's membership, pool still anchored
         on (a)'s positions) X's p90 is below (a)'s.
      4. survivorship, both cities: X's own-match coverage (primary frame) within 1.0 pt of
         (a)'s; sites dropped by the every-arm-places filter <= 5% of (a)'s operational
         sites -- else the candidate is a CAVEAT, never selected.
      5. direction: gainesville's 2026-vintage median along-ray offset under X (primary
         frame) within +-0.75 m of zero.
      6. tie-break: both pass -> (c), unless (b)'s p90 beats (c)'s by > 0.10 m in both
         cities. Neither passes -> (a): 2.6 m stays.
    """
    reasons, results = [], {}
    prim = FROZEN + ARM_A
    gv, bd = rows_by_city[DECIDING_CITY], rows_by_city[GUARD_CITY]
    for x in CANDIDATES:
        res = {}
        ga, gx = _row(gv, prim, ARM_A), _row(gv, prim, x)
        dp90 = _f(ga['p90_m']) - _f(gx['p90_m'])
        dmed = _f(ga['median_m']) - _f(gx['median_m'])
        res['1'] = dp90 > RULE_MARGIN_M and dmed > RULE_MARGIN_M
        reasons.append(f'[{x}] rule 1 ({DECIDING_CITY}): p90 {_f(ga["p90_m"]):.3f} -> '
                       f'{_f(gx["p90_m"]):.3f} (improves {dp90:+.3f}), median '
                       f'{_f(ga["median_m"]):.3f} -> {_f(gx["median_m"]):.3f} (improves '
                       f'{dmed:+.3f}); needs both > {RULE_MARGIN_M} -> '
                       f'{"pass" if res["1"] else "FAIL"}')
        ba, bx = _row(bd, prim, ARM_A), _row(bd, prim, x)
        worse = _f(bx['p90_m']) - _f(ba['p90_m'])
        res['2'] = worse <= RULE_MARGIN_M
        reasons.append(f'[{x}] rule 2 ({GUARD_CITY}): p90 {_f(ba["p90_m"]):.3f} -> '
                       f'{_f(bx["p90_m"]):.3f} ({worse:+.3f}); may not worsen by > '
                       f'{RULE_MARGIN_M} -> {"pass" if res["2"] else "FAIL"}')
        own_x = _f(_row(gv, OWN, x)['own_p90_m'])
        fx = _row(gv, FROZEN + x, x)
        fa = _row(gv, FROZEN + x, ARM_A)
        c3a = own_x < _f(ga['p90_m'])
        c3b = _f(fx['p90_m']) < _f(fa['p90_m'])
        res['3'] = c3a and c3b
        reasons.append(f'[{x}] rule 3 ({DECIDING_CITY}): own-association p90 {own_x:.3f} vs '
                       f'(a) primary {_f(ga["p90_m"]):.3f} -> {"ok" if c3a else "no"}; under '
                       f'{x}\'s membership p90 {x} {_f(fx["p90_m"]):.3f} vs a '
                       f'{_f(fa["p90_m"]):.3f} -> {"ok" if c3b else "no"} -> '
                       f'{"pass" if res["3"] else "FAIL"}')
        cov_ok, drop_ok, bits = True, True, []
        for city in (DECIDING_CITY, GUARD_CITY):
            rows = rows_by_city[city]
            ra, rx = _row(rows, prim, ARM_A), _row(rows, prim, x)
            dcov = _f(rx['coverage']) - _f(ra['coverage'])
            drop = _f(ra['sites_dropped']) / _f(ra['op_sites']) if _f(ra['op_sites']) else 0.0
            cov_ok &= abs(dcov) <= RULE_MAX_COVERAGE_DROP + 1e-12
            drop_ok &= drop <= RULE_MAX_SITE_DROP
            bits.append(f'{city} coverage {x}-a {100 * dcov:+.2f} pt, sites dropped '
                        f'{100 * drop:.2f}%')
        res['4'] = cov_ok and drop_ok
        res['caveat'] = not drop_ok
        reasons.append(f'[{x}] rule 4: ' + '; '.join(bits)
                       + f' (coverage within {100 * RULE_MAX_COVERAGE_DROP:.1f} pt, drop <= '
                       f'{100 * RULE_MAX_SITE_DROP:.0f}%) -> '
                       + ('pass' if res['4'] else 'CAVEAT (site drop)' if not drop_ok
                          else 'FAIL'))
        v = _vrow(vintage_by_city[DECIDING_CITY], prim, x, DECIDING_VINTAGE)
        along = None if v is None else _f(v['median_along_m'])
        res['5'] = along is not None and abs(along) <= RULE_DIRECTION_M
        reasons.append(f'[{x}] rule 5 ({DECIDING_CITY} {DECIDING_VINTAGE}): median along-ray '
                       f'offset {"n/a" if along is None else f"{along:+.3f}"} m, needs within '
                       f'+-{RULE_DIRECTION_M} -> {"pass" if res["5"] else "FAIL"}')
        res['pass'] = all(res[k] for k in '12345')
        results[x] = res
        reasons.append(f'[{x}] -> {"PASSES" if res["pass"] else "does not pass"}')
    passing = [x for x in CANDIDATES if results[x]['pass']]
    if not passing:
        selected = ARM_A
        reasons.append('rule 6: neither candidate passes -> (a) 2.6 m stays')
    elif len(passing) == 1:
        selected = passing[0]
        reasons.append(f'rule 6: only {selected} passes -> {arm_label(selected)}')
    else:
        b_better = all(_f(_row(rows_by_city[c], prim, ARM_C)['p90_m'])
                       - _f(_row(rows_by_city[c], prim, ARM_B)['p90_m']) > RULE_MARGIN_M
                       for c in (DECIDING_CITY, GUARD_CITY))
        selected = ARM_B if b_better else ARM_C
        reasons.append(f'rule 6: both pass; (b) beats (c) on p90 by > {RULE_MARGIN_M} m in '
                       f'both cities: {b_better} -> {arm_label(selected)}')
    reasons.append(f'#79 VERDICT: {arm_label(selected)}'
                   + ('' if selected == ARM_A else ' -- a follow-up PR flips the default'))
    return selected, results, reasons


# ----------------------------------------------------------------------------- output

ARM_COLUMNS = ('city', 'tier', 'frame', 'arm', 'radius_m', 'n_points', 'op_sites',
               'sites_scored', 'sites_dropped', 'n_pool', 'median_m', 'p90_m', 'mean_m',
               f'share_le_{WITHIN_M:g}m', 'own_matched', 'coverage', 'own_median_m',
               'own_p90_m', 'own_mean_m', f'own_share_le_{WITHIN_M:g}m', 'chance_matched',
               'chance_share')
VINTAGE_COLUMNS = ('city', 'tier', 'frame', 'arm', 'radius_m', 'vintage', 'n',
                   'median_along_m', 'q25_along_m', 'q75_along_m', 'median_along_over_range',
                   'median_range_m')


def _fmt(v, nd=4):
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in columns])


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def report_markdown(res):
    city, rows, vrows = res['city'], res['rows'], res['vintage_rows']
    m = lambda v, nd=2: '—' if v is None else f'{v:.{nd}f}'  # noqa: E731
    pct = lambda v: '—' if v is None else f'{100 * v:.1f}%'  # noqa: E731
    lines = [f'# Placement oracle: {city}', '',
             f'Issue #79. Tier {res["tier"]}, rig mask on, GSV flat raycast, 25 m cap; '
             f'{res["n_panos"]} panos'
             + (f' (SMOKE: first {res["limit"]} only)' if res['limit'] else '')
             + f'. Runtime {res["runtime_s"]:.0f} s. Generated by '
               '`scripts/inventory_oracle.py score`; the rule is in docs/placement-oracle.md.',
             '']
    lines.append(characterize_markdown(city, res['inventory'], res['record'],
                                       *res['fused_a']))
    lines += ['## Arms', '', '| arm | camera heights (fuse_sites.camera_height_counts) | '
              'operational sites (own fuse) |', '|---|---|---:|']
    for arm, ch in res['camera_heights'].items():
        lines.append(f'| {arm_label(arm)} | `{json.dumps(ch, sort_keys=True)}` | '
                     f'{res["n_op_sites"][arm]} |')
    lines += ['', 'Per-rig assignment (vintage: median measured depth height -> height):', '']
    lines += ['| vintage | median depth m | (c) height m |', '|---|---:|---:|']
    for y, (med, h) in sorted(res['rig'].items()):
        lines.append(f'| {y} | {med:.3f} | {h:.1f} |')
    for radius in sorted({r['radius_m'] for r in rows}):
        anchoring = ('(a)-anchored pool' if res.get('pool_anchor', ARM_A) == ARM_A else
                     "pool anchored on each frame's own arm (SENSITIVITY, not the verdict)")
        lines += ['', f'## Frozen association, {radius:g} m, {anchoring}', '',
                  '| frame | arm | pool | median m | p90 m | <= 3 m | own coverage | '
                  'own <= 3 m | chance | sites scored / dropped |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in rows:
            if r['frame'] == OWN or r['radius_m'] != radius:
                continue
            lines.append(f'| {r["frame"]} | {arm_label(r["arm"])} | {r["n_pool"]} | '
                         f'{m(r["median_m"])} | {m(r["p90_m"])} | {pct(r["share_le_3m"])} | '
                         f'{pct(r["coverage"])} | {pct(r["own_share_le_3m"])} | '
                         f'{pct(r["chance_share"])} | {r["sites_scored"]} / '
                         f'{r["sites_dropped"]} |')
    lines += ['', '## Own association (each arm fused under itself; favours itself)', '',
              '| arm | radius m | operational sites | coverage | own median m | own p90 m | '
              'own <= 3 m | chance |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['frame'] == OWN:
            lines.append(f'| {arm_label(r["arm"])} | {r["radius_m"]:g} | {r["op_sites"]} | '
                         f'{pct(r["coverage"])} | {m(r["own_median_m"])} | '
                         f'{m(r["own_p90_m"])} | {pct(r["own_share_le_3m"])} | '
                         f'{pct(r["chance_share"])} |')
    lines += ['', '## Along-ray offset by capture year (single-vintage pool sites, 5 m)', '',
              '+ = the site lies beyond the inventory point as seen from its cameras '
              '(ranges run long).', '',
              '| frame | arm | vintage | n | median m | IQR m | median along/range | '
              'median range m |', '|---|---|---|---:|---:|---|---:|---:|']
    for v in vrows:
        lines.append(f'| {v["frame"]} | {arm_label(v["arm"])} | {v["vintage"]} | {v["n"]} | '
                     f'{m(v["median_along_m"])} | {m(v["q25_along_m"])} … '
                     f'{m(v["q75_along_m"])} | {m(v["median_along_over_range"], 3)} | '
                     f'{m(v["median_range_m"], 1)} |')
    return '\n'.join(lines) + '\n'


def write_outputs(res, out=None):
    dest = out_dir_for(res['city'], out)
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / 'arms.csv', res['rows'], ARM_COLUMNS)
    write_csv(dest / 'vintage.csv', res['vintage_rows'], VINTAGE_COLUMNS)
    (dest / 'report.md').write_text(report_markdown(res), encoding='utf-8', newline='\n')
    return dest


# -------------------------------------------------------------------------------- CLI

def cmd_fetch(args):
    for city in args.cities:
        fetch_city(city, refresh=args.refresh, out=args.out)


def cmd_characterize(args):
    for city in args.cities:
        run_dir = REPO_ROOT / 'runs' / city
        inventory, record = load_inventory(city, args.out)
        panos, _ = fs.load_results(run_dir / 'results.jsonl', read_heights=False)
        if args.limit:
            panos = panos[:args.limit]
        sites, frame, _ = fs.fuse(panos, fs.FuseParams(min_confidence=args.tier))
        print(characterize_markdown(city, inventory, record, sites, frame))


def cmd_score(args):
    for city in args.cities:
        res = score_city(city, tier=args.tier, radii=tuple(args.radius),
                         scales=tuple(args.scale), rig_cuts=tuple(args.rig_cut),
                         out=args.out, limit=args.limit, pool_anchor=args.pool_anchor)
        dest = write_outputs(res, args.out)
        print(f'{city}: {res["n_panos"]} panos, {len(res["inventory"])} kept points, '
              f'{res["runtime_s"]:.0f} s -> {dest}')


def cmd_verdict(args):
    rows, vrows = {}, {}
    for city in (DECIDING_CITY, GUARD_CITY):
        dest = out_dir_for(city, args.out)
        rows[city] = read_csv(dest / 'arms.csv')
        vrows[city] = read_csv(dest / 'vintage.csv')
        tiers = {r['tier'] for r in rows[city]}
        if tiers != {_fmt(OPERATIONAL_CONFIDENCE)}:
            print(f'WARNING: {city} arms.csv is at tier {tiers}, not the pre-registered '
                  f'{OPERATIONAL_CONFIDENCE} -- this is not the verdict', file=sys.stderr)
    _selected, _results, reasons = verdict(rows, vrows)
    print('\n'.join(reasons))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('cities', nargs='+')
        p.add_argument('--out', default=None,
                       help='write under <out>/<city>/ instead of runs/<city>/inventory_oracle')
    p = sub.add_parser('fetch', help='pull and cache the inventories (network)')
    common(p)
    p.add_argument('--refresh', action='store_true', help='re-pull a cached snapshot')
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser('characterize', help='offline: inventory vs nearest site under (a)')
    common(p)
    p.add_argument('--tier', type=float, default=OPERATIONAL_CONFIDENCE)
    p.add_argument('--limit', type=int, default=None, help='first N panos (smoke)')
    p.set_defaults(fn=cmd_characterize)
    p = sub.add_parser('score', help='offline: a/b/c, frozen + own association')
    common(p)
    p.add_argument('--tier', type=float, default=OPERATIONAL_CONFIDENCE)
    p.add_argument('--radius', type=float, nargs='+', default=list(RADII_M))
    p.add_argument('--scale', type=float, nargs='*', default=[],
                   help='extra per-pano scale arms (sensitivity, not a verdict)')
    p.add_argument('--rig-cut', type=float, nargs='*', default=[],
                   help='extra per-rig threshold arms (sensitivity, not a verdict)')
    p.add_argument('--pool-anchor', choices=(ARM_A, POOL_FRAME), default=ARM_A,
                   help="a (pre-registered): every frozen frame's pool on (a); frame: "
                        "frozen@X's pool on X -- sensitivity only, needs --out")
    p.add_argument('--limit', type=int, default=None, help='first N panos (smoke)')
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser('verdict', help='the pre-registered rule over arms.csv/vintage.csv '
                       f'(always {DECIDING_CITY} + {GUARD_CITY}: the rule names its cities)')
    p.add_argument('--out', default=None,
                   help='read <out>/<city>/ instead of runs/<city>/inventory_oracle')
    p.set_defaults(fn=cmd_verdict)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == 'score' and PRIMARY_RADIUS_M not in args.radius:
        raise SystemExit(f'--radius must include the primary {PRIMARY_RADIUS_M:g} m')
    if args.cmd == 'score' and args.pool_anchor != ARM_A and not args.out:
        raise SystemExit('--pool-anchor frame is a sensitivity check: pass --out so it never '
                         'overwrites the committed arms.csv that verdict reads')
    args.fn(args)


if __name__ == '__main__':
    main()
