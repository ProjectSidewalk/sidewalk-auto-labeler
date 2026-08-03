"""Stage-2 multi-view association (issue #27): group per-pano detections into
physical curb-ramp sites.

Reads a run's results.jsonl, projects every stored detection to a flat-ground
world point (geo.detection_ground_point with anisotropic error; camera
pitch/roll deliberately NOT applied — the --pose-ablation experiment showed
streetlevel's GSV equirects are already gravity-rectified), and greedily
associates them into sites:

- processed in descending confidence, so every operational (>= --min-confidence)
  detection is placed before any sub-threshold one;
- candidate sites come from a spatial grid within --max-match-m (the GSV link
  graph is too incomplete to generate candidates — see #27's measurements);
- a same-pano cannot-link: two peaks in one heatmap are two physical objects
  (peak_local_max already NMS-separated them), so a detection never joins a site
  that already contains its own pano;
- a chi-square gate on the Mahalanobis distance under the combined covariance,
  then an inverse-covariance weighted position refit (the GLS triangulation of
  the ground-plane estimates — it can never extrapolate outside its members, so
  near-parallel rays degrade gracefully);
- merges that would push the site's triangulation residual per dof past
  --residual-per-dof-max are rejected (the detection seeds a new site instead);
- two-tier membership: sub-threshold detections attach to operational sites as
  in_refit=false "support" (they never move a submission-quality position) but
  refit among themselves in wholly-sub-threshold sites, so stage 4 can measure
  multi-view support below the operational threshold.

No vintage gate (cross-vintage co-detection is confirmation, per #27's measured
capture-delta data); capture dates are recorded per member and the eval's
ablation can re-fuse with FuseParams.max_vintage_months set.

Output: sites.jsonl (one site per line, fused position + covariance + members)
and sites_meta.json (parameters, frame origin, drop counters) beside the input.
Deliberately reads NO manifest.json (cluster-pulled runs lack one) and leaves
send_to_ps.py untouched — what to submit per site is a late decision (#27).

Usage:
    python scripts/fuse_sites.py runs/paterson
    python scripts/fuse_sites.py runs/paterson --pose-ablation   # lock pitch/roll signs
"""
import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geo  # noqa: E402
from detectors import DETECTION_STORAGE_FLOOR, OPERATIONAL_CONFIDENCE  # noqa: E402


@dataclass(frozen=True)
class FuseParams:
    floor: float = DETECTION_STORAGE_FLOOR
    min_confidence: float = OPERATIONAL_CONFIDENCE
    max_range_m: float = geo.DEFAULT_MAX_RANGE_M
    gate_chi2: float = 9.21          # chi-square(2 dof) 99th pct; loose on purpose —
                                     # the covariance model errs small and false splits
                                     # are cheap, while false merges are braked by the
                                     # cannot-link, the hard cap, and residual rejection
    max_match_m: float = 8.0         # above dual-ramp scatter, below corner spacing
    residual_per_dof_max: float = 3.0
    camera_height_m: float = geo.DEFAULT_CAMERA_HEIGHT_M
    apply_pose: bool = False         # GSV equirects are gravity-rectified; applying
                                     # metadata pitch/roll loosens multi-view
                                     # agreement (see geo._world_ray for the numbers)
    sigma_scale: float = 1.0         # inflate all covariances by scale^2 (model tuning)
    max_vintage_months: int | None = None  # eval-ablation only; None = no gate


@dataclass
class SlimPano:
    """One results.jsonl record reduced to what association needs."""
    pano_id: str
    lat: float
    lng: float
    camera_heading: float
    camera_pitch: float | None
    camera_roll: float | None
    capture_date: str | None
    source: str
    detections: list  # [(det_index, x_normalized, y_normalized, confidence)] as stored


@dataclass
class Det:
    """A projected detection, ready to associate."""
    pano_id: str
    det_index: int
    x: float
    y: float
    conf: float
    operational: bool
    e: float
    n: float
    cov: tuple            # sym2 ENU covariance, sigma_scale applied
    ground: geo.GroundEstimate
    months: int | None    # capture date as months-since-year-0
    capture_date: str | None
    source: str


class Site:
    """A physical-ramp hypothesis: information-filter accumulators over its refit
    members plus the full member list. Position/covariance/residual are exact
    closed forms of the accumulators (chi2_total = S - eta^T Lambda^-1 eta is the
    GLS optimum's residual, so no per-member loop is ever needed)."""

    __slots__ = ('id', 'members', 'pano_ids', 'lam', 'eta_e', 'eta_n', 'S',
                 'n_refit', 'n_operational', 'e', 'n', 'cov_p', 'chi2_total',
                 'month_min', 'month_max')

    def __init__(self, site_id, det):
        self.id = site_id
        self.members = []
        self.pano_ids = set()
        self.lam = (0.0, 0.0, 0.0)
        self.eta_e = self.eta_n = self.S = 0.0
        self.n_refit = 0
        self.n_operational = 0
        self.month_min = self.month_max = None
        self._absorb(det)
        self._add_member(det, in_refit=True)

    def _absorb(self, det):
        w = geo.sym2_inv(det.cov)
        self.lam = geo.sym2_add(self.lam, w)
        self.eta_e += w[0] * det.e + w[1] * det.n
        self.eta_n += w[1] * det.e + w[2] * det.n
        self.S += geo.sym2_quadform(w, det.e, det.n)
        self.n_refit += 1
        self.cov_p = geo.sym2_inv(self.lam)
        self.e = self.cov_p[0] * self.eta_e + self.cov_p[1] * self.eta_n
        self.n = self.cov_p[1] * self.eta_e + self.cov_p[2] * self.eta_n
        self.chi2_total = max(0.0, self.S - (self.eta_e * self.e + self.eta_n * self.n))

    def _add_member(self, det, in_refit):
        self.members.append((det, in_refit))
        self.pano_ids.add(det.pano_id)
        if det.operational:
            self.n_operational += 1
        if det.months is not None:
            self.month_min = det.months if self.month_min is None \
                else min(self.month_min, det.months)
            self.month_max = det.months if self.month_max is None \
                else max(self.month_max, det.months)

    def residual_per_dof(self):
        return self.chi2_total / (2 * self.n_refit - 2) if self.n_refit >= 2 else None

    def tentative_residual_per_dof(self, det):
        """residual_per_dof after absorbing det, computed on locals (no rollback)."""
        w = geo.sym2_inv(det.cov)
        lam = geo.sym2_add(self.lam, w)
        eta_e = self.eta_e + w[0] * det.e + w[1] * det.n
        eta_n = self.eta_n + w[1] * det.e + w[2] * det.n
        S = self.S + geo.sym2_quadform(w, det.e, det.n)
        inv = geo.sym2_inv(lam)
        pe = inv[0] * eta_e + inv[1] * eta_n
        pn = inv[1] * eta_e + inv[2] * eta_n
        chi2 = max(0.0, S - (eta_e * pe + eta_n * pn))
        return chi2 / (2 * (self.n_refit + 1) - 2)

    def vintage_ok(self, det, window_months):
        if window_months is None or det.months is None or self.month_min is None:
            return True
        return (max(self.month_max, det.months)
                - min(self.month_min, det.months)) <= window_months


def _months(capture_date):
    """'YYYY-MM...' -> months since year 0, or None."""
    try:
        y, m = str(capture_date)[:7].split('-')
        return int(y) * 12 + int(m) - 1
    except (ValueError, AttributeError):
        return None


def load_results(path):
    """Stream results.jsonl into SlimPanos, discarding links/history/metadata.
    Records without a position or heading can't be raycast and are dropped
    (counted by the caller via the skipped list)."""
    panos, skipped = [], 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec['pano']
            if p.get('lat') is None or p.get('lng') is None \
                    or p.get('camera_heading') is None:
                skipped += 1
                continue
            panos.append(SlimPano(
                pano_id=p['panorama_id'], lat=p['lat'], lng=p['lng'],
                camera_heading=p['camera_heading'],
                camera_pitch=p.get('camera_pitch'), camera_roll=p.get('camera_roll'),
                capture_date=p.get('capture_date'), source=p.get('source') or '',
                detections=[(i, d['x_normalized'], d['y_normalized'], d['confidence'])
                            for i, d in enumerate(rec.get('detections', []))]))
    return panos, skipped


def project(panos, params):
    """Raycast every stored detection >= floor. Returns (dets, frame, drops)."""
    frame = geo.LocalFrame(sum(p.lat for p in panos) / len(panos),
                           sum(p.lng for p in panos) / len(panos))
    drops = {'below_floor': 0, 'horizon': 0, 'out_of_range': 0}
    dets = []
    s2 = params.sigma_scale ** 2
    for p in panos:
        pose = geo.pano_pose({'lat': p.lat, 'lng': p.lng,
                              'camera_heading': p.camera_heading,
                              'camera_pitch': p.camera_pitch,
                              'camera_roll': p.camera_roll, 'source': p.source})
        errors = geo.error_model_for(p.source)
        months = _months(p.capture_date)
        for i, x, y, conf in p.detections:
            if conf < params.floor:
                drops['below_floor'] += 1
                continue
            g = geo.detection_ground_point(
                pose, x, y, camera_height=params.camera_height_m,
                max_range_m=params.max_range_m, errors=errors,
                apply_pose=params.apply_pose)
            if g is None:
                unbounded = geo.detection_ground_point(
                    pose, x, y, camera_height=params.camera_height_m,
                    max_range_m=math.inf, errors=errors,
                    apply_pose=params.apply_pose)
                drops['horizon' if unbounded is None else 'out_of_range'] += 1
                continue
            e, n = frame.to_enu(g.lat, g.lng)
            cov = g.cov_en(errors.sigma_gps_m)
            if s2 != 1.0:
                cov = (cov[0] * s2, cov[1] * s2, cov[2] * s2)
            dets.append(Det(p.pano_id, i, x, y, conf,
                            conf >= params.min_confidence, e, n, cov, g,
                            months, p.capture_date, p.source))
    return dets, frame, drops


def fuse(panos, params):
    """Associate projected detections into sites. Deterministic: input order never
    matters (canonical sort; best-candidate tiebreak on (D2, site id))."""
    dets, frame, drops = project(panos, params)
    dets.sort(key=lambda d: (-d.conf, d.pano_id, d.det_index))
    grid = geo.GridIndex(params.max_match_m)
    cap2 = params.max_match_m ** 2
    sites = []

    def new_site(det):
        site = Site(len(sites), det)
        sites.append(site)
        grid.add(site.e, site.n, site)
        return site

    for det in dets:
        best = None
        seen = set()
        for site in grid.near(det.e, det.n):
            if site.id in seen:
                continue
            seen.add(site.id)
            if det.pano_id in site.pano_ids:
                continue                       # same-pano cannot-link
            de, dn = det.e - site.e, det.n - site.n
            if de * de + dn * dn > cap2:
                continue                       # hard cap; also culls stale grid entries
            if not site.vintage_ok(det, params.max_vintage_months):
                continue
            d2 = geo.sym2_quadform(geo.sym2_inv(geo.sym2_add(det.cov, site.cov_p)),
                                   de, dn)
            if d2 <= params.gate_chi2 and (best is None or (d2, site.id) < best[:2]):
                best = (d2, site.id, site)

        if best is None:
            new_site(det)
            continue
        site = best[2]
        if det.operational or site.n_operational == 0:
            # refit merge — unless it would blow up the triangulation residual
            if site.n_refit >= 1 and \
                    site.tentative_residual_per_dof(det) > params.residual_per_dof_max:
                new_site(det)
                continue
            old_key = grid.key(site.e, site.n)
            site._absorb(det)
            site._add_member(det, in_refit=True)
            if grid.key(site.e, site.n) != old_key:
                grid.add(site.e, site.n, site)
        else:
            site._add_member(det, in_refit=False)   # support only; position untouched

    stats = {'n_panos': len(panos), 'n_projected': len(dets), 'drops': drops,
             'n_sites': len(sites),
             'n_operational_sites': sum(1 for s in sites if s.n_operational),
             'n_multi_pano_sites': sum(1 for s in sites if len(s.pano_ids) > 1),
             'frame_origin': {'lat0': frame.lat0, 'lng0': frame.lng0}}
    return sites, frame, stats


def site_to_json(site, frame):
    lat, lng = frame.to_latlng(site.e, site.n)
    ops_panos = {d.pano_id for d, _ in site.members if d.operational}
    rpd = site.residual_per_dof()
    span = None if site.month_min is None else site.month_max - site.month_min
    return {
        'site_id': site.id,
        'lat': round(lat, 7), 'lng': round(lng, 7),
        'cov_en_m2': [round(v, 4) for v in site.cov_p],
        'n_members': len(site.members),
        'n_panos': len(site.pano_ids),
        'n_operational': site.n_operational,
        'n_operational_panos': len(ops_panos),
        'best_confidence': round(max(d.conf for d, _ in site.members), 6),
        'mean_confidence': round(sum(d.conf for d, _ in site.members)
                                 / len(site.members), 6),
        'residual_per_dof': None if rpd is None else round(rpd, 4),
        'vintage_span_months': span,
        'members': [{
            'pano_id': d.pano_id, 'det_index': d.det_index,
            'x_normalized': d.x, 'y_normalized': d.y, 'confidence': d.conf,
            'lat': round(d.ground.lat, 7), 'lng': round(d.ground.lng, 7),
            'range_m': round(d.ground.range_m, 3),
            'bearing_deg': round(d.ground.bearing_deg, 3),
            'sigma_along_m': round(d.ground.sigma_along_m, 3),
            'sigma_cross_m': round(d.ground.sigma_cross_m, 3),
            'capture_date': d.capture_date, 'source': d.source,
            'in_refit': in_refit,
        } for d, in_refit in site.members],
    }


def write_sites(sites, frame, stats, params, out_path, meta_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for site in sites:
            f.write(json.dumps(site_to_json(site, frame)) + '\n')
    meta = {'params': asdict(params), **stats}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def _percentiles(values, points=(5, 25, 50, 75, 95)):
    if not values:
        return {}
    v = sorted(values)
    return {f'p{p}': round(v[min(len(v) - 1, int(len(v) * p / 100))], 2)
            for p in points}


def summarize(sites, stats):
    lines = [
        f"panos {stats['n_panos']}, projected {stats['n_projected']} detections "
        f"(dropped: {stats['drops']['below_floor']} below floor, "
        f"{stats['drops']['horizon']} at horizon, "
        f"{stats['drops']['out_of_range']} out of range)",
        f"sites {stats['n_sites']} ({stats['n_operational_sites']} operational, "
        f"{stats['n_multi_pano_sites']} multi-pano)",
    ]
    ranges = [d.ground.range_m for s in sites for d, _ in s.members]
    lines.append(f"member range_m percentiles: {_percentiles(ranges)}")
    rpds = [s.residual_per_dof() for s in sites if s.n_refit >= 2]
    if rpds:
        lines.append(f"residual_per_dof (multi-refit sites, n={len(rpds)}): "
                     f"{_percentiles(rpds)}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', help='run directory (with results.jsonl) or a jsonl path')
    ap.add_argument('--floor', type=float, default=DETECTION_STORAGE_FLOOR)
    ap.add_argument('--min-confidence', type=float, default=OPERATIONAL_CONFIDENCE)
    ap.add_argument('--max-range-m', type=float, default=geo.DEFAULT_MAX_RANGE_M)
    ap.add_argument('--gate-chi2', type=float, default=FuseParams.gate_chi2)
    ap.add_argument('--max-match-m', type=float, default=FuseParams.max_match_m)
    ap.add_argument('--residual-per-dof-max', type=float,
                    default=FuseParams.residual_per_dof_max)
    ap.add_argument('--camera-height-m', type=float,
                    default=geo.DEFAULT_CAMERA_HEIGHT_M)
    ap.add_argument('--sigma-scale', type=float, default=1.0)
    ap.add_argument('--apply-pose', action='store_true',
                    help='rotate rays by camera pitch/roll (measured to hurt on '
                         'GSV — see the --pose-ablation report)')
    ap.add_argument('--pose-ablation', action='store_true',
                    help='report within-site spread under each pitch/roll sign '
                         'convention instead of writing sites')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    src = Path(args.run)
    jsonl = src if src.is_file() else src / 'results.jsonl'
    if not jsonl.exists():
        sys.exit(f'no results.jsonl at {jsonl}')
    params = FuseParams(
        floor=args.floor, min_confidence=args.min_confidence,
        max_range_m=args.max_range_m, gate_chi2=args.gate_chi2,
        max_match_m=args.max_match_m,
        residual_per_dof_max=args.residual_per_dof_max,
        camera_height_m=args.camera_height_m, apply_pose=args.apply_pose,
        sigma_scale=args.sigma_scale)

    panos, skipped = load_results(jsonl)
    if skipped:
        print(f'skipped {skipped} records without position/heading')
    if not panos:
        sys.exit('no usable records')

    if args.pose_ablation:
        print(pose_ablation_report(panos, params))
        return

    sites, frame, stats = fuse(panos, params)
    out = args.out or jsonl.parent / 'sites.jsonl'
    meta = out.with_name(out.stem + '_meta.json')
    write_sites(sites, frame, stats, params, out, meta)
    print(summarize(sites, stats))
    print(f'wrote {out} and {meta}')


def pose_ablation_report(panos, params):
    """Empirically lock the pitch/roll sign convention (issue #27 stage 2).

    Association is frozen from a pose-OFF fuse; each member's ground point is
    then recomputed under every sign convention and the within-site pairwise
    member distance is compared. The correct convention tightens the cloud
    (a 2 deg pitch error moves a ground point ~1 m at 10 m, ~4 m at 20 m —
    far above the noise floor over thousands of multi-view sites); a wrong
    sign loosens it. Only operational members from panos that carry pitch/roll
    participate, so Mapillary runs report nothing here.
    """
    from dataclasses import replace

    sites, frame, _ = fuse(panos, replace(params, apply_pose=False))
    by_id = {p.pano_id: p for p in panos}
    groups = []
    for site in sites:
        ms = [d for d, _ in site.members
              if d.operational and by_id[d.pano_id].camera_pitch is not None
              and by_id[d.pano_id].camera_roll is not None]
        if len(ms) >= 2:
            groups.append(ms)
    if not groups:
        return 'no multi-view sites with pitch/roll poses — nothing to ablate'

    conventions = [('off (no pose)', None), ('+pitch +roll', (1, 1)),
                   ('+pitch -roll', (1, -1)), ('-pitch +roll', (-1, 1)),
                   ('-pitch -roll', (-1, -1)), ('+pitch  0', (1, 0))]
    lines = [f'{len(groups)} frozen multi-view sites '
             f'({sum(len(g) for g in groups)} members); '
             'within-site pairwise member distance (m):',
             f'{"convention":>14}  {"mean":>7}  {"median":>7}  {"pairs":>7}']
    for name, signs in conventions:
        pose_cache = {}
        dists = []
        for ms in groups:
            pts = []
            for d in ms:
                p = by_id[d.pano_id]
                key = p.pano_id
                if key not in pose_cache:
                    if signs is None:
                        pose_cache[key] = geo.pano_pose(
                            {'lat': p.lat, 'lng': p.lng,
                             'camera_heading': p.camera_heading,
                             'camera_pitch': None, 'camera_roll': None,
                             'source': p.source})
                    else:
                        sp, sr = signs
                        pose_cache[key] = geo.pano_pose(
                            {'lat': p.lat, 'lng': p.lng,
                             'camera_heading': p.camera_heading,
                             'camera_pitch': sp * p.camera_pitch,
                             'camera_roll': sr * p.camera_roll,
                             'source': p.source})
                g = geo.detection_ground_point(
                    pose_cache[key], d.x, d.y,
                    camera_height=params.camera_height_m,
                    max_range_m=math.inf, errors=geo.error_model_for(p.source))
                if g is None:
                    break
                pts.append(frame.to_enu(g.lat, g.lng))
            else:
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        dists.append(math.hypot(pts[i][0] - pts[j][0],
                                                pts[i][1] - pts[j][1]))
        dists.sort()
        mean = sum(dists) / len(dists)
        lines.append(f'{name:>14}  {mean:7.3f}  {dists[len(dists) // 2]:7.3f}  '
                     f'{len(dists):7d}')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
