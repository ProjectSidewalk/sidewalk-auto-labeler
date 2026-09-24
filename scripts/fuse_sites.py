"""Stage-2 multi-view association (issue #27): group per-pano detections into
physical curb-ramp sites.

Reads a run's results.jsonl, projects every stored detection to a flat-ground
world point (geo.detection_ground_point with anisotropic error; GSV camera
pitch/roll deliberately NOT applied — the --pose-ablation experiment showed
streetlevel's GSV equirects are already gravity-rectified; Mapillary pose is available
but also off by default, see Camera pose below), and greedily
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

Camera pose (issue #42) is --apply-pose: `off` (the flat raycast: ray elevation is the
pixel's pano-frame elevation), `gravity` (rotate each ray by the pano's stored pitch/roll),
`road` (the same, relative to the local road: the grade along the sequence, from
Mapillary's SfM altitude profile, is subtracted first -- see sequence_grades), or the
default `auto`, which today resolves to off for EVERY source: road-relative was withheld
for Mapillary by the #42 shuffled-grade control (AUTO_ROAD_SOURCES says why;
docs/mapillary-tilt-study.md section 10 has the measurement). GSV is measured to want
`off` (its equirects are gravity-rectified; geo._world_ray). For
Mapillary, blocks written since #42 carry pitch/roll and older ones get them derived here
from source_metadata, so no run needs rewriting to be fused posed. sites_meta.json's
`pose` block counts which panos were posed and, under `road`, how many had no usable
sequence neighbour and fell back to gravity-relative.

Camera height (issue #40) is geo.DEFAULT_CAMERA_HEIGHT_M for every pano unless
--camera-height-m says otherwise: another constant, or `per-pano` for GSV's measured
height where there is one -- from the pano block on runs made since #40, else from the
harvested depth/index.csv beside results.jsonl (scripts/harvest_depth.py). Per-pano is
opt-in on evidence; --implied-height is the instrument that measured why, and
docs/camera-height-study.md has the numbers.

Output: sites.jsonl (one site per line, fused position + covariance + members)
and sites_meta.json (parameters, frame origin, drop counters) beside the input.
Deliberately reads NO manifest.json (cluster-pulled runs lack one) and leaves
send_to_ps.py untouched — what to submit per site is a late decision (#27).

Usage:
    python scripts/fuse_sites.py runs/paterson
    python scripts/fuse_sites.py runs/paterson --pose-ablation   # lock pitch/roll signs
    python scripts/fuse_sites.py runs/richmond --apply-pose road # Mapillary, road-relative
    python scripts/fuse_sites.py runs/paterson --implied-height  # camera height the
                                                                 # imagery implies (#40)
"""
import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import depth as depthlib  # noqa: E402
import geo  # noqa: E402
from detectors import (DETECTION_STORAGE_FLOOR, OPERATIONAL_CONFIDENCE,  # noqa: E402
                       on_camera_rig)

# How fusion rotates rays by camera pose (issue #42). The value is recorded in
# sites_meta.json's params, so a sites file always says which frame it is in.
POSE_OFF = 'off'          # flat raycast in the pano frame
POSE_GRAVITY = 'gravity'  # rotate by the stored pitch/roll (gravity-relative)
POSE_ROAD = 'road'        # ...minus the sequence's road grade where there is one
POSE_AUTO = 'auto'        # the default: POSE_ROAD for AUTO_ROAD_SOURCES, POSE_OFF otherwise
POSE_MODES = (POSE_AUTO, POSE_OFF, POSE_GRAVITY, POSE_ROAD)
# Sources `auto` fuses road-relative: NONE, for now. Road-relative passed the first #42
# rule for Mapillary (at the 25 m cap, on one site set and one GT set, it cut p90 and
# median GT-to-site distance against the flat raycast in all five cities), but FAILED the
# pre-registered shuffled-grade control that #52's GSV result called for (study section
# 10.5): giving each frame another frame's grade from the same sequence did about as well
# in Clovis and Laurens -- so the gain cannot be credited to each frame's own road grade
# (pitch and grade share one SfM; subtracting could cancel shared error) -- and recall on
# the flat raycast's own GT pool fell 4.0 / 2.9 points in Richmond / Annapolis. So the
# road-frame default is WITHHELD pending that question; `--apply-pose road` still works.
# Put 'mapillary' back here only on a control that passes. GSV stays flat regardless: its
# equirects are gravity-rectified and applying their pose loosens every city
# (geo._world_ray, #52). Panoramax: optional pers:pitch/roll, convention unmeasured (#57).
AUTO_ROAD_SOURCES = ()

# Consecutive frames of one sequence within this time gap and horizontal distance define
# a local direction of travel and, through the SfM altitude, a road grade. The bounds are
# the #42 study's: under 2 m the altitude difference is SfM noise over nothing; past 40 m
# (or a minute) the two frames may not share a road segment.
GRADE_MAX_GAP_S = 60.0
GRADE_MIN_DIST_M, GRADE_MAX_DIST_M = 2.0, 40.0


@dataclass(frozen=True)
class FuseParams:
    floor: float = DETECTION_STORAGE_FLOOR
    min_confidence: float = OPERATIONAL_CONFIDENCE
    mask_rig: bool = True            # drop detections on the camera vehicle (see
                                     # detectors.on_camera_rig). False reproduces analysis
                                     # published before the mask existed.
    max_range_m: float = geo.DEFAULT_MAX_RANGE_M
    gate_chi2: float = 9.21          # chi-square(2 dof) 99th pct; loose on purpose —
                                     # the covariance model errs small and false splits
                                     # are cheap, while false merges are braked by the
                                     # cannot-link, the hard cap, and residual rejection
    max_match_m: float = 8.0         # above dual-ramp scatter, below corner spacing
    residual_per_dof_max: float = 3.0
    camera_height_m: float | str = geo.DEFAULT_CAMERA_HEIGHT_M  # or geo.PER_PANO
    apply_pose: str = POSE_AUTO      # one of POSE_MODES; see AUTO_ROAD_SOURCES for what
                                     # the default does per source, and why
    sigma_scale: float = 1.0         # inflate all covariances by scale^2 (model tuning)
    max_vintage_months: int | None = None  # eval-ablation only; None = no gate

    def __post_init__(self):
        # A bool here is a caller from before #42 made this three-way; refusing it beats
        # guessing which of gravity/road `True` meant.
        if self.apply_pose not in POSE_MODES:
            raise ValueError(f'apply_pose must be one of {POSE_MODES}, got {self.apply_pose!r}')

    @property
    def rotates(self):
        """Whether rays MAY be rotated (geo's boolean apply_pose). Per pano, pano_pose
        decides: a pano it leaves unposed raycasts flat whatever this says."""
        return self.apply_pose != POSE_OFF


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
    camera_height_m: float | None = None         # measured (#40); None = unmeasured
    camera_height_spread_m: float | None = None
    pose_origin: str | None = None               # 'block' | 'source_metadata' | None (#42)
    sequence_id: str | None = None               # capture sequence (Mapillary)
    grade_deg: float | None = None               # road grade along travel (sequence_grades)
    travel_bearing_deg: float | None = None

    def pose_fields(self, **overrides):
        """The pano-block fields geo.pano_pose reads, with any of them overridden."""
        return {'lat': self.lat, 'lng': self.lng, 'camera_heading': self.camera_heading,
                'camera_pitch': self.camera_pitch, 'camera_roll': self.camera_roll,
                'source': self.source, 'camera_height_m': self.camera_height_m,
                'camera_height_spread_m': self.camera_height_spread_m, **overrides}


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


def load_depth_index(path):
    """{pano_id: (camera_height_m, height_spread_m)} for the MEASURED heights in a
    harvested depth/index.csv (scripts/harvest_depth.py), or {} if there is none.

    This is how a run made before #40 -- whose pano blocks carry no height -- gets its
    measured heights: the four GSV runs were harvested in full. Rows go through the same
    depth.classify_height as a live fetch, so a stand-in ground or an implausible plane is
    left out here exactly as it would be nulled in a pano block.
    """
    path = Path(path)
    if not path.exists():
        return {}
    heights = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not row['camera_height_m']:
                continue
            h, tilt = float(row['camera_height_m']), float(row['ground_tilt_deg'])
            status = depthlib.classify_height(h, tilt, degenerate=row['degenerate'] == '1')
            if status == depthlib.MEASURED:
                heights[row['panorama_id']] = (h, float(row['height_spread_m']))
    return heights


# Block statuses that are not the last word: no payload, or one that did not parse. A
# harvested index may still have the height for these, so it is consulted as for a
# pre-#40 block. Every other status is a decision made on a real payload.
_UNDECIDED = (None, depthlib.NO_DEPTH, depthlib.UNPARSED)


def sequence_grades(frames):
    """{key: (grade_deg, travel_bearing_deg)} from each sequence's SfM altitude profile.

    ``frames`` is an iterable of (key, sequence_id, captured_at_ms, lat, lng, altitude_m).
    Within one sequence, frames are ordered by capture time and each takes the widest
    usable span around it -- (previous, next), else (previous, self), else (self, next)
    -- where both ends have an altitude, are at most 2 * GRADE_MAX_GAP_S apart and lie
    GRADE_MIN_DIST_M..GRADE_MAX_DIST_M apart horizontally. The grade is the altitude
    change over that distance (positive uphill), the bearing the direction of travel.
    Frames with no usable span, or no sequence or timestamp, are absent from the result:
    the caller decides what they fall back to, and counts them.

    Moved from scripts/mapillary_tilt.py (add_sequence_grade), which measured it; the
    study now calls this, so production and the published numbers share one grade.
    """
    by_seq = {}
    for key, seq, t, lat, lng, alt in frames:
        if seq is None or t is None:
            continue
        by_seq.setdefault(seq, []).append((t, key, lat, lng, alt))
    out = {}
    for fs_ in by_seq.values():
        fs_.sort(key=lambda f: f[0])     # stable: equal timestamps keep input order
        for i, (_t, key, _lat, _lng, _alt) in enumerate(fs_):
            prev = fs_[i - 1] if i > 0 else None
            nxt = fs_[i + 1] if i + 1 < len(fs_) else None
            for a, b in ((prev, nxt), (prev, fs_[i]), (fs_[i], nxt)):
                if a is None or b is None or a[4] is None or b[4] is None:
                    continue
                if (b[0] - a[0]) / 1000.0 > 2 * GRADE_MAX_GAP_S:
                    continue
                d = geo.haversine_m(a[2], a[3], b[2], b[3])
                if not (GRADE_MIN_DIST_M <= d <= GRADE_MAX_DIST_M):
                    continue
                e, n = geo.LocalFrame(a[2], a[3]).to_enu(b[2], b[3])
                out[key] = (math.degrees(math.atan2(b[4] - a[4], d)),
                            math.degrees(math.atan2(e, n)) % 360.0)
                break
    return out


def pose_mode_for(pano, mode):
    """The mode a pano is actually raycast under: POSE_AUTO resolves by source."""
    if mode == POSE_AUTO:
        return POSE_ROAD if pano.source in AUTO_ROAD_SOURCES else POSE_OFF
    return mode


def pano_pose(pano, mode):
    """geo.Pose for a SlimPano under an apply_pose mode.

    Under POSE_OFF (or POSE_AUTO for a source not in AUTO_ROAD_SOURCES) the pose comes
    back WITHOUT pitch/roll, so the raycast is flat whatever apply_pose the caller passes
    geo. Under POSE_ROAD a posed pano with a sequence grade gets its pitch/roll re-expressed
    relative to the road (geo.road_relative_pitch_roll); without a grade it keeps the
    gravity-relative angles -- the fallback sites_meta.json counts."""
    mode = pose_mode_for(pano, mode)
    if mode == POSE_OFF:
        return geo.pano_pose(pano.pose_fields(camera_pitch=None, camera_roll=None))
    if mode == POSE_ROAD and pano.grade_deg is not None \
            and pano.camera_pitch is not None and pano.camera_roll is not None:
        pitch, roll = geo.road_relative_pitch_roll(
            float(pano.camera_pitch), float(pano.camera_roll), float(pano.camera_heading),
            pano.grade_deg, pano.travel_bearing_deg)
        return geo.pano_pose(pano.pose_fields(camera_pitch=pitch, camera_roll=roll))
    return geo.pano_pose(pano.pose_fields())


def pose_counts(panos, params):
    """Where the run's panos got their pose and how each was raycast: `flat`,
    `gravity`, `road_relative`, or `gravity_fallback` -- a pano road mode wanted to
    correct but whose sequence gave no grade. sites_meta.json records it because that
    fallback is the convention the #42 study found WRONG for a vehicle rig on a slope,
    so its rate has to be visible rather than silent."""
    counts = {'mode': params.apply_pose, 'panos': len(panos), 'posed': 0,
              'derived_from_source_metadata': 0, 'flat': 0, 'gravity': 0,
              'road_relative': 0, 'gravity_fallback': 0}
    for p in panos:
        posed = p.camera_pitch is not None and p.camera_roll is not None
        counts['posed'] += posed
        counts['derived_from_source_metadata'] += p.pose_origin == 'source_metadata'
        mode = pose_mode_for(p, params.apply_pose)
        if not posed or mode == POSE_OFF:
            counts['flat'] += 1
        elif mode == POSE_GRAVITY:
            counts['gravity'] += 1
        elif p.grade_deg is not None:
            counts['road_relative'] += 1
        else:
            counts['gravity_fallback'] += 1
    return counts


def load_results(path, depth_index=None, read_heights=True):
    """Stream results.jsonl into SlimPanos, discarding links/history/metadata.
    Records without a position or heading can't be raycast and are dropped
    (counted by the caller via the skipped list).

    Camera heights come from the pano block when it has the #40 fields and they were
    decided on a real payload (a null there is then final). A block from before #40, or
    one whose fetch got no usable payload, is looked up in `depth_index` -- by default
    the depth/index.csv beside the file, if the run's depth was harvested. Note that file
    is a local artifact (not in git): without it, per-pano silently falls back to the
    default for a pre-#40 run, and sites_meta.json's `camera_heights` is the tell.
    read_heights=False skips the index (a 6-13 MB read) for callers that raycast at a
    fixed height anyway.

    Camera pose (#42): pitch/roll come from the block; a Mapillary block that predates
    them (null) gets them from its own source_metadata.computed_rotation through the same
    geo.mapillary_pitch_roll a fresh run writes, so fusion never depends on a backfill.
    Each pano's road grade (sequence_grades) is attached for --apply-pose road."""
    path = Path(path)
    index = load_depth_index(depth_index or path.parent / 'depth' / 'index.csv') \
        if read_heights else {}
    panos, skipped, frames = [], 0, []
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
            if p.get('camera_height_status') in _UNDECIDED:
                height, spread = index.get(p['panorama_id'], (None, None))
            else:
                height, spread = p.get('camera_height_m'), p.get('camera_height_spread_m')
            pitch, roll = p.get('camera_pitch'), p.get('camera_roll')
            meta = p.get('source_metadata') or {}
            origin = 'block' if pitch is not None and roll is not None else None
            if origin is None and p.get('source') == 'mapillary':
                pitch, roll = geo.mapillary_pitch_roll(meta.get('computed_rotation'))
                origin = 'source_metadata' if pitch is not None else None
            panos.append(SlimPano(
                pano_id=p['panorama_id'], lat=p['lat'], lng=p['lng'],
                camera_heading=p['camera_heading'],
                camera_pitch=pitch, camera_roll=roll,
                capture_date=p.get('capture_date'), source=p.get('source') or '',
                detections=[(i, d['x_normalized'], d['y_normalized'], d['confidence'])
                            for i, d in enumerate(rec.get('detections', []))],
                camera_height_m=height, camera_height_spread_m=spread,
                pose_origin=origin, sequence_id=p.get('sequence_id')))
            frames.append((len(panos) - 1, p.get('sequence_id'), meta.get('captured_at'),
                           p['lat'], p['lng'], meta.get('computed_altitude')))
    for i, (grade, bearing) in sequence_grades(frames).items():
        panos[i].grade_deg, panos[i].travel_bearing_deg = grade, bearing
    return panos, skipped


def project(panos, params):
    """Raycast every stored detection >= floor. Returns (dets, frame, drops)."""
    frame = geo.LocalFrame(sum(p.lat for p in panos) / len(panos),
                           sum(p.lng for p in panos) / len(panos))
    drops = {'below_floor': 0, 'on_rig': 0, 'horizon': 0, 'out_of_range': 0}
    dets = []
    s2 = params.sigma_scale ** 2
    for p in panos:
        pose = pano_pose(p, params.apply_pose)
        errors = geo.error_model_for(p.source)
        months = _months(p.capture_date)
        for i, x, y, conf in p.detections:
            if conf < params.floor:
                drops['below_floor'] += 1
                continue
            if params.mask_rig and on_camera_rig(y):
                # On the camera vehicle, not the street: it would raycast to a ghost point
                # ~1.5 m from the camera and, since the rig is fixed in the pano frame, do
                # so on every pano of the sequence - a trail of spurious sites.
                drops['on_rig'] += 1
                continue
            g = geo.detection_ground_point(
                pose, x, y, camera_height=params.camera_height_m,
                max_range_m=params.max_range_m, errors=errors,
                apply_pose=params.rotates)
            if g is None:
                unbounded = geo.detection_ground_point(
                    pose, x, y, camera_height=params.camera_height_m,
                    max_range_m=math.inf, errors=errors,
                    apply_pose=params.rotates)
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
             'camera_heights': camera_height_counts(panos, params),
             'pose': pose_counts(panos, params),
             'frame_origin': {'lat0': frame.lat0, 'lng0': frame.lng0}}
    return sites, frame, stats


def camera_height_counts(panos, params):
    """How the run's panos got their camera height -- the provenance a reader of
    sites_meta.json needs to know which frame the positions are in."""
    if params.camera_height_m != geo.PER_PANO:
        return {'fixed_m': params.camera_height_m, 'panos': len(panos)}
    measured = sum(1 for p in panos if p.camera_height_m is not None)
    return {'measured': measured, 'fallback': len(panos) - measured}


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


def camera_height_arg(value):
    """argparse type for --camera-height-m: meters, or geo.PER_PANO."""
    return value if value == geo.PER_PANO else float(value)


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
    ap.add_argument('--camera-height-m', type=camera_height_arg,
                    default=geo.DEFAULT_CAMERA_HEIGHT_M,
                    help='camera height in meters for every pano, or "per-pano" for '
                         "each GSV pano's depth-measured height where it has one (#40; "
                         'opt-in -- see docs/camera-height-study.md)')
    ap.add_argument('--sigma-scale', type=float, default=1.0)
    ap.add_argument('--apply-pose', choices=POSE_MODES, nargs='?', const=POSE_GRAVITY,
                    default=FuseParams.apply_pose,
                    help='rotate rays by camera pose: auto (the default; currently off for '
                         'every source -- the #42 shuffled-grade control withheld road for '
                         'Mapillary, see AUTO_ROAD_SOURCES), off '
                         '(flat raycast), gravity (stored pitch/roll; a bare --apply-pose '
                         "means this), or road (minus the sequence's road grade). Measured "
                         'to hurt on GSV -- see the --pose-ablation report')
    ap.add_argument('--pose-ablation', action='store_true',
                    help='report within-site spread under each pitch/roll sign '
                         'convention instead of writing sites')
    ap.add_argument('--implied-height', action='store_true',
                    help='report the camera height the imagery implies (bearing-only '
                         'triangulation of multi-view sites) against the measured one, '
                         'by capture year, instead of writing sites (#40)')
    ap.add_argument('--depth-index', type=Path, default=None,
                    help='harvested depth/index.csv to read heights from for a run '
                         'made before #40 (default: <run>/depth/index.csv)')
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

    panos, skipped = load_results(
        jsonl, args.depth_index,
        read_heights=args.camera_height_m == geo.PER_PANO or args.implied_height)
    if skipped:
        print(f'skipped {skipped} records without position/heading')
    if not panos:
        sys.exit('no usable records')

    if args.pose_ablation:
        print(pose_ablation_report(panos, params))
        return
    if args.implied_height:
        print(implied_height_report(panos, params))
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

    That participation filter is why the report leads with pose coverage. On GSV every
    pano carries pitch/roll and the ablation covers the whole run; Panoramax is mixed
    (pers:pitch/pers:roll are optional — measured 28% of panos carry them, see
    geo.pano_pose), so the numbers below would otherwise describe a self-selected subset
    while reading like a statement about the run. A subset drawn by "which rig wrote a
    pose" is not a random one, so the header says so out loud.
    """
    from dataclasses import replace

    sites, frame, _ = fuse(panos, replace(params, apply_pose=POSE_OFF))
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

    posed = sum(1 for p in panos
                if p.camera_pitch is not None and p.camera_roll is not None)
    coverage = [f'pose coverage: {posed}/{len(panos)} panos carry pitch+roll '
                f'({100.0 * posed / len(panos):.0f}%)']
    if posed < len(panos):
        coverage.append(
            '  ⚠ mixed population — the table below describes only the posed panos, '
            'which are\n    self-selected by capture rig, not a random sample of the run.')

    conventions = [('off (no pose)', None), ('+pitch +roll', (1, 1)),
                   ('+pitch -roll', (1, -1)), ('-pitch +roll', (-1, 1)),
                   ('-pitch -roll', (-1, -1)), ('+pitch  0', (1, 0))]
    lines = coverage + [f'{len(groups)} frozen multi-view sites '
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
                            p.pose_fields(camera_pitch=None, camera_roll=None))
                    else:
                        sp, sr = signs
                        pose_cache[key] = geo.pano_pose(
                            p.pose_fields(camera_pitch=sp * p.camera_pitch,
                                          camera_roll=sr * p.camera_roll))
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
        if not dists:  # every ray in every group missed the ground under this convention
            lines.append(f'{name:>14}  {"—":>7}  {"—":>7}  {0:7d}')
            continue
        mean = sum(dists) / len(dists)
        lines.append(f'{name:>14}  {mean:7.3f}  {dists[len(dists) // 2]:7.3f}  '
                     f'{len(dists):7d}')
    return '\n'.join(lines)


IMPLIED_MIN_ANGLE_DEG = 30.0   # below this two bearings barely constrain a range
IMPLIED_RANGE_M = (1.0, 30.0)  # a triangulated range outside this is a bad pairing


def implied_heights(sites, frame, by_id):
    """{pano_id: [implied camera height]} from bearing-only triangulation (#40).

    Two panos that see one ramp fix its position from their bearings alone -- the 2D
    intersection of the two rays, which does not depend on camera height -- and a pano
    whose ray meets the ground at that range from depression d implies a camera height of
    range * tan(d). Only operational members, and only pairs whose rays cross at
    IMPLIED_MIN_ANGLE_DEG or more.
    """
    implied = {}
    for site in sites:
        ms = [d for d, _ in site.members if d.operational]
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                di, dj = ms[a], ms[b]
                pi, pj = by_id[di.pano_id], by_id[dj.pano_id]
                ci, cj = frame.to_enu(pi.lat, pi.lng), frame.to_enu(pj.lat, pj.lng)
                bi = math.radians(di.ground.bearing_deg)
                bj = math.radians(dj.ground.bearing_deg)
                ui, uj = (math.sin(bi), math.cos(bi)), (math.sin(bj), math.cos(bj))
                cross = ui[0] * uj[1] - ui[1] * uj[0]
                if abs(cross) < math.sin(math.radians(IMPLIED_MIN_ANGLE_DEG)):
                    continue
                dx, dy = cj[0] - ci[0], cj[1] - ci[1]
                ri = (dx * uj[1] - dy * uj[0]) / cross
                rj = (dx * ui[1] - dy * ui[0]) / cross
                lo, hi = IMPLIED_RANGE_M
                if not (lo < ri < hi and lo < rj < hi):
                    continue
                for d, r in ((di, ri), (dj, rj)):
                    depression = (d.y - 0.5) * math.pi
                    implied.setdefault(d.pano_id, []).append(r * math.tan(depression))
    return implied


def _median(values):
    v = sorted(values)
    n = len(v)
    return (v[n // 2] + v[(n - 1) // 2]) / 2.0


def implied_height_report(panos, params):
    """Measured vs imagery-implied camera height, by capture year (#40).

    The implied height is independent of the height model only per pair; which pairs
    exist is not, because association ran under params.camera_height_m, and implied
    heights drift toward whatever height the association used (paterson's 2025 rig reads
    1.93 m associated at 1.8, 2.14 m at 2.6). So read it as a fixed-point search: re-run
    with --camera-height-m set near the implied value until the two agree. With
    `per-pano`, iterate on a scale instead (docs/camera-height-study.md does, by
    monkeypatching; the self-consistent scale was 1.06-1.16 by city).
    """
    sites, frame, _ = fuse(panos, params)
    by_id = {p.pano_id: p for p in panos}
    implied = {pid: _median(hs) for pid, hs in implied_heights(sites, frame, by_id).items()}
    if not implied:
        return 'no multi-view pairs to triangulate'

    by_year = {}
    for pid, h in implied.items():
        p = by_id[pid]
        by_year.setdefault((p.capture_date or '????')[:4], []).append((h, p.camera_height_m))
    measured = [(h, m) for rows in by_year.values() for h, m in rows if m is not None]
    lines = [f'associated at camera height {params.camera_height_m}; '
             f'{len(implied)} panos with a triangulated implied height',
             f'implied height: median {_median(implied.values()):.3f} m over all panos',
             f'{"capture":>7}  {"panos":>6}  {"measured":>8}  {"med measured":>12}  '
             f'{"med implied":>11}  {"implied/measured":>16}']
    for year in sorted(by_year):
        rows = by_year[year]
        meas = [(h, m) for h, m in rows if m is not None]
        med_m = f'{_median(m for _, m in meas):12.3f}' if meas else f'{"—":>12}'
        ratio = f'{_median(h / m for h, m in meas):16.3f}' if meas else f'{"—":>16}'
        lines.append(f'{year:>7}  {len(rows):6d}  {len(meas):8d}  {med_m}  '
                     f'{_median(h for h, _ in rows):11.3f}  {ratio}')
    if measured:
        lines.append(f'{"all":>7}  {len(implied):6d}  {len(measured):8d}  '
                     f'{_median(m for _, m in measured):12.3f}  '
                     f'{_median(implied.values()):11.3f}  '
                     f'{_median(h / m for h, m in measured):16.3f}')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
