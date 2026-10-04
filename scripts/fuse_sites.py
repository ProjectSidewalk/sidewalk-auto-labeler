"""Stage-2 multi-view association (issue #27): group per-pano detections into
physical curb-ramp sites.

Reads a run's results.jsonl, projects every stored detection to a flat-ground
world point (geo.detection_ground_point with anisotropic error; GSV camera
pitch/roll deliberately NOT applied — the --pose-ablation experiment showed
that applying them loosens every city (the full pose overshoots; see
geo._world_ray and issue #113); Mapillary pose is available
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
`off` (applying its full pose loosens every city; geo._world_ray, #113). For
Mapillary, blocks written since #42 carry pitch/roll and older ones get them derived here
from source_metadata, so no run needs rewriting to be fused posed. sites_meta.json's
`pose` block counts which panos were posed and, under `road`, how many had no usable
sequence neighbour and fell back to gravity-relative.

Camera height: the CLI default is `auto` (issue #79). GSV panos are raycast at a
per-rig height, derived from the run's own depth-measured heights by capture year:
gsv_rig_assignment, the rule the #79 placement oracle selected as arm (d)
(docs/placement-oracle.md). Mapillary and Panoramax panos, and a GSV run with no measured
height at all -- or none in a year that meets the measurement minimum -- stay at
geo.DEFAULT_CAMERA_HEIGHT_M, as do undated GSV panos. sites_meta.json's `camera_heights`
block records what `auto` resolved to, with the year-by-year assignment. Only this CLI
defaults to `auto`, and only for a fuse: --pose-ablation and --implied-height keep 2.6 m
unless asked, as do FuseParams() and every analysis script, so published numbers stay
reproducible.

Explicitly: --camera-height-m takes a constant (2.6 reproduces every fuse before #79), or
`per-pano` for GSV's measured
height where there is one -- from the pano block on runs made since #40, else from the
harvested depth/index.csv beside results.jsonl (scripts/harvest_depth.py). Per-pano is
opt-in on evidence; --implied-height is the instrument that measured why, and
docs/camera-height-study.md has the numbers. For Mapillary/Panoramax, which serve no
depth, `per-rig` reads a per-capture-rig height table, runs/<name>/camera_heights.json
(issue #53; scripts/mapillary_height.py measures and writes it, bound to results.jsonl by
its sha256) -- also opt-in, see docs/mapillary-camera-height.md.

Output: sites.jsonl (one site per line, fused position + covariance + members)
and sites_meta.json (parameters, frame origin, drop counters) beside the input.
Deliberately reads NO manifest.json (cluster-pulled runs lack one) and leaves
send_to_ps.py untouched — what to submit per site is a late decision (#27).

Usage:
    python scripts/fuse_sites.py runs/paterson                   # camera height: auto
    python scripts/fuse_sites.py runs/paterson --camera-height-m 2.6   # the pre-#79 default
    python scripts/fuse_sites.py runs/paterson --pose-ablation   # lock pitch/roll signs
    python scripts/fuse_sites.py runs/richmond --apply-pose road # Mapillary, road-relative
    python scripts/fuse_sites.py runs/paterson --implied-height  # camera height the
                                                                 # imagery implies (#40)
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import depth as depthlib  # noqa: E402
import geo  # noqa: E402
from detectors import (BORDER_EXCLUDE, DECODE_ARGMAX, DETECTION_STORAGE_FLOOR,  # noqa: E402
                       OPERATIONAL_CONFIDENCE, on_camera_rig, record_border, record_decode,
                       single_border, single_decode)
from collections import Counter  # noqa: E402

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
# Put 'mapillary' back here only on a control that passes. GSV stays flat regardless:
# applying its full pose loosens every city (geo._world_ray, #52). Its equirects are
# rig-frame, not gravity-rectified (#113). A partial pose (~0.18 pitch, ~0.38 roll)
# tightens held-out sites and beats a magnitude-matched shuffle, but FAILED #116's
# pre-registered recall clause (Bend, 2 of 157 ramps), so there is no `partial` mode
# (docs/gsv-partial-pose-study.md). Panoramax:
# optional pers:pitch/roll, convention unmeasured (#57).
# #51 re-ran the same control with a road grade that never saw the SfM (USGS 3DEP DEM,
# --grade-source dem) and it FAILED again, on (i) and (ii) (docs/dem-grade-study.md): the
# DEM grade matches the SfM one in fusion, so a better grade does not change this.
#
# WHO FOLLOWS THIS: every caller that leaves FuseParams.apply_pose at its default --
# fuse_sites.py's own CLI (--apply-pose auto, so production sites.jsonl) and
# site_explorer.py (a bare fs.FuseParams()). The analysis scripts that reproduce a
# committed artifact PIN `off` explicitly and do not follow it: eval_sites.py (CLI default
# off), mined_precision.py, eval_ps_clustering.py, reprojection_residual.py, and mapillary_tilt.py's ablation /
# eval / precondition. So adding a source here silently changes what the followers produce (and
# nothing in their output says so beyond sites_meta.json's `pose` block) -- check them.
AUTO_ROAD_SOURCES = ()

# Sources whose pose --apply-pose gravity/road should not be trusted to rotate, and why;
# fuse_sites.py warns (stderr) when a run holds any. Matched as for site_explorer: a known
# name, else GSV (legacy GSV records store streetlevel's raw source string, e.g. "launch").
UNGRADED_POSE_WARNINGS = {
    'gsv': 'GSV has no sequence grade, so `road` is 100% gravity fallback; rotating its '
           'rays by the full pose loosened every city (#52, #113)',
    'panoramax': "Panoramax's pitch/roll convention is unmeasured (#57)",
}

# Consecutive frames of one sequence within this time gap and horizontal distance define
# a local direction of travel and, through the SfM altitude, a road grade. The bounds are
# the #42 study's: under 2 m the altitude difference is SfM noise over nothing; past 40 m
# (or a minute) the two frames may not share a road segment.
GRADE_MAX_GAP_S = 60.0
GRADE_MIN_DIST_M, GRADE_MAX_DIST_M = 2.0, 40.0

# Where road mode's grade comes from (#51). `sfm` is production's sequence_grades on the
# SfM altitude; the other two are columns of runs/<name>/dem/grades.csv, written by
# scripts/dem_grade.py: `sfm-smoothed` (a +-20 m least-squares fit on the same SfM
# altitude) and `dem` (the same fit on USGS 3DEP elevation, independent of the SfM).
# The travel bearing is sequence_grades' in every case; only the grade value changes.
GRADE_SFM, GRADE_SFM_SMOOTHED, GRADE_DEM = 'sfm', 'sfm-smoothed', 'dem'
GRADE_SOURCES = (GRADE_SFM, GRADE_SFM_SMOOTHED, GRADE_DEM)
GRADE_CSV_COLUMN = {GRADE_SFM_SMOOTHED: 'grade_sfm_smoothed_deg', GRADE_DEM: 'grade_dem_deg'}


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
    sigma_peak_px: float | None = None  # heatmap-peak 1-sigma, heatmap px (#111); None =
                                     # geo.SIGMA_PEAK_PX_DEFAULT via the source's ErrorModel
    max_vintage_months: int | None = None  # eval-ablation only; None = no gate
    grade_source: str = GRADE_SFM    # one of GRADE_SOURCES (#51); load_results applies it,
                                     # this records it in sites_meta.json

    def __post_init__(self):
        # A bool here is a caller from before #42 made this three-way; refusing it beats
        # guessing which of gravity/road `True` meant.
        if self.apply_pose not in POSE_MODES:
            raise ValueError(f'apply_pose must be one of {POSE_MODES}, got {self.apply_pose!r}')
        if self.grade_source not in GRADE_SOURCES:
            raise ValueError(f'grade_source must be one of {GRADE_SOURCES}, '
                             f'got {self.grade_source!r}')

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
    ground_tilt_deg: float | None = None         # the ground plane's tilt (#44 QC reads it)
    camera_height_vintage_m: float | None = None  # median measured height, same capture
                                                  # year in this run (#44 QC gate)
    pose_origin: str | None = None               # 'block' | 'source_metadata' | None (#42)
    sequence_id: str | None = None               # capture sequence (Mapillary)
    grade_deg: float | None = None               # road grade along travel (sequence_grades)
    travel_bearing_deg: float | None = None
    grade_origin: str | None = None              # GRADE_SOURCES member that set grade_deg
    height_group: str | None = None              # camera_heights.json group (#53, per-rig)
    height_table: dict | None = None             # ...and that table's provenance (shared)
    height_from_index: bool = False              # height read from depth/index.csv (#47)
    decode: str = DECODE_ARGMAX                  # the record's peak decode (#111)
    border: str = BORDER_EXCLUDE                 # the record's peak border rule (#130)

    def pose_fields(self, **overrides):
        """The pano-block fields geo.pano_pose reads, with any of them overridden."""
        return {'lat': self.lat, 'lng': self.lng, 'camera_heading': self.camera_heading,
                'camera_pitch': self.camera_pitch, 'camera_roll': self.camera_roll,
                'source': self.source, 'camera_height_m': self.camera_height_m,
                'camera_height_spread_m': self.camera_height_spread_m,
                'ground_tilt_deg': self.ground_tilt_deg,
                'camera_height_vintage_m': self.camera_height_vintage_m, **overrides}


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
    """{pano_id: (camera_height_m, height_spread_m, ground_tilt_deg)} for the MEASURED
    heights in a harvested depth/index.csv (scripts/harvest_depth.py), or {} if there is none.

    This is how a run made before #40 -- whose pano blocks carry no height -- gets its
    measured heights: the four GSV runs were harvested in full. Rows go through the same
    depth.classify_height as a live fetch, so a stand-in ground or an implausible plane is
    left out here exactly as it would be nulled in a pano block.

    An index from before #47 (no n_standin_planes column) is REFUSED
    (depth.require_current_index): its spread still takes Google's stand-in planes in, and
    nothing else in the file says so. `harvest_depth.py <run> --reindex` rebuilds it offline.
    """
    path = Path(path)
    if not path.exists():
        return {}
    heights = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        depthlib.require_current_index(reader.fieldnames, path)
        for row in reader:
            if not row['camera_height_m']:
                continue
            h, tilt = float(row['camera_height_m']), float(row['ground_tilt_deg'])
            status = depthlib.classify_height(h, tilt, degenerate=row['degenerate'] == '1')
            if status == depthlib.MEASURED:
                heights[row['panorama_id']] = (h, float(row['height_spread_m']), tilt)
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
              'road_relative': 0, 'gravity_fallback': 0,
              # #51: which grade road mode subtracts, and on how many panos load_results
              # replaced the SfM grade with it (0 under `sfm`; a mismatch is the tell)
              'grade_source': params.grade_source,
              'grade_replaced': sum(1 for p in panos
                                    if p.grade_origin not in (None, GRADE_SFM))}
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


def pose_source_warnings(panos, mode):
    """Warnings (strings) for an explicit --apply-pose gravity/road over panos it was not
    measured for: GSV and Panoramax. Empty for off/auto, and for all-Mapillary runs."""
    if mode not in (POSE_GRAVITY, POSE_ROAD):
        return []
    counts = {}
    for p in panos:
        kind = source_kind(p.source)
        if kind in UNGRADED_POSE_WARNINGS:
            counts[kind] = counts.get(kind, 0) + 1
    return [f'WARNING: --apply-pose {mode} on {n} {kind} pano(s): {UNGRADED_POSE_WARNINGS[kind]}'
            for kind, n in sorted(counts.items())]


HEIGHT_TABLE_NAME = 'camera_heights.json'   # per-rig camera heights beside results.jsonl

# Camera-height AUTO (#79): the fuse_sites CLI default. GSV panos get a per-rig height by
# capture year, the rule the placement oracle selected (arm (d), docs/placement-oracle.md):
# a year whose median depth-measured height is below GSV_RIG_CUT_M is the low 2025-26 rig
# and raycasts at GSV_RIG_LOW_M; every other year at GSV_RIG_HIGH_M -- but a year keeps its
# own median-derived height only with >= GSV_RIG_MIN_MEASURED measured panos that are
# >= GSV_RIG_MIN_SHARE of its dated panos, since a median of a few payloads misfired on
# old imagery (Gainesville 2015/2018 under the oracle's (c)). The rig cannot be read from
# the year itself (the new rig arrives in 2025 in Paterson, 2026 in Gainesville) or from
# any GSV metadata field. scripts/inventory_oracle.py scores exactly this function.
HEIGHT_AUTO = 'auto'
GSV_RIG_CUT_M = 2.1          # = height_qc.LOW_VINTAGE_M / OPTION_C[0] (a test ties them)
GSV_RIG_LOW_M = 2.0
GSV_RIG_HIGH_M = 2.5
GSV_RIG_MIN_MEASURED = 50
GSV_RIG_MIN_SHARE = 0.5


def rig_year_qualifies(n_measured, n_dated, min_measured=GSV_RIG_MIN_MEASURED,
                       min_share=GSV_RIG_MIN_SHARE):
    """Whether a capture year is measured well enough to keep its median-derived height:
    >= min_measured measured panos AND >= min_share of the year's dated GSV panos, both
    inclusive.

    Example:
        >>> rig_year_qualifies(50, 100), rig_year_qualifies(49, 49), rig_year_qualifies(60, 121)
        (True, False, False)
    """
    return n_measured >= min_measured and n_measured / n_dated >= min_share


def source_kind(source):
    """'mapillary', 'panoramax' or 'gsv' for a pano block's `source` string."""
    src = (source or '').lower()
    return 'mapillary' if 'mapillary' in src else 'panoramax' if 'panoramax' in src else 'gsv'


def gsv_rig_assignment(panos, cut=GSV_RIG_CUT_M, low=GSV_RIG_LOW_M, high=GSV_RIG_HIGH_M,
                       min_measured=GSV_RIG_MIN_MEASURED, min_share=GSV_RIG_MIN_SHARE):
    """{capture year: (median measured height or None, n_measured, n_dated, height)} over
    the GSV panos in `panos` (measured = camera_height_m non-null as load_results read it).

    Example: 606 dated panos, 225 measured at a 1.93 m median -> 37% < 50% -> `high`
    (2.5 m); 23,766 dated, 21,950 measured at 1.76 m -> `low` (2.0 m).
    """
    heights, dated = {}, {}
    for p in panos:
        year = (p.capture_date or '')[:4]
        if not year or source_kind(p.source) != 'gsv':
            continue
        dated[year] = dated.get(year, 0) + 1
        if p.camera_height_m is not None:
            heights.setdefault(year, []).append(p.camera_height_m)
    out = {}
    for year, n in dated.items():
        hs = sorted(heights.get(year, []))
        med = _median(hs) if hs else None
        ok = med is not None and rig_year_qualifies(len(hs), n, min_measured, min_share)
        out[year] = (med, len(hs), n, (low if med < cut else high) if ok else high)
    return out


def apply_gsv_rig_heights(panos, assignment):
    """Copies of `panos` for a geo.PER_RIG fuse under `assignment` (gsv_rig_assignment):
    a dated GSV pano takes its year's height, with no spread (the error model's flat sigma)
    and height_group '<year>:<height>'; an undated GSV pano, and every Mapillary/Panoramax
    pano, gets None -- geo.camera_height_for's DEFAULT_CAMERA_HEIGHT_M under PER_RIG."""
    out = []
    for p in panos:
        year = (p.capture_date or '')[:4]
        h = (assignment[year][3] if source_kind(p.source) == 'gsv' and year in assignment
             else None)
        out.append(replace(p, camera_height_m=h, camera_height_spread_m=None,
                           camera_height_vintage_m=None, height_table=None,
                           height_group=f'{year}:{h:g}' if h is not None else None))
    return out


def resolve_auto_height(panos):
    """(panos, camera_height_m for FuseParams, provenance dict) for --camera-height-m auto.

    GSV panos with at least one capture year that meets the measurement minimum
    (rig_year_qualifies) -> PER_RIG over gsv_rig_assignment. Otherwise -> the constant
    DEFAULT_CAMERA_HEIGHT_M for every pano: no GSV pano, no measured height on any (a pre-#40
    run whose depth was never harvested), or measured heights too thin in every year (a
    partly harvested depth/index.csv). Without one well-measured year the rig cannot be
    told apart, and gsv_rig_assignment would put every pano at GSV_RIG_HIGH_M -- 2.5 m
    everywhere, a silent default change nobody measured. The oracle's arm (d) scores
    gsv_rig_assignment directly and is unaffected: every GSV oracle city has such a year."""
    n_gsv = sum(source_kind(p.source) == 'gsv' for p in panos)
    measured = sum(source_kind(p.source) == 'gsv' and p.camera_height_m is not None
                   for p in panos)
    rule = {'cut_m': GSV_RIG_CUT_M, 'low_m': GSV_RIG_LOW_M, 'high_m': GSV_RIG_HIGH_M,
            'min_measured': GSV_RIG_MIN_MEASURED, 'min_share': GSV_RIG_MIN_SHARE}
    assignment = gsv_rig_assignment(panos) if measured else {}
    if not any(rig_year_qualifies(k, n) for _, k, n, _ in assignment.values()):
        why = ('no GSV panos' if not n_gsv else
               f'none of {n_gsv} GSV panos has a measured height (harvest depth first: '
               'scripts/harvest_depth.py)' if not measured else
               f'{measured} of {n_gsv} GSV panos measured, but no capture year has '
               f'>= {GSV_RIG_MIN_MEASURED} measured that are >= {GSV_RIG_MIN_SHARE:.0%} '
               'of it (finish the depth harvest: scripts/harvest_depth.py)')
        return panos, geo.DEFAULT_CAMERA_HEIGHT_M, {
            'mode': HEIGHT_AUTO, 'resolved': geo.DEFAULT_CAMERA_HEIGHT_M, 'reason': why,
            'gsv_panos': n_gsv, 'gsv_measured': measured, 'rule': rule}
    return apply_gsv_rig_heights(panos, assignment), geo.PER_RIG, {
        'mode': HEIGHT_AUTO, 'resolved': 'gsv-per-rig', 'gsv_panos': n_gsv,
        'gsv_measured': measured, 'rule': rule,
        'assignment': {y: {'median_m': None if m is None else round(m, 4), 'measured': k,
                           'dated': n, 'height_m': h}
                       for y, (m, k, n, h) in sorted(assignment.items())}}


def file_sha256(path):
    """Hex sha256 of a file, streamed (results.jsonl runs to ~100 MB)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_grades_csv(path, grade_source, results_path=None):
    """{panorama_id: grade_deg or None} from scripts/dem_grade.py's grades.csv, for a
    non-`sfm` grade source (#51).

    Refuses (SystemExit, naming the command that writes it) when:
    - the file, its grades.json sidecar or the column is missing;
    - the sidecar's results_sha256 is not `results_path`'s -- the grades were sampled at
      another file's positions (e.g. results.jsonl's, fused as results.raw.jsonl), so
      each frame would mix two position sets;
    - the CSV is not the one the sidecar recorded (truncated, stale or edited);
    - a cell is not a finite number. An EMPTY cell is a legitimate "no grade here".
    """
    path = Path(path)
    column = GRADE_CSV_COLUMN[grade_source]
    hint = (f'run `python scripts/dem_grade.py <city>` to write it (grade source '
            f'{grade_source!r} reads its {column} column)')
    if not path.exists():
        raise SystemExit(f'no grades file at {path}: {hint}')
    sidecar = path.with_suffix('.json')
    if not sidecar.exists():
        raise SystemExit(f'no {sidecar.name} beside {path}, so nothing ties it to a '
                         f'results file: {hint}')
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    if results_path is not None and meta.get('results_sha256') != file_sha256(results_path):
        raise SystemExit(f"{path} was built from {meta.get('results_file')!r} "
                         f"(sha256 {str(meta.get('results_sha256'))[:12]}...), not from "
                         f'{Path(results_path).name} as it is now: its grades were sampled at '
                         f'that file\'s positions. Re-run dem_grade.py on this file, or pass '
                         f'--grades for the right one')
    if meta.get('grades_sha256') != file_sha256(path):
        raise SystemExit(f'{path} is not the file {sidecar.name} recorded (truncated, stale '
                         f'or edited): {hint}')
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or ()):
            raise SystemExit(f'{path} has no {column} column: {hint}')
        out = {}
        for row in reader:
            v = row[column]
            if v:
                v = float(v)
                if not math.isfinite(v):
                    raise SystemExit(f"{path}: non-finite {column} {row[column]!r} for "
                                     f"{row['panorama_id']}")
            out[row['panorama_id']] = v if v != '' else None
        return out


def apply_height_table(panos, table_path, results_path):
    """Fill SlimPano.camera_height_m / camera_height_spread_m from a per-rig table (#53).

    Refuses (ValueError) a table whose results_sha256 is not this results.jsonl's -- a
    table measured on another file says nothing about this one -- and a run holding any
    non-crowdsourced pano: GSV serves a real per-pano height (use `per-pano`). Every
    sequence resolves through table['sequences'] to one group; a group whose height_m is
    the table's default_m (it failed a rule, or was never measured) leaves the pano at
    None, so camera_height_for falls back exactly as a 2.6 m fuse would. Whether a group
    applies is the table's own `applied` flag. Under PER_RIG the spread field carries the
    group's 1-sigma (geo.camera_height_for). The lookup key is SlimPano.sequence_id, which
    load_results takes from pano.sequence_id else source_metadata.sequence -- the same
    expression mapillary_height.pano_groups keys the table by."""
    table_path = Path(table_path)
    with open(table_path, encoding='utf-8') as f:
        table = json.load(f)
    if table.get('schema') != 1:
        raise ValueError(f'{table_path}: unknown camera-height table schema '
                         f'{table.get("schema")!r}')
    sha = file_sha256(results_path)
    if table.get('results_sha256') != sha:
        raise ValueError(f'{table_path} was measured on a different results.jsonl '
                         f'(table {table.get("results_sha256")}, file {sha}); re-run '
                         'scripts/mapillary_height.py')
    other = {p.source for p in panos if p.source not in geo.CROWDSOURCED_SOURCES}
    if other:
        raise ValueError(f'per-rig heights are for crowdsourced sources only; this run '
                         f'holds {sorted(other)} panos (GSV: use per-pano)')
    prov = {'table': str(table_path), 'sha256': file_sha256(table_path),
            'grain': table.get('grain'), 'recommended': table.get('recommended')}
    for p in panos:
        key = table['sequences'].get(p.sequence_id)
        group = table['groups'].get(key) if key is not None else None
        p.height_group, p.height_table = key, prov
        if group is None or not group.get('applied'):
            p.camera_height_m = p.camera_height_spread_m = None
        else:
            p.camera_height_m, p.camera_height_spread_m = group['height_m'], group['sigma_m']
    return panos


def load_results(path, depth_index=None, read_heights=True, height_table=None,
                 grade_source=GRADE_SFM, grades_path=None, allow_mixed_decode=False,
                 allow_mixed_border=False):
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
    Each pano's road grade (sequence_grades) is attached for --apply-pose road.

    grade_source (#51): under anything but `sfm`, each graded pano's grade_deg is then
    replaced from `grades_path` (default: dem/grades.csv beside the file); the travel
    bearing is kept. A pano with no value there gets grade_deg=None, i.e. the gravity
    fallback pose_counts already counts.

    height_table (#53): a camera_heights.json path, for a PER_RIG fuse -- the heights
    come from it instead (apply_height_table; it refuses a mismatched file or a GSV run).

    Peak decode (#111): each SlimPano carries its record's decode (detectors.record_decode),
    and a file mixing argmax and gaussian records raises ValueError unless
    allow_mixed_decode -- the two place the same peaks in different frames. Likewise the
    border rule (#130; detectors.record_border): a file mixing `exclude` and `keep` records
    raises unless allow_mixed_border."""
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
            from_index = False
            if p.get('camera_height_status') in _UNDECIDED:
                height, spread, tilt = index.get(p['panorama_id'], (None, None, None))
                from_index = height is not None
            else:
                height, spread = p.get('camera_height_m'), p.get('camera_height_spread_m')
                tilt = p.get('ground_tilt_deg')
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
                ground_tilt_deg=tilt, pose_origin=origin,
                sequence_id=p.get('sequence_id') or meta.get('sequence'),
                height_from_index=from_index, decode=record_decode(rec),
                border=record_border(rec)))
            frames.append((len(panos) - 1, p.get('sequence_id'), meta.get('captured_at'),
                           p['lat'], p['lng'], meta.get('computed_altitude')))
    single_decode(Counter(p.decode for p in panos), path.name, allow_mixed_decode)
    single_border(Counter(p.border for p in panos), path.name, allow_mixed_border)
    for i, (grade, bearing) in sequence_grades(frames).items():
        panos[i].grade_deg, panos[i].travel_bearing_deg = grade, bearing
        panos[i].grade_origin = GRADE_SFM
    if grade_source != GRADE_SFM:
        if grade_source not in GRADE_CSV_COLUMN:
            raise ValueError(f'grade_source must be one of {GRADE_SOURCES}, '
                             f'got {grade_source!r}')
        grades = load_grades_csv(grades_path or path.parent / 'dem' / 'grades.csv',
                                 grade_source, results_path=path)
        missing = [p.pano_id for p in panos
                   if p.travel_bearing_deg is not None and p.pano_id not in grades]
        if missing:
            raise SystemExit(f'{len(missing)} graded pano(s) have no row in the grades file '
                             f'(first: {missing[0]}); it does not cover {path.name}')
        for p in panos:
            if p.travel_bearing_deg is None:
                continue                    # no travel direction: nothing to rotate about
            p.grade_deg = grades[p.pano_id]
            p.grade_origin = grade_source if p.grade_deg is not None else None
    attach_vintage_medians(panos)
    if height_table is not None:
        apply_height_table(panos, height_table, path)
    return panos, skipped


def attach_vintage_medians(panos, min_panos=None):
    """Set each measured pano's camera_height_vintage_m: the median measured height over
    the run's panos of the same capture year, the vintage depth.believe_height's QC gate
    compares against (#44). A run is bound to one imagery source, so the year is the
    vintage -- exactly as scripts/height_qc.py measured it. A vintage with fewer than
    `min_panos` measured panos (default depth.QC_MIN_VINTAGE_PANOS), or an undated pano,
    gets no median: a median of one or two panos can hardly be deviated from, and the
    undated bucket would pool unrelated years."""
    min_panos = depthlib.QC_MIN_VINTAGE_PANOS if min_panos is None else min_panos
    by_year = {}
    for p in panos:
        year = (p.capture_date or '')[:4]
        if p.camera_height_m is not None and year:
            by_year.setdefault(year, []).append(p.camera_height_m)
    medians = {y: _median(v) for y, v in by_year.items() if len(v) >= min_panos}
    for p in panos:
        if p.camera_height_m is not None:
            p.camera_height_vintage_m = medians.get((p.capture_date or '')[:4])


def project(panos, params):
    """Raycast every stored detection >= floor. Returns (dets, frame, drops)."""
    frame = geo.LocalFrame(sum(p.lat for p in panos) / len(panos),
                           sum(p.lng for p in panos) / len(panos))
    drops = {'below_floor': 0, 'on_rig': 0, 'horizon': 0, 'out_of_range': 0}
    dets = []
    s2 = params.sigma_scale ** 2
    for p in panos:
        pose = pano_pose(p, params.apply_pose)
        errors = geo.error_model_for(p.source, params.sigma_peak_px)
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


def fuse(panos, params, allow_mixed_decode=False, allow_mixed_border=False):
    """Associate projected detections into sites. Deterministic: input order never
    matters (canonical sort; best-candidate tiebreak on (D2, site id)).

    Panos written under different peak decodes (#111) are refused (ValueError) unless
    allow_mixed_decode: a caller that assembles panos from two files gets the same guard
    load_results applies to one. The same for border rules (#130) and allow_mixed_border."""
    single_decode(Counter(p.decode for p in panos), 'the panos to fuse', allow_mixed_decode)
    single_border(Counter(p.border for p in panos), 'the panos to fuse', allow_mixed_border)
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

    # Why a detection opened a new site instead of joining one (#111: the peak sigma feeds
    # the gate and the residual). chi2_gate: some site passed the cap, the cannot-link and
    # the vintage window, and none passed the gate. residual: the best site passed the gate
    # but the refit residual would have exceeded residual_per_dof_max.
    rejections = {'chi2_gate': 0, 'residual': 0}
    for det in dets:
        best = None
        gated = False
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
            gated = True
            if d2 <= params.gate_chi2 and (best is None or (d2, site.id) < best[:2]):
                best = (d2, site.id, site)

        if best is None:
            rejections['chi2_gate'] += gated
            new_site(det)
            continue
        site = best[2]
        if det.operational or site.n_operational == 0:
            # refit merge — unless it would blow up the triangulation residual
            if site.n_refit >= 1 and \
                    site.tentative_residual_per_dof(det) > params.residual_per_dof_max:
                rejections['residual'] += 1
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
             'rejections': rejections,
             'camera_heights': camera_height_counts(panos, params),
             'pose': pose_counts(panos, params),
             'frame_origin': {'lat0': frame.lat0, 'lng0': frame.lng0}}
    return sites, frame, stats


def camera_height_counts(panos, params):
    """How the run's panos got their camera height -- the provenance a reader of
    sites_meta.json needs to know which frame the positions are in."""
    if params.camera_height_m == geo.PER_RIG:
        prov = next((p.height_table for p in panos if p.height_table), None) or {}
        groups = {}
        for p in panos:
            if p.camera_height_m is not None:
                groups[p.height_group] = groups.get(p.height_group, 0) + 1
        applied = sum(groups.values())
        return {'mode': geo.PER_RIG, 'table': prov.get('table'), 'sha256': prov.get('sha256'),
                'grain': prov.get('grain'), 'applied': applied,
                'fallback': len(panos) - applied, 'applied_by_group': groups}
    if params.camera_height_m != geo.PER_PANO:
        return {'fixed_m': params.camera_height_m, 'panos': len(panos)}
    measured, flagged = 0, {}
    for p in panos:
        if p.camera_height_m is None:
            continue
        measured += 1
        _, _, reason = depthlib.believe_height(
            p.camera_height_m, p.camera_height_spread_m, p.ground_tilt_deg,
            vintage_median_m=p.camera_height_vintage_m)
        if reason != depthlib.MEASURED:
            flagged[reason] = flagged.get(reason, 0) + 1
    # measured = raycast at its own height; flagged_qc = the subset of those the #44 QC
    # gate flags (kept all the same); fallback = no measurement, raycast at the default.
    out = {'measured': measured, 'flagged_qc': flagged,
           'fallback': len(panos) - measured}
    # Which spread the per-pano sigma is: an index is refused unless it is #47's
    # (load_depth_index), so any height read from one carries the measured-planes spread.
    # Blocks written by sources/gsv.py before #47 carry the old one and are not counted.
    from_index = sum(1 for p in panos if p.height_from_index and p.camera_height_m is not None)
    if from_index:
        out['spread_definition'] = depthlib.SPREAD_DEFINITION
        out['measured_from_index'] = from_index
    return out


def height_table_path(camera_height, results_path, height_table=None):
    """The camera_heights.json a PER_RIG load reads (default: beside results.jsonl), or
    None for any other mode. Raises ValueError when per-rig has no table to read."""
    if camera_height != geo.PER_RIG:
        return None
    table = Path(height_table) if height_table else Path(results_path).parent / HEIGHT_TABLE_NAME
    if not table.exists():
        raise ValueError(f'per-rig needs a camera-height table; none at {table} '
                         '(scripts/mapillary_height.py writes it)')
    return table


def load_at_height(results_path, camera_height, *, depth_index=None, height_table=None,
                   read_heights=None, resolve_auto=True, grade_source=GRADE_SFM,
                   grades_path=None, allow_mixed_decode=False, allow_mixed_border=False):
    """Load results.jsonl and resolve a --camera-height-m value in ONE place (#56).

    fuse_sites' CLI, eval_sites, eval_ps_clustering and mined_precision all load through
    here, so `per-pano`, `per-rig` and `auto` cannot mean different things in different
    scripts. `camera_height` is a number, geo.PER_PANO, geo.PER_RIG or HEIGHT_AUTO:

    - a number: every pano at it (the index is not read unless read_heights=True);
    - per-pano: the pano block's measured height, else depth/index.csv beside the file
      (or `depth_index`); unmeasured panos fall back to geo.DEFAULT_CAMERA_HEIGHT_M;
    - per-rig: camera_heights.json beside the file (or `height_table`);
    - auto: per-pano's heights, then resolve_auto_height (per-rig by capture year for GSV,
      the constant otherwise). resolve_auto=False keeps the heights unresolved and
      returns the constant, which is what --pose-ablation / --implied-height need.

    Returns (panos, skipped, height, auto): `height` is what FuseParams.camera_height_m
    takes, `auto` is resolve_auto_height's provenance (None unless auto resolved). Raises
    ValueError on a missing or mismatched height table (callers turn it into an exit).

    Example:
        >>> panos, skipped, height, auto = load_at_height('runs/richmond/results.jsonl',
        ...                                               HEIGHT_AUTO)  # doctest: +SKIP
        >>> height, auto['resolved'], auto['reason']                    # doctest: +SKIP
        (2.6, 2.6, 'no GSV panos')
    """
    if read_heights is None:
        read_heights = camera_height in (geo.PER_PANO, HEIGHT_AUTO)
    table = height_table_path(camera_height, results_path, height_table)
    panos, skipped = load_results(results_path, depth_index, read_heights=read_heights,
                                  height_table=table, grade_source=grade_source,
                                  grades_path=grades_path,
                                  allow_mixed_decode=allow_mixed_decode,
                                  allow_mixed_border=allow_mixed_border)
    if camera_height != HEIGHT_AUTO:
        return panos, skipped, camera_height, None
    if not resolve_auto:
        return panos, skipped, geo.DEFAULT_CAMERA_HEIGHT_M, None
    panos, height, auto = resolve_auto_height(panos)
    return panos, skipped, height, auto


def resolved_height_counts(panos, params, auto=None):
    """camera_height_counts with an `auto` resolution folded in: auto's provenance over
    the per-rig block it resolved to, minus that block's camera_heights.json fields,
    which a GSV auto fuse never has. What sites_meta.json and the eval reports record."""
    counts = camera_height_counts(panos, params)
    if auto is None:
        return counts
    counts = {k: v for k, v in counts.items() if k not in ('mode', 'table', 'sha256', 'grain')}
    return {**auto, **counts}


def describe_auto(auto):
    """One line for how `auto` resolved, e.g. 'auto -> 2.6 (no GSV panos)'."""
    return (f"auto -> {auto['resolved']}"
            + (f" ({auto['reason']})" if 'reason' in auto else ''))


def height_resolution_lines(requested, panos, params, auto=None):
    """Markdown bullets stating how a report's camera height resolved (#56): the mode,
    how many panos took a measured/table height vs fell back to the constant, the per-rig
    year table under auto, and the spread definition when an index supplied heights.
    Empty for a numeric request, so a constant-height report is unchanged.

    Example:
        >>> class P: camera_height_m = 2.6
        >>> height_resolution_lines(2.6, [], P())
        []
    """
    if not isinstance(requested, str):
        return []
    c = resolved_height_counts(panos, params, auto)
    n = len(panos)
    d = geo.DEFAULT_CAMERA_HEIGHT_M
    lines = [f'- camera height mode `{requested}`'
             + (f": {describe_auto(auto)}" if auto is not None else '')]
    if 'fixed_m' in c:          # auto resolved to the constant
        lines.append(f'- all {n} panos raycast at the constant {c["fixed_m"]:g} m '
                     '(no pano took a measured or per-rig height)')
    else:
        used = n - c['fallback']
        what = 'a measured height' if params.camera_height_m == geo.PER_PANO \
            else 'a per-rig height'
        lines.append(f'- {used} of {n} panos took {what}; {c["fallback"]} fell back to '
                     f'the {d:g} m constant'
                     + (' -- ALL of them, so this frame equals the constant one'
                        if used == 0 else ''))
        if c.get('flagged_qc'):
            lines.append(f"- flagged by the #44 QC gate (kept all the same): "
                         f"{c['flagged_qc']}")
        if c.get('applied_by_group') and auto is None:   # auto: the year table below
            lines.append('- per-rig groups (panos): ' + ', '.join(
                f'{g} {k}' for g, k in sorted(c['applied_by_group'].items())))
        if c.get('spread_definition'):
            lines.append(f"- {c['measured_from_index']} heights read from depth/index.csv; "
                         f"spread definition: {c['spread_definition']}")
    for year, a in (c.get('assignment') or {}).items():
        med = 'n/a' if a['median_m'] is None else f"{a['median_m']:.2f} m"
        lines.append(f"  - {year}: median {med}, {a['measured']} of {a['dated']} measured "
                     f"-> {a['height_m']:g} m")
    return lines


def frame_suffix(camera_height):
    """Output-directory suffix for a scoring frame, so a report in one frame never
    overwrites another: '' for the default constant, '_h<metres, 2 dp>' for any other
    number, '_<mode>' for a named mode.

    Example:
        >>> [frame_suffix(h) for h in (2.6, 2.341219672825709, 'per-pano', 'auto')]
        ['', '_h2.34', '_per-pano', '_auto']
    """
    if isinstance(camera_height, str):
        return f'_{camera_height}'
    return '' if camera_height == geo.DEFAULT_CAMERA_HEIGHT_M else f'_h{camera_height:.2f}'


def frame_label(camera_height):
    """'2.6 m' for a number, the mode name for a named mode -- for report text."""
    return camera_height if isinstance(camera_height, str) else f'{camera_height:g} m'


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
    """argparse type for --camera-height-m: meters, geo.PER_PANO or geo.PER_RIG."""
    return value if value in (geo.PER_PANO, geo.PER_RIG) else float(value)


def fuse_camera_height_arg(value):
    """fuse_sites' own --camera-height-m: camera_height_arg plus HEIGHT_AUTO (#79). Also
    the type of eval_ps_clustering's --camera-height-m and mined_precision's
    --camera-height, which accept `auto` explicitly (#56); eval_sites and the other
    analysis scripts keep camera_height_arg, so `auto` never reaches them by accident.

    Example:
        >>> [fuse_camera_height_arg(v) for v in ('2.6', 'auto', 'per-pano', 'per-rig')]
        [2.6, 'auto', 'per-pano', 'per-rig']
    """
    return value if value == HEIGHT_AUTO else camera_height_arg(value)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', help='run directory (with results.jsonl) or a jsonl path')
    ap.add_argument('--floor', type=float, default=DETECTION_STORAGE_FLOOR)
    ap.add_argument('--allow-mixed-decode', action='store_true',
                    help='fuse a file whose records mix argmax and gaussian peak decodes '
                         '(#111; refused by default: the two are different frames). '
                         'sites_meta.json then records the mix')
    ap.add_argument('--allow-mixed-border', action='store_true',
                    help='fuse a file whose records mix the exclude and keep peak border '
                         'rules (#130; refused by default: keep adds the seam-band peaks '
                         'exclude drops). sites_meta.json then records the mix')
    ap.add_argument('--min-confidence', type=float, default=OPERATIONAL_CONFIDENCE)
    ap.add_argument('--max-range-m', type=float, default=geo.DEFAULT_MAX_RANGE_M)
    ap.add_argument('--gate-chi2', type=float, default=FuseParams.gate_chi2)
    ap.add_argument('--max-match-m', type=float, default=FuseParams.max_match_m)
    ap.add_argument('--residual-per-dof-max', type=float,
                    default=FuseParams.residual_per_dof_max)
    ap.add_argument('--camera-height-m', type=fuse_camera_height_arg, default=HEIGHT_AUTO,
                    help='"auto" (the default, #79): GSV panos at a per-rig height by '
                         'capture year from the run\'s depth-measured heights; undated GSV '
                         'panos, Mapillary/Panoramax, and a GSV run with no well-measured '
                         'year at 2.6 m -- see gsv_rig_assignment. --pose-ablation and '
                         '--implied-height resolve auto to 2.6 m. Or a height in meters '
                         'for every pano (2.6 = the pre-#79 default), or "per-pano" for '
                         "each GSV pano's depth-measured height where it has one (#40; "
                         'opt-in -- see docs/camera-height-study.md), or "per-rig" for a '
                         "Mapillary/Panoramax run's camera_heights.json (#53; opt-in -- "
                         'see docs/mapillary-camera-height.md)')
    ap.add_argument('--height-table', type=Path, default=None,
                    help='camera_heights.json for --camera-height-m per-rig '
                         '(default: beside results.jsonl)')
    ap.add_argument('--sigma-scale', type=float, default=1.0)
    ap.add_argument('--sigma-peak-px', type=float, default=None,
                    help='heatmap-peak 1-sigma in heatmap px (#111; default '
                         f'geo.SIGMA_PEAK_PX_DEFAULT = {geo.SIGMA_PEAK_PX_DEFAULT}). '
                         f'{geo.SIGMA_PEAK_COARSE_CELL_PX:.2f} is the uniform quantization of one '
                         '8-px coarse cell; see docs/heatmap-grid.md')
    # A value is REQUIRED (no nargs='?'): an optional value would swallow the positional
    # run directory in `--apply-pose runs/x`, and a bare flag would have to guess a mode.
    ap.add_argument('--apply-pose', choices=POSE_MODES, default=FuseParams.apply_pose,
                    metavar='{off,auto,gravity,road}',
                    help='rotate rays by camera pose: auto (the default; currently off for '
                         'every source -- the #42 shuffled-grade control withheld road for '
                         'Mapillary, see AUTO_ROAD_SOURCES), off (flat raycast), gravity '
                         "(stored pitch/roll), or road (minus the sequence's road grade). "
                         'Measured to hurt on GSV -- see the --pose-ablation report')
    ap.add_argument('--grade-source', choices=GRADE_SOURCES, default=GRADE_SFM,
                    help='where --apply-pose road takes the road grade from: sfm (the '
                         "default; the sequence's SfM altitude), sfm-smoothed or dem "
                         '(runs/<name>/dem/grades.csv from scripts/dem_grade.py; #51)')
    ap.add_argument('--grades', type=Path, default=None,
                    help='grades.csv for --grade-source (default: <run>/dem/grades.csv)')
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
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    src = Path(args.run)
    jsonl = src if src.is_file() else src / 'results.jsonl'
    if not jsonl.exists():
        sys.exit(f'no results.jsonl at {jsonl}')
    params = FuseParams(
        floor=args.floor, min_confidence=args.min_confidence,
        max_range_m=args.max_range_m, gate_chi2=args.gate_chi2,
        max_match_m=args.max_match_m,
        residual_per_dof_max=args.residual_per_dof_max,
        camera_height_m=(geo.DEFAULT_CAMERA_HEIGHT_M if args.camera_height_m == HEIGHT_AUTO
                         else args.camera_height_m),
        apply_pose=args.apply_pose,
        sigma_scale=args.sigma_scale, sigma_peak_px=args.sigma_peak_px,
        grade_source=args.grade_source)

    if args.height_table is not None and args.camera_height_m != geo.PER_RIG:
        print('WARNING: --height-table is ignored unless --camera-height-m per-rig',
              file=sys.stderr)
    # --implied-height compares the measured heights with the imagery's, so it keeps
    # them; --pose-ablation reproduces #27's experiment, run at 2.6 m (params already
    # holds it). Only a fuse resolves auto.
    resolve = not (args.implied_height or args.pose_ablation)
    try:
        panos, skipped, resolved, auto = load_at_height(
            jsonl, args.camera_height_m, depth_index=args.depth_index,
            height_table=args.height_table,
            read_heights=(args.camera_height_m in (geo.PER_PANO, HEIGHT_AUTO)
                          or args.implied_height),
            resolve_auto=resolve,
            grade_source=args.grade_source, grades_path=args.grades,
            allow_mixed_decode=args.allow_mixed_decode,
            allow_mixed_border=args.allow_mixed_border)
    except ValueError as e:
        sys.exit(str(e))
    if skipped:
        print(f'skipped {skipped} records without position/heading')
    if not panos:
        sys.exit('no usable records')
    for warning in pose_source_warnings(panos, args.apply_pose):
        print(warning, file=sys.stderr)
    if auto is not None:
        params = replace(params, camera_height_m=resolved)
        print(f'camera height: {describe_auto(auto)}',
              file=sys.stderr if 'reason' in auto and auto['gsv_panos'] else sys.stdout)

    if args.pose_ablation:
        print(pose_ablation_report(panos, params, args.allow_mixed_decode,
                                   args.allow_mixed_border))
        return
    if args.implied_height:
        print(implied_height_report(panos, params, args.allow_mixed_decode,
                                    args.allow_mixed_border))
        return

    sites, frame, stats = fuse(panos, params, allow_mixed_decode=args.allow_mixed_decode,
                               allow_mixed_border=args.allow_mixed_border)
    if auto is not None:
        stats['camera_heights'] = resolved_height_counts(panos, params, auto)
    # Which frame the sites are in (#111): the decode every member's record was written under.
    decodes = Counter(p.decode for p in panos)
    stats['detection_decode'] = (single_decode(decodes, jsonl.name) if len(decodes) == 1
                                 else dict(sorted(decodes.items())))
    # ...and whether those records kept the seam-band peaks (#130).
    borders = Counter(p.border for p in panos)
    stats['detection_border'] = (single_border(borders, jsonl.name) if len(borders) == 1
                                 else dict(sorted(borders.items())))
    out = args.out or jsonl.parent / 'sites.jsonl'
    meta = out.with_name(out.stem + '_meta.json')
    write_sites(sites, frame, stats, params, out, meta)
    print(summarize(sites, stats))
    print(f'wrote {out} and {meta}')


def pose_ablation_report(panos, params, allow_mixed_decode=False, allow_mixed_border=False):
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

    sites, frame, _ = fuse(panos, replace(params, apply_pose=POSE_OFF),
                           allow_mixed_decode=allow_mixed_decode,
                           allow_mixed_border=allow_mixed_border)
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
                    max_range_m=math.inf,
                    errors=geo.error_model_for(p.source, params.sigma_peak_px))
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


def implied_height_report(panos, params, allow_mixed_decode=False, allow_mixed_border=False):
    """Measured vs imagery-implied camera height, by capture year (#40).

    The implied height is independent of the height model only per pair; which pairs
    exist is not, because association ran under params.camera_height_m, and implied
    heights drift toward whatever height the association used (paterson's 2025 rig reads
    1.93 m associated at 1.8, 2.14 m at 2.6). So read it as a fixed-point search: re-run
    with --camera-height-m set near the implied value until the two agree. With
    `per-pano`, iterate on a scale instead (docs/camera-height-study.md does, by
    monkeypatching; the self-consistent scale was 1.06-1.16 by city).
    """
    sites, frame, _ = fuse(panos, params, allow_mixed_decode=allow_mixed_decode,
                           allow_mixed_border=allow_mixed_border)
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
