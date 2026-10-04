"""Figures for the addendum of docs/gsv-partial-pose-study.md (#116 follow-up).

Four figures (and the data behind one markdown table), each answering one question:

    fig5_consistency    Does production's `--apply-pose partial` (frozen pooled constants)
                        place members like the study's fitted arms? (Paterson TEST half,
                        site-bootstrap CIs)
    fig6_loss_bar       What does the pool-sized recall clause demand, and how often does it
                        fail -- against a real loss, at the null, and under churn -- next to
                        the discordant-pairs sign test?
    fig7_site_examples  What does a fraction of the tilt do to real multi-view sites, and what
                        does the full pose do on the most-tilted ones?
    fig9_vintage_k      EXPLORATORY: does the leaked fraction differ by capture vintage?
    (Table 8 is a markdown table in the doc; its data is addendum_store_gate_laurens_gsv.json.)

Usage (no GPU, no network; needs matplotlib and numpy, which are analysis-only and
deliberately not in requirements.txt -- `pip install matplotlib`):
    python scripts/gsv_partial_pose_figures.py              # redraw from committed data/
    python scripts/gsv_partial_pose_figures.py --refresh    # recompute data/ first
The #121 figures (fig1-fig4) stay with `gsv_partial_pose.py figures`; this script draws only
the addendum figures, so the pinned confirm code in gsv_partial_pose.py is untouched.

Every figure is drawn from CSV/JSON files under docs/figures/gsv-partial-pose/data/
(prefix `addendum_`), so a redraw is byte-reproducible for a given matplotlib / FreeType /
platform: PNG at 200 dpi and SVG (date stripped, element ids salted with a constant, LF
line endings). `--refresh` rebuilds that data: the consistency, vintage and gate files are
copied from committed run outputs; the loss-bar table and the power simulations are
recomputed (seed 116); the site examples, site candidates and the consistency bootstrap
are re-fused from runs/paterson/{results.jsonl,depth/index.csv} (the only step that needs
run inputs, which are not committed).
"""
import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import gsv_partial_pose as gpp  # noqa: E402

FIG_DIR = gpp.FIG_DIR
DATA = gpp.FIG_DATA
SEED = gpp.SEED
SIM_REPS = 20_000                     # per (scenario, n) point of the power curves
SIM_N = tuple(range(50, 801, 10))
# (label, p_lost, p_gained): what each curve is
SIM_SCENARIOS = (('null: 1% lost, 0% gained', 0.01, 0.0),
                 ('2% lost, 0% gained', 0.02, 0.0),
                 ('3% lost, 0% gained', 0.03, 0.0),
                 ('churn: 3% lost, 2% gained', 0.03, 0.02),
                 ('churn: 2% lost, 2% gained', 0.02, 0.02))
SITE_EXAMPLES = 3
SITE_MIN_VIEWS = 4
BOOT_B = 1000                          # site-bootstrap replicates for fig5's CIs
CONSISTENCY_ARMS = (gpp.ARM_OFF, gpp.ARM_SHUFFLED, gpp.ARM_PARTIAL, gpp.ARM_LOCO,
                    gpp.ARM_POOLED, gpp.ARM_MIRROR)
SITE_ARMS = (gpp.ARM_OFF, gpp.ARM_POOLED, gpp.ARM_FULL, gpp.ARM_MIRROR)

# One visual system with the #121 figures (gpp._style; dataviz reference palette, fixed
# slots). Arm colours follow the arm across every figure; partial-pooled takes the next
# unused slot (green). The adjacent order used in fig5 was run through the dataviz
# validator (passes CVD and normal-vision floors; magenta < 3:1 gets direct labels).
# Shuffled arms keep orange/yellow; the churn scenarios of fig6 therefore take the one
# remaining slot (red), told apart by line style.
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
COLORS = {**gpp.FIG_COLORS, gpp.ARM_POOLED: '#008300'}
MARKERS = {**gpp.FIG_MARKERS, gpp.ARM_POOLED: 'P'}
RING_STYLE = {gpp.ARM_OFF: ':', gpp.ARM_POOLED: '--', gpp.ARM_FULL: '-', gpp.ARM_MIRROR: '-.'}
SCENARIO_STYLE = {SIM_SCENARIOS[0][0]: ('#86b6ef', '-'), SIM_SCENARIOS[1][0]: ('#2a78d6', '-'),
                  SIM_SCENARIOS[2][0]: ('#184f95', '-'), SIM_SCENARIOS[3][0]: ('#e34948', '-'),
                  SIM_SCENARIOS[4][0]: ('#e34948', '--')}
LABEL = {gpp.ARM_OFF: 'off (flat, production default)',
         gpp.ARM_SHUFFLED: 'partial-shuffled (control)',
         gpp.ARM_PARTIAL: 'partial (city fit)', gpp.ARM_LOCO: 'partial-loco',
         gpp.ARM_POOLED: 'partial-pooled (production)', gpp.ARM_MIRROR: 'partial-mirror',
         gpp.ARM_FULL: 'full pose'}
FOOT = 10                             # smallest font: legible when the PNG is shown 1200 px wide


def _np():
    try:
        import numpy as np
    except ImportError:
        raise SystemExit('this analysis script needs numpy (pip install numpy)')
    return np


# --------------------------------------------------------------------------- data

def _write_csv(path, rows):
    gpp.write_csv(path, rows)


def recall_bar_rows():
    """k*(n) and the fixed 1.0-pt rule it replaces, as the smallest net loss that FAILS."""
    return [{'n': n, 'k_star': gpp.recall_loss_bar(n),
             'fixed_1pt_fail_at': math.floor(0.01 * n) + 1} for n in range(50, 801)]


def sign_test_critical(max_m, alpha=0.05):
    """c[m]: smallest lost count with P(Binomial(m, 0.5) >= c) <= alpha (m + 1 if none)."""
    out = []
    for m in range(max_m + 1):
        tail, c = 0.0, m + 1
        for k in range(m, -1, -1):
            tail += math.comb(m, k) / 2 ** m
            if tail > alpha:
                break
            c = k
        out.append(c)
    return out


def power_rows(reps=SIM_REPS, seed=SEED):
    """P(FAIL) of clause (ii) as frozen (L >= k*(n)) and of the discordant-pairs sign test
    (lost significantly more than half of lost + gained, one-sided 0.05), per scenario
    and n. One numpy generator, fixed iteration order, so the table is reproducible."""
    np = _np()
    rng = np.random.default_rng(seed)
    crit = np.array(sign_test_critical(800))
    rows = []
    for label, pl, pg in SIM_SCENARIOS:
        for n in SIM_N:
            d = rng.multinomial(n, [pl, pg, 1.0 - pl - pg], size=reps)
            lost, gained = d[:, 0], d[:, 1]
            rule = ((lost - gained) >= gpp.recall_loss_bar(n)).mean()
            sign = (lost >= crit[lost + gained]).mean()
            rows.append({'scenario': label, 'p_lost': pl, 'p_gained': pg, 'n': n,
                         'reps': reps, 'p_fail_rule': float(rule), 'p_fail_sign': float(sign)})
    return rows


POWER_TABLE_CELLS = ((157, 0.02, 0.0), (300, 0.02, 0.0), (157, 0.01, 0.0), (300, 0.03, 0.02),
                     (157, 0.03, 0.02))


def power_table_rows(reps=200_000, seed=SEED):
    """The addendum's five-cell power/size table: exactly the doc's snippet (one generator,
    seed 116, the cells drawn in this order), so the committed CSV is the quoted numbers."""
    np = _np()
    rng = np.random.default_rng(seed)
    rows = []
    for n, pl, pg in POWER_TABLE_CELLS:
        d = rng.multinomial(n, [pl, pg, 1 - pl - pg], size=reps)
        rows.append({'n': n, 'p_lost': pl, 'p_gained': pg, 'reps': reps,
                     'k_star': gpp.recall_loss_bar(n),
                     'p_fail_rule': float(((d[:, 0] - d[:, 1]) >= gpp.recall_loss_bar(n)).mean())})
    return rows


def _site_arms(runs_root, seed=SEED):
    """Paterson TEST half at auto: (test panos, {arm: (panos, params)}) for off /
    partial-pooled (production) / full / partial-mirror (pooled constants, signs flipped)."""
    panos, height = gpp.load_city('paterson', fs.HEIGHT_AUTO, runs_root)
    _train, test = gpp.split_halves(panos, seed)
    k = geo.PARTIAL_POSE_K_GSV
    base = fs.FuseParams(min_confidence=gpp.OPERATIONAL_CONFIDENCE, camera_height_m=height)
    grav = gpp.replace(base, apply_pose=fs.POSE_GRAVITY)
    return test, {gpp.ARM_OFF: (test, gpp.replace(base, apply_pose=fs.POSE_OFF)),
                  gpp.ARM_POOLED: (test, gpp.replace(base, apply_pose=fs.POSE_PARTIAL)),
                  gpp.ARM_FULL: (gpp.posed_panos(test, 1.0, 1.0), grav),
                  gpp.ARM_MIRROR: (gpp.posed_panos(test, -k[0], -k[1]), grav)}


def site_rows(runs_root, n_sites=SITE_EXAMPLES, seed=SEED):
    """(candidate rows, example rows). Association is frozen from off. A CANDIDATE is a site
    with >= SITE_MIN_VIEWS operational members on distinct panos (one row each: mean member
    |tilt| and views placed per arm). Two example sets, n_sites each, by largest mean |tilt|
    (ties: seeded shuffle, seed 116):
      every_arm    -- among candidates every arm places in full (the #121-era figure 7);
      most_tilted  -- among ALL candidates, whatever the arms place.
    One example row per member: camera, |tilt|, and the ground point per arm (blank when
    that arm cannot place it inside 25 m)."""
    test, arms = _site_arms(runs_root, seed)
    sites, frame, _ = fs.fuse(*arms[gpp.ARM_OFF])
    place = gpp.make_placer(arms)
    by_id = {p.pano_id: p for p in test}
    cands = []
    for s in sites:
        ms = [d for d, _ in s.members if d.operational]
        if len(ms) < SITE_MIN_VIEWS or len({d.pano_id for d in ms}) != len(ms):
            continue
        pts = {}
        for arm in arms:
            gs = [place(arm, d.pano_id, d.x, d.y) for d in ms]
            pts[arm] = [None if g is None else frame.to_enu(g.lat, g.lng) for g in gs]
        tilts = [math.hypot(*gpp.stored_tilt(by_id[d.pano_id])) for d in ms]
        cands.append((sum(tilts) / len(tilts), s, ms, pts, tilts))
    random.Random(seed).shuffle(cands)
    cands.sort(key=lambda c: -c[0])
    cand_rows = [{'site_id': s.id, 'n_views': len(ms), 'mean_tilt_deg': mt,
                  **{f'placed_{a}': sum(p is not None for p in pts[a]) for a in arms},
                  'every_arm_places_all': all(p is not None for a in arms for p in pts[a])}
                 for mt, s, ms, pts, _t in cands]
    every = [c for c in cands if all(p is not None for a in arms for p in c[3][a])]
    rows = []
    for which, chosen in (('every_arm', every[:n_sites]), ('most_tilted', cands[:n_sites])):
        for rank, (mean_tilt, s, ms, pts, tilts) in enumerate(chosen, 1):
            for i, d in enumerate(ms):
                p = by_id[d.pano_id]
                ce, cn = frame.to_enu(p.lat, p.lng)
                pitch, roll = gpp.stored_tilt(p)
                row = {'set': which, 'rank': rank, 'site_id': s.id,
                       'candidates': len(every) if which == 'every_arm' else len(cands),
                       'site_mean_tilt_deg': mean_tilt, 'pano_id': d.pano_id,
                       'det_index': d.det_index, 'cam_e': ce, 'cam_n': cn,
                       'pitch_deg': pitch, 'roll_deg': roll, 'tilt_deg': tilts[i]}
                for arm in arms:
                    pt = pts[arm][i]
                    row[f'{arm}_e'], row[f'{arm}_n'] = (None, None) if pt is None else pt
                rows.append(row)
    return cand_rows, rows


def _nearest_rank(np, v, q):
    v = np.sort(v)
    return float(v[min(len(v) - 1, int(round((len(v) - 1) * q)))])


def consistency_bootstrap_rows(runs_root, B=BOOT_B, seed=SEED):
    """Site-bootstrap CIs (percentile, 95%) for fig5: each arm's median and p90 within-site
    pair distance minus off's, and the three pairwise contrasts among the partial arms, on
    Paterson's TEST half at both heights. Pairs are #116's common pairs (co-associated by
    every decision arm), each assigned to its off site; a replicate resamples off sites with
    replacement (one draw shared by every arm, so the differences are paired) and recomputes
    each statistic over the pairs that arm places -- the definition of the committed rows,
    which the point estimates reproduce (asserted)."""
    np = _np()
    table = gpp.read_csv(gpp.POOLED_DIR / 'coefficients.csv')
    committed = {(r['height'], r['arm']): r
                 for r in gpp.read_csv(REPO_ROOT / 'runs' / 'paterson' / gpp.OUT_NAME
                                       / 'consistency_pairs.csv')
                 if r['frame'] == gpp.FRAME_REASSOC}
    rng = np.random.default_rng(seed)
    contrasts = [(a, gpp.ARM_OFF) for a in CONSISTENCY_ARMS if a != gpp.ARM_OFF] + [
        (gpp.ARM_POOLED, gpp.ARM_PARTIAL), (gpp.ARM_POOLED, gpp.ARM_LOCO),
        (gpp.ARM_PARTIAL, gpp.ARM_LOCO)]
    rows = []
    for height in gpp.HEIGHTS:
        label = gpp.height_label(height)
        panos, fuse_height = gpp.load_city('paterson', height, runs_root)
        _train, test = gpp.split_halves(panos, seed)
        coefs = {gpp.ARM_PARTIAL: gpp.coef_pair(table, label, 'city', 'paterson'),
                 gpp.ARM_LOCO: gpp.coef_pair(table, label, 'loco', 'paterson')}
        arms, _own = gpp.build_arms(test, coefs, fuse_height, seed=seed)
        arms[gpp.ARM_POOLED] = (test, gpp.replace(arms[gpp.ARM_OFF][1],
                                                  apply_pose=fs.POSE_PARTIAL))
        fused = gpp._fuse_arms(arms)
        common = sorted(set.intersection(*(gpp.site_pairs(fused[a][0])
                                           for a in gpp.DECISION_ARMS)))
        site_of = {}
        for s in fused[gpp.ARM_OFF][0]:
            for d, _ in s.members:
                site_of[(d.pano_id, d.det_index)] = s.id
        pair_site = np.array([site_of[pr[0]] for pr in common])
        uniq, site_idx = np.unique(pair_site, return_inverse=True)
        by_pano = {p.pano_id: p for p in test}
        place = gpp.make_placer(arms)
        frame = fused[gpp.ARM_OFF][1]
        dist = {}
        for arm in CONSISTENCY_ARMS:
            vals, pos = [], {}
            for pr in common:
                pts = []
                for key in pr:
                    if key not in pos:
                        _i, x, y, _c = by_pano[key[0]].detections[key[1]]
                        g = place(arm, key[0], x, y)
                        pos[key] = None if g is None else frame.to_enu(g.lat, g.lng)
                    pts.append(pos[key])
                vals.append(np.nan if None in pts else
                            math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))
            dist[arm] = np.array(vals)
        for arm in CONSISTENCY_ARMS:          # point estimates == the committed rows
            v = dist[arm][~np.isnan(dist[arm])]
            for stat, q in (('median_m', 0.5), ('p90_m', 0.9)):
                assert abs(_nearest_rank(np, v, q) - float(committed[label, arm][stat])) < 6e-5, \
                    (label, arm, stat)
        n_sites = len(uniq)
        reps = {c: {'median_m': [], 'p90_m': []} for c in contrasts}
        for _b in range(B):
            w = np.bincount(rng.integers(0, n_sites, n_sites), minlength=n_sites)[site_idx]
            stats = {}
            for arm in CONSISTENCY_ARMS:
                ok = ~np.isnan(dist[arm]) & (w > 0)
                v = np.repeat(dist[arm][ok], w[ok])
                stats[arm] = {'median_m': _nearest_rank(np, v, 0.5),
                              'p90_m': _nearest_rank(np, v, 0.9)}
            for a, b in contrasts:
                for stat in ('median_m', 'p90_m'):
                    reps[a, b][stat].append(stats[a][stat] - stats[b][stat])
        for a, b in contrasts:
            for stat in ('median_m', 'p90_m'):
                point = float(committed[label, a][stat]) - float(committed[label, b][stat])
                r = np.array(reps[a, b][stat])
                rows.append({'height': label, 'arm': a, 'minus': b, 'stat': stat,
                             'point': point, 'ci_lo': float(np.percentile(r, 2.5)),
                             'ci_hi': float(np.percentile(r, 97.5)), 'B': B,
                             'n_sites': n_sites, 'n_pairs': len(common)})
    return rows


def refresh(runs_root):
    DATA.mkdir(parents=True, exist_ok=True)
    runs = REPO_ROOT / 'runs'
    shutil.copyfile(runs / 'paterson' / gpp.OUT_NAME / 'consistency_pairs.csv',
                    DATA / 'addendum_consistency_pairs.csv')
    dry = runs / 'laurens_gsv' / gpp.OUT_NAME / 'confirm_exploratory_auto'
    shutil.copyfile(dry / 'vintage_k.csv', DATA / 'addendum_vintage_k_laurens_gsv.csv')
    shutil.copyfile(dry / 'store_pose_gate.json', DATA / 'addendum_store_gate_laurens_gsv.json')
    _write_csv(DATA / 'addendum_loss_bar.csv', recall_bar_rows())
    _write_csv(DATA / 'addendum_power.csv', power_rows())
    _write_csv(DATA / 'addendum_power_table.csv', power_table_rows())
    cands, examples = site_rows(runs_root)
    _write_csv(DATA / 'addendum_site_candidates.csv', cands)
    _write_csv(DATA / 'addendum_site_examples.csv', examples)
    _write_csv(DATA / 'addendum_consistency_bootstrap.csv', consistency_bootstrap_rows(runs_root))


# ------------------------------------------------------------------------ figures

def _setup():
    try:
        import matplotlib
    except ImportError:
        raise SystemExit('the figures need matplotlib, which is analysis-only and '
                         'deliberately not in requirements.txt: pip install matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gpp._style(plt)
    plt.rcParams.update({'font.size': 10.5, 'svg.hashsalt': 'gsv-partial-pose-116',
                         'axes.titlesize': 11.5, 'axes.titleweight': 'bold',
                         'axes.titlelocation': 'left', 'legend.frameon': False})
    return plt


def _save(fig, name):
    fig.savefig(FIG_DIR / f'{name}.png', dpi=200, metadata={'Software': None})
    # SVG is text: render to memory and write LF line endings, so a Windows render is
    # byte-identical to the committed (LF) file and to a Linux/macOS render.
    import io
    buf = io.StringIO()
    fig.savefig(buf, format='svg', metadata={'Date': None, 'Creator': None})
    (FIG_DIR / f'{name}.svg').write_bytes(buf.getvalue().replace('\r\n', '\n').encode('utf-8'))


def _f(r, k):
    return float(r[k])


def fig5_consistency(plt, rows, boot):
    """Change from off with site-bootstrap 95% CIs; the partial arms' mutual agreement is
    stated as a contrast with its own CI."""
    order = (gpp.ARM_SHUFFLED, gpp.ARM_PARTIAL, gpp.ARM_LOCO, gpp.ARM_POOLED, gpp.ARM_MIRROR)
    by = {(r['height'], r['arm']): r for r in rows if r['frame'] == gpp.FRAME_REASSOC}
    ci = {(r['height'], r['arm'], r['minus'], r['stat']): r for r in boot}
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), sharey=True)
    gap = len(order) + 2.2
    ys, labels = [], []
    for gi, h in enumerate(('auto', '2.6')):
        n_off = int(by[h, gpp.ARM_OFF]['pairs_scored'])
        for ai, arm in enumerate(order):
            ys.append(gi * gap + ai)
            n = int(by[h, arm]['pairs_scored'])
            labels.append(LABEL[arm] + (f'\n[{n:,} of {n_off:,} pairs placed]' if n != n_off else ''))
    for ax, stat, name in ((axes[0], 'median_m', 'Median'), (axes[1], 'p90_m', 'p90')):
        for gi, h in enumerate(('auto', '2.6')):
            off = _f(by[h, gpp.ARM_OFF], stat)
            y0 = gi * gap
            for ai, arm in enumerate(order):
                c = ci[h, arm, gpp.ARM_OFF, stat]
                dv, lo, hi = _f(c, 'point'), _f(c, 'ci_lo'), _f(c, 'ci_hi')
                ax.barh(y0 + ai, dv, height=0.68, color=COLORS[arm], edgecolor=SURFACE,
                        linewidth=2, hatch='///' if arm == gpp.ARM_SHUFFLED else None)
                ax.errorbar(dv, y0 + ai, xerr=[[dv - lo], [hi - dv]], fmt='none', ecolor=INK,
                            elinewidth=1.3, capsize=3)
                right = max(dv, hi, 0)
                ax.annotate(f'{dv:+.3f} [{lo:+.3f}, {hi:+.3f}]', (right, y0 + ai),
                            xytext=(6, 0), textcoords='offset points', va='center',
                            fontsize=9.5, color=INK)
            pc = ci[h, gpp.ARM_POOLED, gpp.ARM_PARTIAL, stat]
            lc = ci[h, gpp.ARM_POOLED, gpp.ARM_LOCO, stat]
            ax.annotate(f'production minus city fit: {_f(pc, "point"):+.4f} '
                        f'[{_f(pc, "ci_lo"):+.3f}, {_f(pc, "ci_hi"):+.3f}]\n'
                        f'production minus LOCO: {_f(lc, "point"):+.4f} '
                        f'[{_f(lc, "ci_lo"):+.3f}, {_f(lc, "ci_hi"):+.3f}]',
                        (0.36, y0 + len(order) - 0.15), xycoords=ax.get_yaxis_transform(),
                        fontsize=9.5, color=INK2, va='top')
            n = int(by[h, gpp.ARM_OFF]['pairs_scored'])
            ax.annotate(f'height {h}{" m" if h == "2.6" else ""}: off = {off:.3f} m '
                        f'({n:,} pairs)', (0.01, y0 - 0.95), xycoords=ax.get_yaxis_transform(),
                        fontsize=10.5, color=INK, weight='bold', ha='left',
                        bbox={'fc': SURFACE, 'ec': 'none', 'pad': 1})
        ax.axvline(0, color=INK, lw=1.2)
        ax.set_title(f'{name} within-site pair distance')
        ax.set_xlabel(f'{name.lower()} minus off (m), 95% site-bootstrap CI\n'
                      '<- tighter | looser ->')
        ax.grid(axis='x')
        ax.set_axisbelow(True)
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, -0.45), max(hi, 0.45) + 0.45)
    axes[0].set_yticks(ys, labels)
    axes[0].set_ylim(gap + len(order) + 1.3, -1.5)
    fig.suptitle("Production's `--apply-pose partial` matches the fitted study arms, and all "
                 'three beat the shuffled control', x=0.01, ha='left', fontsize=12.5,
                 weight='bold')
    fig.text(0.01, 0.012, 'Data: data/addendum_consistency_pairs.csv, '
             f'data/addendum_consistency_bootstrap.csv ({BOOT_B:,} site resamples, seed 116).',
             fontsize=FOOT, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    return fig


def fig6_loss_bar(plt, bar_rows, power):
    np = _np()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8),
                             gridspec_kw={'width_ratios': (1, 1, 1)})
    ax = axes[0]
    n = [int(r['n']) for r in bar_rows]
    ax.step(n, [int(r['k_star']) for r in bar_rows], where='post', color='#2a78d6', lw=2.2,
            label='k*(n): pool-sized bar (frozen)')
    ax.step(n, [int(r['fixed_1pt_fail_at']) for r in bar_rows], where='post', color=MUTED,
            lw=2, ls='--', label='fixed 1.0 pt (#116, replaced)')
    ax.plot(157, 2, 'o', color=INK, ms=9, mec=SURFACE, mew=2, zorder=5)
    ax.annotate('Bend, #116 half split:\nn = 157, L = 2\nfails the 1.0-pt bar (2),\n'
                'passes k*(157) = 5', (157, 2), xytext=(470, 1.6), textcoords='data',
                fontsize=FOOT, color=INK, arrowprops={'arrowstyle': '-', 'color': INK2, 'lw': 1})
    for nn in (100, 300, 500):
        k = gpp.recall_loss_bar(nn)
        ax.annotate(f'k*({nn}) = {k}', (nn, k), xytext=(4, 6), textcoords='offset points',
                    fontsize=FOOT, color=INK2, ha='left', va='bottom')
    ax.set_xlabel('n = off-pool ramps (recall denominator, 2.5 m)')
    ax.set_ylabel('net ramps lost (lost - gained) that FAILS (ii)')
    ax.set_title('A. What the clause demands')
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=FOOT)
    for ax, col, title in ((axes[1], 'p_fail_rule', 'B. Frozen rule: P(FAIL)'),
                           (axes[2], 'p_fail_sign', 'C. Sign-test alternative: P(FAIL)')):
        for label, _pl, _pg in SIM_SCENARIOS:
            rs = [r for r in power if r['scenario'] == label]
            color, ls = SCENARIO_STYLE[label]
            ax.plot([int(r['n']) for r in rs],
                    np.maximum([_f(r, col) for r in rs], 2e-3), color=color, ls=ls, lw=2,
                    label=label.replace('null:', "1.0-pt null:"))
        ax.axhline(0.05, color=INK, lw=1.2, ls=':')
        ax.annotate('alpha = 0.05', (800, 0.05), xytext=(0, 4), textcoords='offset points',
                    ha='right', fontsize=FOOT, color=INK)
        ax.set_yscale('log')
        ax.set_ylim(2e-3, 1.3)
        ax.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                      ['0.005', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1'])
        ax.set_xlabel('n = off-pool ramps')
        ax.set_title(title)
        ax.grid()
        ax.set_axisbelow(True)
    axes[1].set_ylabel('P(clause (ii) FAILS), log scale')
    r160 = next(r for r in power if r['scenario'] == SIM_SCENARIOS[1][0] and r['n'] == '160')
    axes[1].annotate(f'real 2% loss: fails\n{_f(r160, "p_fail_rule"):.0%} at n = 160',
                     (160, _f(r160, 'p_fail_rule')), xytext=(230, 0.0042), textcoords='data',
                     fontsize=FOOT, color=INK,
                     arrowprops={'arrowstyle': '-', 'color': INK2, 'lw': 1})
    axes[2].legend(loc='lower right', fontsize=9.5, title='true per-ramp change vs off',
                   title_fontsize=9.5, bbox_to_anchor=(1.0, 0.07))
    fig.suptitle('The pool-sized recall bar is looser than 1.0 pt at every n, weak against a '
                 'real 2% loss, and fails too often under churn', x=0.01, ha='left',
                 fontsize=12.5, weight='bold')
    fig.text(0.01, 0.012, f'B and C: {SIM_REPS:,} multinomial draws per point, seed 116. Data: '
             'data/addendum_loss_bar.csv, data/addendum_power.csv.', fontsize=FOOT, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def _spread(pts):
    d = sorted(math.hypot(a[0] - b[0], a[1] - b[1])
               for i, a in enumerate(pts) for b in pts[i + 1:])
    return d[len(d) // 2] if len(d) % 2 else (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2


def _pt(r, a):
    return None if r[f'{a}_e'] in ('', None) else (_f(r, f'{a}_e'), _f(r, f'{a}_n'))


def fig7_site_examples(plt, rows, cands):
    from matplotlib.lines import Line2D
    n_cand = len(cands)
    n_every = sum(1 for c in cands if c['every_arm_places_all'] == 'True')
    dropped_full = sum(1 for c in cands if int(c['placed_full']) < int(c['n_views']))
    dropped_pooled = sum(1 for c in cands
                         if int(c[f'placed_{gpp.ARM_POOLED}']) < int(c['n_views']))
    assert dropped_pooled == 0, dropped_pooled
    fig = plt.figure(figsize=(16, 13.4), layout='constrained')
    subs = fig.subfigures(2, 1, hspace=0.03)
    notes = {'every_arm': (f'A. The 3 most-tilted of the {n_every:,} sites every arm places in '
                           'full (the selection the first version of this figure used)'),
             'most_tilted': (f'B. The 3 most-tilted of all {n_cand:,} candidate sites: the full '
                             'pose drops views here (hollow diamonds); partial drops none')}
    for which, sub in zip(('every_arm', 'most_tilted'), subs):
        sub.suptitle(notes[which], x=0.01, ha='left', fontsize=12, weight='bold')
        axes = sub.subplots(1, SITE_EXAMPLES)
        for ci, ax in enumerate(axes):
            rs = [r for r in rows if r['set'] == which and int(r['rank']) == ci + 1]
            pts = {a: [_pt(r, a) for r in rs] for a in SITE_ARMS}
            o = (sum(q[0] for q in pts[gpp.ARM_OFF]) / len(rs),
                 sum(q[1] for q in pts[gpp.ARM_OFF]) / len(rs))
            for r, (ge, gn) in zip(rs, pts[gpp.ARM_OFF]):
                ux, uy = ge - _f(r, 'cam_e'), gn - _f(r, 'cam_n')
                nrm = math.hypot(ux, uy)
                ax.plot([ge - o[0] - 3 * ux / nrm, ge - o[0] + 3 * ux / nrm],
                        [gn - o[1] - 3 * uy / nrm, gn - o[1] + 3 * uy / nrm], color=GRID,
                        lw=1.2, zorder=1)
            lines = []
            for a in SITE_ARMS:
                placed = [q for q in pts[a] if q is not None]
                for q in placed:
                    ax.plot(q[0] - o[0], q[1] - o[1], MARKERS[a], color=COLORS[a], ms=9,
                            mec=SURFACE, mew=1.5, zorder=3)
                if len(placed) == len(rs):
                    cen = (sum(q[0] for q in placed) / len(rs), sum(q[1] for q in placed) / len(rs))
                    ax.add_patch(plt.Circle((cen[0] - o[0], cen[1] - o[1]), 0.35, fill=False,
                                            ec=COLORS[a], lw=2.2, ls=RING_STYLE[a], zorder=2))
                    lines.append(f'{a:<15} {_spread(placed):.2f} m')
                else:
                    for q_off, q in zip(pts[gpp.ARM_OFF], pts[a]):
                        if q is None:
                            ax.plot(q_off[0] - o[0], q_off[1] - o[1], 'D', ms=15, mfc='none',
                                    mec=COLORS[a], mew=2, zorder=4)
                    lines.append(f'{a:<15} drops {len(rs) - len(placed)} of {len(rs)}')
            ax.text(0.02, 0.02, 'median member spread\n' + '\n'.join(lines),
                    transform=ax.transAxes, fontsize=9.5, family='monospace', color=INK,
                    va='bottom', bbox={'fc': SURFACE, 'ec': GRID, 'lw': 1})
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid()
            ax.set_axisbelow(True)
            ax.set_xlabel('east (m) from the off centroid')
            if ci == 0:
                ax.set_ylabel('north (m)')
            ax.set_title(f"site {rs[0]['site_id']}: {len(rs)} views, mean |tilt| "
                         f"{_f(rs[0], 'site_mean_tilt_deg'):.2f} deg", fontsize=11)
    handles = [Line2D([], [], marker=MARKERS[a], color=COLORS[a], ls=RING_STYLE[a], ms=9,
                      lw=1.8, label=f'{LABEL[a]}; ring = centroid') for a in SITE_ARMS]
    handles.append(Line2D([], [], marker='D', mfc='none', mec=COLORS[gpp.ARM_FULL], ms=12,
                          ls='none', label='a view the full pose cannot place (> 25 m)'))
    rule = ('Rule (fixed before drawing): Paterson TEST half, auto height, association frozen '
            f'from off; candidates = sites with >= {SITE_MIN_VIEWS} operational views on '
            'distinct panos;\nranked by mean member |tilt|, ties by seed 116. Data: '
            'data/addendum_site_examples.csv, data/addendum_site_candidates.csv.')
    fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=10.5, title=rule,
               title_fontsize=10.5)
    fig.suptitle(f'Partial places every view of all {n_cand:,} candidate sites; the full pose '
                 f'drops a view on {dropped_full} ({dropped_full / n_cand:.0%}), including the '
                 'most tilted', x=0.01, ha='left', fontsize=13, weight='bold')
    return fig


def fig9_vintage_k(plt, rows, coefs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    own = next(r for r in coefs if r['height'] == 'auto' and r['scope'] == 'city'
               and r['city'] == 'laurens_gsv' and r['tier'] == '0.3000')
    entries = [(f"whole run, {r['vintage']} ({int(r['n_rows']):,} rows, "
                f"{int(r['n_panos'])} panos)", r, '#2a78d6') for r in rows]
    entries.append((f"#116 TRAIN half, all years ({int(own['n_rows']):,} rows)", own, MUTED))
    for ax, (k, se, name, frozen) in zip(axes, (
            ('k_pitch', 'se_pitch', 'k_pitch', geo.PARTIAL_POSE_K_GSV[0]),
            ('k_roll', 'se_roll', 'k_roll', geo.PARTIAL_POSE_K_GSV[1]))):
        for yi, (_lab, r, color) in enumerate(entries):
            if r.get(k) in ('', None):
                ax.annotate('< 50 fit rows: not fit', (0.45, yi), fontsize=FOOT,
                            color=INK2, va='center')
                continue
            ax.errorbar(_f(r, k), yi, xerr=1.96 * _f(r, se), fmt='o', color=color, ms=7, lw=2,
                        capsize=3)
            ax.annotate(f'{_f(r, k):.2f} [{_f(r, k) - 1.96 * _f(r, se):.2f}, '
                        f'{_f(r, k) + 1.96 * _f(r, se):.2f}]', (_f(r, k), yi), xytext=(0, 9),
                        textcoords='offset points', ha='center', fontsize=FOOT, color=INK)
        ax.axvline(frozen, color=INK, lw=1.3, ls='--')
        ax.annotate(f'frozen {frozen}', (frozen, -0.75), xytext=(-4, 0),
                    textcoords='offset points', fontsize=FOOT, color=INK, ha='right')
        ax.set_xlim(0, 0.8)
        ax.set_xlabel(f'{name} (95% CI, pano-clustered SE)')
        ax.grid(axis='x')
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(entries)), [e[0] for e in entries])
    axes[0].set_ylim(len(entries) - 0.4, -1.1)
    fig.suptitle("EXPLORATORY (a #116 train city, in-sample): laurens_gsv's whole-run CI "
                 'excludes the frozen constants on both axes', x=0.01, ha='left',
                 fontsize=12.5, weight='bold')
    fig.text(0.01, 0.015, 'Reported only; never scored. Data: '
             'data/addendum_vintage_k_laurens_gsv.csv; TRAIN half: '
             'runs/_pooled/partial_pose/coefficients.csv.', fontsize=FOOT, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    return fig


def draw():
    plt = _setup()
    read = gpp.read_csv
    figs = {
        'fig5_consistency': fig5_consistency(plt, read(DATA / 'addendum_consistency_pairs.csv'),
                                             read(DATA / 'addendum_consistency_bootstrap.csv')),
        'fig6_loss_bar': fig6_loss_bar(plt, read(DATA / 'addendum_loss_bar.csv'),
                                       read(DATA / 'addendum_power.csv')),
        'fig7_site_examples': fig7_site_examples(plt, read(DATA / 'addendum_site_examples.csv'),
                                                 read(DATA / 'addendum_site_candidates.csv')),
        'fig9_vintage_k': fig9_vintage_k(plt, read(DATA / 'addendum_vintage_k_laurens_gsv.csv'),
                                         read(DATA / 'coefficients.csv')),
    }
    for name, fig in figs.items():
        _save(fig, name)
        plt.close(fig)
        print(f'  wrote {FIG_DIR / name}.png/.svg', file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--refresh', action='store_true',
                    help='recompute data/addendum_* first (site examples, candidates and the '
                         'consistency bootstrap re-fuse runs/paterson)')
    ap.add_argument('--runs-root', default=str(REPO_ROOT / 'runs'))
    args = ap.parse_args(argv)
    if args.refresh:
        refresh(args.runs_root)
    draw()


if __name__ == '__main__':
    main()
