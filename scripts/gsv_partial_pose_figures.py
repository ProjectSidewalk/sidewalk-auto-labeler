"""Figures for the 2026-10-01 addendum of docs/gsv-partial-pose-study.md (#116 follow-up).

Five figures, each answering one question:

    fig5_consistency    Does production's `--apply-pose partial` (frozen pooled constants)
                        place members like the study's fitted arms? (Paterson TEST half)
    fig6_loss_bar       What does the pool-sized recall clause demand, and how often does it
                        fail -- against a real loss, at the null, and under churn?
    fig7_site_examples  What does a fraction of the tilt do to real multi-view sites?
    fig8_store_gate     When may a store-built city's pose be used? (the store-pose gate)
    fig9_vintage_k      EXPLORATORY: does the leaked fraction differ by capture vintage?

Usage (no GPU, no network):
    python scripts/gsv_partial_pose_figures.py              # redraw from committed data/
    python scripts/gsv_partial_pose_figures.py --refresh    # recompute data/ first
The #121 figures (fig1-fig4) stay with `gsv_partial_pose.py figures`; this script draws only
the addendum figures, so the pinned confirm code in gsv_partial_pose.py is untouched.

Every figure is drawn from CSV/JSON files under docs/figures/gsv-partial-pose/data/
(prefix `addendum_`), so a redraw is byte-reproducible: PNG at 200 dpi and SVG, with the
SVG date stripped and its element ids salted with a constant. `--refresh` rebuilds that
data: the consistency, vintage and gate files are copied from the committed run outputs;
the loss-bar table and the power simulation are recomputed (seed 116); the site examples
are re-fused from runs/paterson/{results.jsonl,depth/index.csv} (the only step that
needs run inputs, which are not committed).
"""
import argparse
import csv
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

# One visual system with the #121 figures (gpp._style; dataviz reference palette, fixed
# slots). Arm colours follow the arm across every figure; partial-pooled takes the next
# unused slot (green). The adjacent order used in fig5 was run through the dataviz
# validator (passes CVD and normal-vision floors; magenta < 3:1 gets direct labels).
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
COLORS = {**gpp.FIG_COLORS, gpp.ARM_POOLED: '#008300'}
MARKERS = {**gpp.FIG_MARKERS, gpp.ARM_POOLED: 'P'}
SEQ_BLUE = ('#86b6ef', '#2a78d6', '#184f95')       # 1%, 2%, 3% loss (ordinal, light->dark)
CHURN = ('#eb6834', '#eda100')
LABEL = {gpp.ARM_OFF: 'off (flat, production default)',
         gpp.ARM_SHUFFLED: 'partial-shuffled (control)',
         gpp.ARM_PARTIAL: 'partial (city fit)', gpp.ARM_LOCO: 'partial-loco',
         gpp.ARM_POOLED: 'partial-pooled (production)', gpp.ARM_MIRROR: 'partial-mirror',
         gpp.ARM_FULL: 'full pose'}


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
    import numpy as np
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
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for n, pl, pg in POWER_TABLE_CELLS:
        d = rng.multinomial(n, [pl, pg, 1 - pl - pg], size=reps)
        rows.append({'n': n, 'p_lost': pl, 'p_gained': pg, 'reps': reps,
                     'k_star': gpp.recall_loss_bar(n),
                     'p_fail_rule': float(((d[:, 0] - d[:, 1]) >= gpp.recall_loss_bar(n)).mean())})
    return rows


def site_example_rows(runs_root, n_sites=SITE_EXAMPLES, seed=SEED):
    """Re-fuse Paterson's TEST half (auto) under off; among sites with >= SITE_MIN_VIEWS
    operational members on distinct panos that every arm places, take the n_sites with the
    largest mean member |tilt| (ties: seeded shuffle, seed 116). One row per member with
    its camera and its ground point under off / partial-pooled / full / partial-mirror
    (association frozen from off; mirror uses the pooled constants with both signs flipped)."""
    panos, height = gpp.load_city('paterson', fs.HEIGHT_AUTO, runs_root)
    _train, test = gpp.split_halves(panos, seed)
    k = geo.PARTIAL_POSE_K_GSV
    base = fs.FuseParams(min_confidence=gpp.OPERATIONAL_CONFIDENCE, camera_height_m=height)
    grav = gpp.replace(base, apply_pose=fs.POSE_GRAVITY)
    arms = {gpp.ARM_OFF: (test, gpp.replace(base, apply_pose=fs.POSE_OFF)),
            gpp.ARM_POOLED: (test, gpp.replace(base, apply_pose=fs.POSE_PARTIAL)),
            gpp.ARM_FULL: (gpp.posed_panos(test, 1.0, 1.0), grav),
            gpp.ARM_MIRROR: (gpp.posed_panos(test, -k[0], -k[1]), grav)}
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
            if any(g is None for g in gs):
                break
            pts[arm] = [frame.to_enu(g.lat, g.lng) for g in gs]
        else:
            tilts = [math.hypot(*gpp.stored_tilt(by_id[d.pano_id])) for d in ms]
            cands.append((sum(tilts) / len(tilts), s, ms, pts, tilts))
    random.Random(seed).shuffle(cands)
    cands.sort(key=lambda c: -c[0])
    rows = []
    for rank, (mean_tilt, s, ms, pts, tilts) in enumerate(cands[:n_sites], 1):
        for i, d in enumerate(ms):
            p = by_id[d.pano_id]
            ce, cn = frame.to_enu(p.lat, p.lng)
            pitch, roll = gpp.stored_tilt(p)
            row = {'rank': rank, 'site_id': s.id, 'candidates': len(cands),
                   'site_mean_tilt_deg': mean_tilt, 'pano_id': d.pano_id,
                   'det_index': d.det_index, 'cam_e': ce, 'cam_n': cn,
                   'pitch_deg': pitch, 'roll_deg': roll, 'tilt_deg': tilts[i]}
            for arm in arms:
                row[f'{arm}_e'], row[f'{arm}_n'] = pts[arm][i]
            rows.append(row)
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
    _write_csv(DATA / 'addendum_site_examples.csv', site_example_rows(runs_root))


# ------------------------------------------------------------------------ figures

def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gpp._style(plt)
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'gsv-partial-pose-116',
                         'axes.titlesize': 11, 'axes.titleweight': 'bold',
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


def fig5_consistency(plt, rows):
    """Change from off (bars from zero = off) so millimetre agreement and decimetre gains
    are both visible; off's own value is written on its row."""
    order = (gpp.ARM_SHUFFLED, gpp.ARM_PARTIAL, gpp.ARM_LOCO, gpp.ARM_POOLED, gpp.ARM_MIRROR)
    by = {(r['height'], r['arm']): r for r in rows if r['frame'] == gpp.FRAME_REASSOC}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), sharey=True)
    gap = len(order) + 1.6
    ys, labels = [], []
    for gi, h in enumerate(('auto', '2.6')):
        n_off = int(by[h, gpp.ARM_OFF]['pairs_scored'])
        for ai, arm in enumerate(order):
            ys.append(gi * gap + ai)
            n = int(by[h, arm]['pairs_scored'])
            labels.append(LABEL[arm] + (f'  [{n:,} pairs]' if n != n_off else ''))
    for ax, stat, name in ((axes[0], 'median_m', 'Median'), (axes[1], 'p90_m', 'p90')):
        for gi, h in enumerate(('auto', '2.6')):
            off = _f(by[h, gpp.ARM_OFF], stat)
            y0 = gi * gap
            for ai, arm in enumerate(order):
                dv = _f(by[h, arm], stat) - off
                ax.barh(y0 + ai, dv, height=0.72, color=COLORS[arm], edgecolor=SURFACE,
                        linewidth=2, hatch='///' if arm == gpp.ARM_SHUFFLED else None)
                ax.annotate(f'{dv:+.3f}', (dv, y0 + ai), xytext=(5 if dv >= 0 else -5, 0),
                            textcoords='offset points', va='center', fontsize=9,
                            ha='left' if dv >= 0 else 'right', color=INK)
            trio = [_f(by[h, a], stat) for a in (gpp.ARM_PARTIAL, gpp.ARM_LOCO, gpp.ARM_POOLED)]
            ax.annotate(f'partial arms agree\nwithin {max(trio) - min(trio):.4f} m',
                        (0.70, y0 + 2), xycoords=ax.get_yaxis_transform(), va='center',
                        fontsize=8.5, color=INK2)
            n = int(by[h, gpp.ARM_OFF]['pairs_scored'])
            ax.annotate(f'height {h}{" m" if h == "2.6" else ""}: off = {off:.3f} m '
                        f'({n:,} pairs)', (0.01, y0 - 0.95), xycoords=ax.get_yaxis_transform(),
                        fontsize=9.5, color=INK, weight='bold', ha='left',
                        bbox={'fc': SURFACE, 'ec': 'none', 'pad': 1})
        ax.axvline(0, color=INK, lw=1.2)
        ax.set_title(f'{name} within-site pair distance')
        ax.set_xlabel(f'{name.lower()} minus off (m)  <- tighter | looser ->')
        ax.grid(axis='x')
        ax.set_axisbelow(True)
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, -0.45), max(hi, 0.45))
    axes[0].set_yticks(ys, labels)
    axes[0].set_ylim(gap + len(order) - 0.4, -1.5)
    fig.suptitle("Production's `--apply-pose partial` matches the fitted study arms, and all "
                 'three beat the shuffled control', x=0.01, ha='left', fontsize=12,
                 weight='bold')
    fig.text(0.01, 0.012, 'Paterson #116 TEST half (seed 116), tier 0.30, each arm '
             're-associated, on the pairs every #116 decision arm co-associates.\nData: '
             'data/addendum_consistency_pairs.csv (= runs/paterson/partial_pose/'
             'consistency_pairs.csv), columns median_m, p90_m, pairs_scored.', fontsize=8,
             color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def fig6_loss_bar(plt, bar_rows, power):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    ax = axes[0]
    n = [int(r['n']) for r in bar_rows]
    ax.step(n, [int(r['k_star']) for r in bar_rows], where='post', color='#2a78d6', lw=2,
            label='k*(n): pool-sized bar (frozen rule)')
    ax.step(n, [int(r['fixed_1pt_fail_at']) for r in bar_rows], where='post', color=MUTED,
            lw=2, ls='--', label='fixed 1.0 pt (#116, replaced)')
    ax.plot(157, 2, 'o', color='#e34948', ms=9, mec=SURFACE, mew=2, zorder=5)
    ax.annotate('Bend, #116 half split: n = 157, L = 2\nFAIL under 1.0 pt (bar 2),\n'
                'pass under k*(157) = 5', (157, 2), xytext=(560, 2.6),
                textcoords='data', fontsize=9, color=INK,
                arrowprops={'arrowstyle': '-', 'color': INK2, 'lw': 1})
    for nn in (100, 300, 500):
        k = gpp.recall_loss_bar(nn)
        ax.annotate(f'k*({nn}) = {k}', (nn, k), xytext=(4, 6), textcoords='offset points',
                    fontsize=8.5, color=INK2, ha='left', va='bottom')
    ax.set_xlabel('n = off-pool ramps (recall denominator at 2.5 m)')
    ax.set_ylabel('net ramps lost (lost - gained) at which (ii) FAILS')
    ax.set_title('What the recall clause demands')
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=9)
    ax = axes[1]
    styles = dict(zip([s[0] for s in SIM_SCENARIOS], (*SEQ_BLUE, *CHURN)))
    for label, _pl, _pg in SIM_SCENARIOS:
        rs = [r for r in power if r['scenario'] == label]
        xs = [int(r['n']) for r in rs]
        ax.plot(xs, [_f(r, 'p_fail_rule') for r in rs], color=styles[label], lw=2,
                label=label.replace('null:', "frozen rule's null:"))
        ax.plot(xs, [_f(r, 'p_fail_sign') for r in rs], color=styles[label], lw=1.4, ls=':')
    ax.axhline(0.05, color=INK, lw=1, ls='--')
    ax.annotate('alpha = 0.05', (800, 0.05), xytext=(0, 4), textcoords='offset points',
                ha='right', fontsize=9, color=INK)
    r157 = next(r for r in power if r['scenario'] == SIM_SCENARIOS[1][0] and r['n'] == '160')
    ax.annotate(f'a real 2% loss fails only\n{_f(r157, "p_fail_rule"):.0%} of the time at '
                'n = 160', (160, _f(r157, 'p_fail_rule')), xytext=(540, 0.36),
                textcoords='data', fontsize=9, color=INK,
                arrowprops={'arrowstyle': '-', 'color': INK2, 'lw': 1})
    ax.set_ylim(0, 1)
    ax.set_xlabel('n = off-pool ramps')
    ax.set_ylabel('P(clause (ii) FAILS)')
    ax.set_title('How often it fails: solid = frozen rule, dotted = sign test')
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8.5,
              title='true per-ramp change vs off', title_fontsize=8.5)
    fig.suptitle('The pool-sized recall bar is looser than 1.0 pt at every n, weak against a '
                 'real 2% loss, and fails too often under churn', x=0.01, ha='left',
                 fontsize=12, weight='bold')
    fig.text(0.01, 0.005, f'Right: {SIM_REPS:,} multinomial draws per point, seed 116. '
             'Sign test = exact one-sided binomial on discordant ramps (lost vs lost + gained, '
             'p = 0.5); its null is "no net loss", so under 2%/2% churn it is the size, and '
             'under 1%/0% and 3%/2% it is power. Data: data/addendum_loss_bar.csv, '
             'data/addendum_power.csv.', fontsize=8, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return fig


def _spread(pts):
    d = sorted(math.hypot(a[0] - b[0], a[1] - b[1])
               for i, a in enumerate(pts) for b in pts[i + 1:])
    return d[len(d) // 2] if len(d) % 2 else (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2


def fig7_site_examples(plt, rows):
    arms = (gpp.ARM_OFF, gpp.ARM_POOLED, gpp.ARM_FULL, gpp.ARM_MIRROR)
    ranks = sorted({int(r['rank']) for r in rows})
    fig, axes = plt.subplots(2, len(ranks), figsize=(5.2 * len(ranks), 10.2),
                             gridspec_kw={'height_ratios': (1, 1.25)})
    for ci, rank in enumerate(ranks):
        rs = [r for r in rows if int(r['rank']) == rank]
        pts = {a: [(_f(r, f'{a}_e'), _f(r, f'{a}_n')) for r in rs] for a in arms}
        cen = {a: (sum(p[0] for p in pts[a]) / len(rs), sum(p[1] for p in pts[a]) / len(rs))
               for a in arms}
        o = cen[gpp.ARM_OFF]
        ax = axes[0, ci]
        for r, (ge, gn) in zip(rs, pts[gpp.ARM_OFF]):
            ce, cn = _f(r, 'cam_e') - o[0], _f(r, 'cam_n') - o[1]
            ax.plot([ce, ge - o[0]], [cn, gn - o[1]], color=GRID, lw=1.2, zorder=1)
            ax.plot(ce, cn, 's', color=INK2, ms=7, mec=SURFACE, mew=1.5, zorder=3)
            ax.annotate(f"{_f(r, 'tilt_deg'):.1f} deg", (ce, cn), xytext=(5, 5),
                        textcoords='offset points', fontsize=8.5, color=INK2)
        ax.plot(0, 0, '*', color=INK, ms=12, zorder=4)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_title(f"Site {rank}: {len(rs)} views, mean |tilt| "
                     f"{_f(rs[0], 'site_mean_tilt_deg'):.1f} deg")
        ax.set_xlabel('east (m), site at the star')
        if ci == 0:
            ax.set_ylabel('north (m)')
        ax = axes[1, ci]
        for a in arms:
            for e, n in pts[a]:
                ax.plot(e - o[0], n - o[1], MARKERS[a], color=COLORS[a], ms=8, mec=SURFACE,
                        mew=1.5, alpha=0.95, zorder=3,
                        label=LABEL[a] if (ci == 0 and (e, n) == pts[a][0]) else None)
            ax.plot(cen[a][0] - o[0], cen[a][1] - o[1], 'o', ms=17, mfc='none',
                    mec=COLORS[a], mew=2, zorder=2)
        for r, (ge, gn) in zip(rs, pts[gpp.ARM_OFF]):
            ux, uy = ge - _f(r, 'cam_e'), gn - _f(r, 'cam_n')
            nrm = math.hypot(ux, uy)
            ax.plot([ge - o[0] - 3 * ux / nrm, ge - o[0] + 3 * ux / nrm],
                    [gn - o[1] - 3 * uy / nrm, gn - o[1] + 3 * uy / nrm], color=GRID, lw=1,
                    zorder=1)
        text = '\n'.join(f'{a:<15} {_spread(pts[a]):.2f} m' for a in arms)
        ax.text(0.02, 0.02, 'median member spread\n' + text, transform=ax.transAxes,
                fontsize=8.5, family='monospace', color=INK, va='bottom',
                bbox={'fc': SURFACE, 'ec': GRID, 'lw': 1})
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlabel('east (m) from the off centroid')
        if ci == 0:
            ax.set_ylabel('north (m)')
        ax.set_title(f'member spread: off {_spread(pts[gpp.ARM_OFF]):.2f} m -> partial '
                     f'{_spread(pts[gpp.ARM_POOLED]):.2f} m', fontsize=10.5)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.925))
    fig.suptitle('On the most-tilted Paterson sites the pose slides each ground point along its '
                 'own ray; site by site the\neffect is mixed (partial tightens sites 1 and 3, '
                 'loosens site 2), which is why #116 scores thousands of pairs', x=0.01,
                 ha='left', fontsize=12, weight='bold')
    n_c = rows[0]['candidates']
    fig.text(0.01, 0.005, f'Rule fixed before drawing: Paterson TEST half, auto height; of '
             f'the {int(n_c):,} sites with >= {SITE_MIN_VIEWS} operational views on distinct '
             'panos placed by every arm, the 3 with the largest mean member |tilt| (ties: '
             'seed 116). Association frozen from off. Top: cameras (squares, |tilt| '
             "labelled) and their off rays. Bottom: each view's ground point per arm; rings = "
             'member centroid; spread = median pairwise member distance. '
             'Data: data/addendum_site_examples.csv.',
             fontsize=8, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.89))
    return fig


def fig8_store_gate(plt, gate):
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.axis('off')
    cells = [['(+1, +1)', 'PS pitch, PS roll', 'same convention as streetlevel'],
             ['(+1, -1)', 'PS pitch, -PS roll', 'roll sign flipped'],
             ['(-1, +1)', '-PS pitch, PS roll', 'pitch sign flipped'],
             ['(-1, -1)', '-PS pitch, -PS roll', 'both flipped']]
    t = ax.table(cellText=cells, colLabels=['mapping', 'compared as', 'meaning'],
                 colWidths=[0.11, 0.2, 0.27], loc='upper left', cellLoc='left',
                 colLoc='left', bbox=(0, 0.44, 0.58, 0.54))
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    for (r, _c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor('#f0efec' if r == 0 else SURFACE)
        if r == 0:
            cell.set_text_props(weight='bold')
    rule = (f"PASS only if ONE mapping agrees within {gate['tol_deg']} deg on BOTH angles\n"
            f"for >= {gate['min_agree']:.0%} of >= {gate['min_compared']} comparable panos "
            f"(seeded sample,\nseed {gate['sample_seed']}, up to 200; streetlevel METADATA "
            "only, no imagery).\nTwo mappings over the bar = sign not identified = FAIL.\n"
            'FAIL or no network: confirm refuses to score the city.\n'
            'PASS: the mapping is applied in memory and the panos are posed.')
    ax.text(0.66, 0.92, 'The rule', fontsize=10.5, weight='bold', va='top', color=INK,
            transform=ax.transAxes)
    ax.text(0.66, 0.84, rule, fontsize=9, va='top', color=INK, transform=ax.transAxes)
    rec = (f"laurens_gsv dry run: status {gate['status'].upper()} - {gate['store_panos']} "
           f"store-built panos of {gate['run_panos']:,} ({gate['reason']}).")
    ax.text(0, 0.30, 'Recorded so far', fontsize=10.5, weight='bold', color=INK,
            transform=ax.transAxes)
    ax.text(0, 0.21, rec, fontsize=9, color=INK, transform=ax.transAxes)
    ax.text(0, 0.11, 'Vancouver (store-built, source_detail ps_store) is eligible for the '
            'confirmatory run only after a PASS.\nA plain `fuse_sites --apply-pose partial` '
            'still raycasts ps_store panos flat: the mapping is not persisted.', fontsize=9,
            color=INK2, transform=ax.transAxes)
    fig.suptitle('When may a store-built city\'s pose be used? Only after one sign mapping is '
                 'verified against streetlevel', x=0.01, ha='left', fontsize=12, weight='bold')
    fig.text(0.01, 0.01, 'Data: data/addendum_store_gate_laurens_gsv.json (= runs/laurens_gsv/'
             'partial_pose/confirm_exploratory_auto/store_pose_gate.json).', fontsize=8,
             color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return fig


def fig9_vintage_k(plt, rows, coefs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    vint = [r for r in rows]
    labels = [f"{r['vintage']} ({int(r['n_rows']):,} rows, {int(r['n_panos'])} panos)"
              for r in vint]
    own = next(r for r in coefs if r['height'] == 'auto' and r['scope'] == 'city'
               and r['city'] == 'laurens_gsv' and r['tier'] == '0.3000')
    for ax, (k, se, name, frozen) in zip(axes, (
            ('k_pitch', 'se_pitch', 'k_pitch', geo.PARTIAL_POSE_K_GSV[0]),
            ('k_roll', 'se_roll', 'k_roll', geo.PARTIAL_POSE_K_GSV[1]))):
        for yi, r in enumerate(vint):
            if r[k] in ('', None):
                ax.annotate('too few fit rows (< 50): not fit', (0.435, yi), fontsize=8.5,
                            color=INK2, va='center')
                continue
            ax.errorbar(_f(r, k), yi, xerr=1.96 * _f(r, se), fmt='o', color='#2a78d6',
                        ms=7, lw=2, capsize=3)
            ax.annotate(f'{_f(r, k):.2f}', (_f(r, k), yi), xytext=(0, 9),
                        textcoords='offset points', ha='center', fontsize=9, color=INK)
        ax.axvline(frozen, color=INK, lw=1.2, ls='--')
        own_k = float(own[k])
        left = own_k < frozen
        ax.annotate(f'frozen {frozen}', (frozen, -0.62), xytext=(-4 if not left else 4, 0),
                    textcoords='offset points', fontsize=8.5, color=INK,
                    ha='right' if not left else 'left')
        ax.axvline(own_k, color=MUTED, lw=1.2, ls=':')
        ax.annotate(f'#116 TRAIN-half fit {own_k:.2f}', (own_k, len(vint) - 0.45),
                    xytext=(-4 if left else 4, 0), textcoords='offset points', fontsize=8.5,
                    color=INK2, ha='right' if left else 'left')
        ax.set_xlim(0, 0.8)
        ax.set_xlabel(f'{name} (95% CI, pano-clustered SE)')
        ax.grid(axis='x')
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(vint)), labels)
    axes[0].set_ylim(len(vint) - 0.3, -0.9)
    fig.suptitle('EXPLORATORY (a #116 train city, in-sample): laurens_gsv\'s leaked fractions '
                 'by capture vintage', x=0.01, ha='left', fontsize=12, weight='bold')
    fig.text(0.01, 0.01, 'Reported only; never scored. Data: '
             'data/addendum_vintage_k_laurens_gsv.csv (= runs/laurens_gsv/partial_pose/'
             'confirm_exploratory_auto/vintage_k.csv).', fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    return fig


def draw():
    plt = _setup()
    read = gpp.read_csv
    figs = {
        'fig5_consistency': fig5_consistency(plt, read(DATA / 'addendum_consistency_pairs.csv')),
        'fig6_loss_bar': fig6_loss_bar(plt, read(DATA / 'addendum_loss_bar.csv'),
                                       read(DATA / 'addendum_power.csv')),
        'fig7_site_examples': fig7_site_examples(plt, read(DATA / 'addendum_site_examples.csv')),
        'fig8_store_gate': fig8_store_gate(plt, json.loads(
            (DATA / 'addendum_store_gate_laurens_gsv.json').read_text(encoding='utf-8'))),
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
                    help='recompute data/addendum_* first (site examples need runs/paterson)')
    ap.add_argument('--runs-root', default=str(REPO_ROOT / 'runs'))
    args = ap.parse_args(argv)
    if args.refresh:
        refresh(args.runs_root)
    draw()


if __name__ == '__main__':
    main()
