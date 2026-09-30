"""gsv_partial_pose (#116): roll wrap, arm signs, the |tilt|-bucketed shuffle, the half
split, and the coefficient fit on a synthetic city with a known leak -- offline."""
import math
import random

import geo
import fuse_sites as fs
import gsv_partial_pose as gpp
from test_fuse_sites import FRAME

LEAK = (0.2, 0.5)     # the synthetic city's true (k_pitch, k_roll)


def _pano(pid, pitch=1.0, roll=359.4, dets=()):
    return fs.SlimPano(pid, 40.0, -74.0, 90.0, pitch, roll, '2024-06', 'launch', list(dets))


def test_stored_tilt_wraps_an_unwrapped_roll():
    assert gpp.stored_tilt(_pano('a', pitch=1.0, roll=359.4)) == (1.0, -0.6000000000000227)
    assert gpp.stored_tilt(_pano('b', pitch=None)) is None


def test_arm_pose_flips_pitch_keeps_roll():
    assert gpp.arm_pose(2.0, -1.0, 1.0, 1.0) == (-2.0, -1.0)
    assert gpp.arm_pose(2.0, -1.0, 0.25, 0.5) == (-0.5, -0.5)
    assert gpp.arm_pose(2.0, -1.0, -0.25, -0.5) == (0.5, 0.5)    # the mirror


def test_full_arm_puts_a_nose_down_pano_ahead_closer_than_flat():
    """Streetlevel pitch > 0 is nose DOWN: a pixel straight ahead then points lower in the
    world than in the image, so the corrected ray meets the ground sooner."""
    p = _pano('a', pitch=2.0, roll=0.0)
    flat = geo.detection_ground_point(geo.pano_pose(p.pose_fields(camera_pitch=None,
                                                                  camera_roll=None)),
                                      0.5, 0.56, apply_pose=False)
    posed = gpp.posed_panos([p], 1.0, 1.0)[0]
    full = geo.detection_ground_point(fs.pano_pose(posed, fs.POSE_GRAVITY), 0.5, 0.56,
                                      apply_pose=True)
    assert full.range_m < flat.range_m


def test_posed_panos_leaves_unposed_panos_alone():
    p = _pano('a', pitch=None)
    assert gpp.posed_panos([p], 0.3, 0.5) == [p]


def test_tilt_bucket_edges():
    assert gpp.tilt_bucket(0.0, 0.0) == 0
    assert gpp.tilt_bucket(0.3, 0.4) == 1          # |tilt| = 0.5 is the lower edge
    assert gpp.tilt_bucket(30.0, 40.0) == len(gpp.TILT_EDGES) - 2


def test_shuffle_permutes_only_within_buckets_and_is_deterministic():
    rng = random.Random(3)
    panos = [_pano(f'p{i}', pitch=rng.uniform(-4, 4), roll=rng.uniform(-3, 3) % 360)
             for i in range(300)]
    tilts, own = gpp.shuffled_tilts(panos, seed=116)
    again, _ = gpp.shuffled_tilts(list(reversed(panos)), seed=116)
    assert tilts == again
    by_bucket_before, by_bucket_after = {}, {}
    for p in panos:
        t = gpp.stored_tilt(p)
        b = gpp.tilt_bucket(*t)
        by_bucket_before.setdefault(b, []).append(t)
        by_bucket_after.setdefault(b, []).append(tilts[p.pano_id])
        assert gpp.tilt_bucket(*tilts[p.pano_id]) == b
    for b in by_bucket_before:
        assert sorted(by_bucket_before[b]) == sorted(by_bucket_after[b])
    assert own < 30       # a permutation, not the identity


def test_half_split_is_stable_and_near_even():
    ids = [f'pano{i:05d}' for i in range(4000)]
    test = [i for i in ids if gpp.in_test_half(i)]
    assert 1800 < len(test) < 2200
    assert test == [i for i in ids if gpp.in_test_half(i, seed=116)]
    assert test != [i for i in ids if gpp.in_test_half(i, seed=117)]


def test_ols_clustered_recovers_slopes():
    rng = random.Random(0)
    X = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(4000)]
    y = [0.1 + 0.2 * a + 0.5 * b + rng.gauss(0, 0.05) for a, b in X]
    beta, se, g = gpp.ols_clustered(y, X, [i // 4 for i in range(4000)])
    assert g == 1000
    assert abs(beta[0] - 0.1) < 0.01 and abs(beta[1] - 0.2) < 0.01 and abs(beta[2] - 0.5) < 0.01
    assert all(s > 0 for s in se)


def _synthetic_city(leak, n_cams=120, seed=5):
    """Cameras over a grid of ramps; each camera's stored tilt leaks `leak` of itself into
    the image rows (first order: theta = elev + k_p pitch cos b + k_r roll sin b)."""
    rng = random.Random(seed)
    ramps = [(40.0 * i, 40.0 * j) for i in range(4) for j in range(4)]
    panos = []
    for c in range(n_cams):
        re, rn = rng.choice(ramps)
        ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(4, 14)
        pe, pn = re + dist * math.sin(ang), rn + dist * math.cos(ang)
        heading = rng.uniform(0, 360)
        pitch, roll = rng.uniform(-3, 3), rng.uniform(-3, 3)
        dets = []
        for te, tn in ramps:
            de, dn = te - pe, tn - pn
            d = math.hypot(de, dn)
            if d > 20:
                continue
            bearing = math.degrees(math.atan2(de, dn))
            phi = math.radians(geo.norm_deg(bearing - heading))
            theta = -math.atan(geo.DEFAULT_CAMERA_HEIGHT_M / d) + math.radians(
                leak[0] * pitch * math.cos(phi) + leak[1] * roll * math.sin(phi))
            dets.append((len(dets), 0.5 + phi / (2 * math.pi), 0.5 - theta / math.pi, 0.9))
        lat, lng = FRAME.to_latlng(pe, pn)
        panos.append(fs.SlimPano(f'c{c:03d}', lat, lng, heading, pitch, roll % 360.0,
                                 '2024-06', 'launch', dets))
    return panos, ramps


def test_fit_recovers_a_known_leak_and_the_arm_undoes_it():
    panos, ramps = _synthetic_city(LEAK)
    c = gpp.fit_coefficients(gpp.fit_rows(panos, geo.DEFAULT_CAMERA_HEIGHT_M))
    assert abs(c['k_pitch'] - LEAK[0]) < 0.05 and abs(c['k_roll'] - LEAK[1]) < 0.05
    coefs = {gpp.ARM_PARTIAL: (c['k_pitch'], c['k_roll']),
             gpp.ARM_LOCO: (c['k_pitch'], c['k_roll'])}
    arms, _ = gpp.build_arms(panos, coefs, geo.DEFAULT_CAMERA_HEIGHT_M,
                             arms=(gpp.ARM_OFF, gpp.ARM_PARTIAL, gpp.ARM_MIRROR))
    place = gpp.make_placer(arms)

    def mean_error(arm):
        errs = []
        for p in panos:
            pe, pn = FRAME.to_enu(p.lat, p.lng)
            for _i, x, y, _c in p.detections:
                g = place(arm, p.pano_id, x, y)
                if g is None:
                    continue
                ge, gn = FRAME.to_enu(g.lat, g.lng)
                errs.append(min(math.hypot(ge - te, gn - tn) for te, tn in ramps))
        return sum(errs) / len(errs)
    off, partial, mirror = (mean_error(a) for a in (gpp.ARM_OFF, gpp.ARM_PARTIAL,
                                                    gpp.ARM_MIRROR))
    assert partial < 0.5 * off < mirror


def test_verdict_passes_only_when_every_clause_holds():
    def pr(city, arm, med, p90, n=1000):
        return {'city': city, 'height': 'auto', 'frame': gpp.FRAME_REASSOC, 'arm': arm,
                'pairs_scored': str(n), 'median_m': str(med), 'p90_m': str(p90)}
    pairs = []
    for city in ('x', 'y'):
        pairs += [pr(city, gpp.ARM_OFF, 2.0, 4.0), pr(city, gpp.ARM_PARTIAL, 1.9, 3.9),
                  pr(city, gpp.ARM_LOCO, 1.9, 3.9), pr(city, gpp.ARM_SHUFFLED, 2.1, 4.1),
                  pr(city, gpp.ARM_LOCO_SHUFFLED, 2.1, 4.1)]
    gt = [{'city': 'x', 'height': 'auto', 'arm': a, 'recall_off_pool_2p5m': '0.90',
           'gt_marks_unplaceable': '3', 'off_pool_ramps': '100'}
          for a in (gpp.ARM_OFF, gpp.ARM_PARTIAL, gpp.ARM_LOCO)]
    inv = [{'city': c, 'height': 'auto', 'frame': gpp.FRAME_FROZEN, 'arm': a,
            'radius_m': '5.0000', 'median_m': '1.5', 'p90_m': '3.0'}
           for c in gpp.INVENTORY_CITIES for a in (gpp.ARM_OFF, gpp.ARM_PARTIAL, gpp.ARM_LOCO)]
    ok, clauses, _ = gpp.verdict(pairs, gt, inv, cities=('x', 'y'))
    assert ok and all(clauses.values())
    # the candidate loses to its shuffle in one city -> (i) fails
    pairs[3 + 5] = pr('y', gpp.ARM_SHUFFLED, 1.8, 4.1)
    ok, clauses, _ = gpp.verdict(pairs, gt, inv, cities=('x', 'y'))
    assert not ok and not clauses['i'] and clauses['ii'] and clauses['iv']


def test_pano_crop_wraps_the_seam_and_centres_the_mark(tmp_path):
    """A mark at x = 0.99 must pull columns from both edges; the mark is the centre column.
    A pano missing from the bundle is None, not an error."""
    from PIL import Image
    (tmp_path / 'panos').mkdir()
    img = Image.new('RGB', (400, 200), (0, 0, 0))
    img.paste((255, 0, 0), (0, 0, 20, 200))      # red at the left edge
    img.paste((0, 0, 255), (380, 0, 400, 200))   # blue at the right edge
    img.save(tmp_path / 'panos' / 'p.jpg', quality=95)
    arr, mark_row = gpp.pano_crop(tmp_path, 'p', 0.99, 0.5, half_w=0.05, half_h=0.1, out_px=1000)
    w = arr.shape[1]
    assert arr[:, 1, 2].mean() > 150 and arr[:, w - 2, 0].mean() > 150   # blue then red
    assert abs(mark_row - arr.shape[0] / 2) <= 1
    assert gpp.pano_crop(tmp_path, 'missing', 0.5, 0.5) is None


# --- #116 follow-up: production's `partial` and the confirmatory rule -------------------

def test_study_arm_is_productions_partial_pose():
    rng = random.Random(7)
    panos = [_pano(f'p{i}', pitch=rng.uniform(-4, 4), roll=rng.uniform(-3, 3) % 360)
             for i in range(50)] + [_pano('none', pitch=None), _pano('half', roll=None)]
    assert gpp.arm_pose is geo.partial_pitch_roll
    assert gpp.production_pose_mismatches(panos) == []


def test_recall_loss_bar_is_the_exact_binomial_tail():
    def tail(n, k, p=0.01):
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))
    for n in (50, 100, 157, 200, 300, 500, 800):
        k = gpp.recall_loss_bar(n)
        assert tail(n, k) <= 0.05 < tail(n, k - 1)
    assert [gpp.recall_loss_bar(n) for n in (100, 157, 200, 300, 500)] == [4, 5, 6, 7, 10]
    table = gpp.recall_loss_bar_table()
    assert table[0][0] == 50 and table[-1][1] == 800
    assert all(a[1] + 1 == b[0] and a[2] < b[2] for a, b in zip(table, table[1:]))


def _confirm_rows(n_pairs=1000, lost=0, gained=0, pool=157):
    pairs = [{'frame': gpp.FRAME_REASSOC, 'arm': a, 'pairs_scored': str(n_pairs),
              'median_m': m, 'p90_m': p}
             for a, m, p in ((gpp.ARM_OFF, '2.0', '4.0'), (gpp.ARM_PARTIAL, '1.9', '3.9'),
                             (gpp.ARM_SHUFFLED, '2.1', '4.1'))]
    gt = [{'arm': gpp.ARM_OFF, 'off_pool_ramps': str(pool), 'gt_marks_unplaceable': '5',
           'lost_vs_off_2p5m': '0', 'gained_vs_off_2p5m': '0'},
          {'arm': gpp.ARM_PARTIAL, 'off_pool_ramps': str(pool), 'gt_marks_unplaceable': '5',
           'lost_vs_off_2p5m': str(lost), 'gained_vs_off_2p5m': str(gained)}]
    inv = [{'city': c, 'frame': gpp.FRAME_FROZEN, 'arm': a, 'radius_m': '5.0000',
            'median_m': '1.0', 'p90_m': '2.0'}
           for c in gpp.INVENTORY_CITIES for a in (gpp.ARM_OFF, gpp.ARM_PARTIAL)]
    return pairs, gt, inv


def test_confirm_verdict_sizes_the_recall_clause_to_the_pool():
    # Bend's #116 half split (2 lost, 0 gained of 157) clears the pool-sized bar (k* = 5)
    assert gpp.confirm_verdict(*_confirm_rows(lost=2))[0] == 'PASS'
    assert gpp.confirm_verdict(*_confirm_rows(lost=6, gained=2))[0] == 'PASS'   # L = 4
    out, state, _ = gpp.confirm_verdict(*_confirm_rows(lost=7, gained=2))       # L = 5
    assert out == 'FAIL' and state['ii'] is False
    # too few pairs to score (i): inconclusive, never a pass
    assert gpp.confirm_verdict(*_confirm_rows(n_pairs=499))[0] == 'INCONCLUSIVE'
    assert gpp.confirm_verdict(*_confirm_rows(pool=49))[0] == 'INCONCLUSIVE'
    # a FAIL anywhere beats an inconclusive
    assert gpp.confirm_verdict(*_confirm_rows(n_pairs=499, lost=9))[0] == 'FAIL'


def test_confirm_refuses_a_train_city_unless_exploratory():
    import pytest
    with pytest.raises(SystemExit, match='train set'):
        gpp.main(['confirm', 'bend'])
