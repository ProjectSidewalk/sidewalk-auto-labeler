"""fuse_sites: the stage-2 world-space associator (issue #27).

All geometry is synthetic: panos are placed in a local ENU frame around BASE and
aimed so their detections raycast to known ground points, then the associator's
invariants are asserted in world space.
"""
import json
import math

import pytest

import geo
import fuse_sites as fs

BASE = (40.0, -74.0)
FRAME = geo.LocalFrame(*BASE)
H = geo.DEFAULT_CAMERA_HEIGHT_M


def make_pano(pano_id, pe, pn, targets, heading_deg=None,
              source='launch', capture='2024-06', height=None):
    """A SlimPano at ENU (pe, pn) whose i-th detection raycasts exactly to the
    i-th (te, tn, conf) target. Pitch/roll stay None so the flat raycast is exact.
    `height` is the camera's true (and recorded, measured) height; None means H,
    unmeasured."""
    if heading_deg is None:
        te, tn, _ = targets[0]
        heading_deg = math.degrees(math.atan2(te - pe, tn - pn)) % 360.0
    dets = []
    for i, (te, tn, conf) in enumerate(targets):
        de, dn = te - pe, tn - pn
        dist = math.hypot(de, dn)
        bearing = math.degrees(math.atan2(de, dn))
        x = 0.5 + geo.norm_deg(bearing - heading_deg) / 360.0
        y = 0.5 + math.atan((height or H) / dist) / math.pi
        dets.append((i, x, y, conf))
    lat, lng = FRAME.to_latlng(pe, pn)
    return fs.SlimPano(pano_id, lat, lng, heading_deg, None, None,
                       capture, source, dets, camera_height_m=height)


def enu_of(frame, te, tn):
    """A target's coordinates in the frame fuse() chose."""
    return frame.to_enu(*FRAME.to_latlng(te, tn))


def test_ring_of_panos_fuses_to_one_site():
    panos = [make_pano('p1', -10, 0, [(0, 0, 0.9)]),
             make_pano('p2', 10, 0, [(0, 0, 0.9)]),
             make_pano('p3', 0, -10, [(0, 0, 0.9)]),
             make_pano('p4', 0, 10, [(0, 0, 0.9)])]
    sites, frame, stats = fs.fuse(panos, fs.FuseParams())
    assert len(sites) == 1
    site = sites[0]
    assert len(site.pano_ids) == 4 and site.n_operational == 4
    te, tn = enu_of(frame, 0, 0)
    assert math.hypot(site.e - te, site.n - tn) < 0.5
    assert site.residual_per_dof() < 1.0


def test_same_pano_cannot_link_keeps_two_peaks_apart():
    # Two peaks in one heatmap, 3 m apart along the same ray: mutually compatible
    # under the gate, but they are two physical objects.
    panos = [make_pano('p1', 0, 0, [(10, 0, 0.9), (13, 0, 0.8)])]
    sites, _, _ = fs.fuse(panos, fs.FuseParams())
    assert len(sites) == 2


def test_dual_ramp_association_is_best_match_not_first_match():
    # Ramps A=(0,0) and B=(3,0); both panos see both. p2's detections arrive
    # B-first, so a first-match associator would join B's detection to site A.
    panos = [make_pano('p1', 0, -10, [(0, 0, 0.9), (3, 0, 0.9)], heading_deg=0.0),
             make_pano('p2', 0, 10, [(3, 0, 0.85), (0, 0, 0.85)], heading_deg=180.0)]
    sites, frame, _ = fs.fuse(panos, fs.FuseParams())
    assert len(sites) == 2
    for site in sites:
        assert len(site.members) == 2 and len(site.pano_ids) == 2
    positions = sorted((s.e, s.n) for s in sites)
    ae, an = enu_of(frame, 0, 0)
    be, bn = enu_of(frame, 3, 0)
    assert math.hypot(positions[0][0] - ae, positions[0][1] - an) < 0.5
    assert math.hypot(positions[1][0] - be, positions[1][1] - bn) < 0.5


def test_subthreshold_support_never_moves_an_operational_site():
    panos = [make_pano('p1', 0, -10, [(0, 0, 0.9)]),
             make_pano('p2', 10, 0, [(0.5, 0, 0.2)])]  # sub-threshold, 0.5 m off
    sites, frame, _ = fs.fuse(panos, fs.FuseParams())
    assert len(sites) == 1
    site = sites[0]
    assert site.n_operational == 1 and len(site.members) == 2
    flags = {d.pano_id: in_refit for d, in_refit in site.members}
    assert flags == {'p1': True, 'p2': False}
    # position is p1's estimate alone, not pulled toward p2's offset point
    te, tn = enu_of(frame, 0, 0)
    assert math.hypot(site.e - te, site.n - tn) < 0.1


def test_wholly_subthreshold_detections_form_their_own_refitting_sites():
    panos = [make_pano('p1', 0, -10, [(0, 0, 0.2)]),
             make_pano('p2', 0, 10, [(0, 0, 0.15)])]
    sites, _, _ = fs.fuse(panos, fs.FuseParams())
    assert len(sites) == 1
    site = sites[0]
    assert site.n_operational == 0
    assert all(in_refit for _, in_refit in site.members)
    assert site.n_refit == 2


def test_residual_rejection_splits_gate_passing_outliers():
    # Cross-ray offsets: ~2 m merges (residual/dof ~1), ~3.9 m passes the 9.21
    # gate but exceeds residual_per_dof_max=3.0 and must split.
    def city(offset):
        return [make_pano('p1', 0, -10, [(0, 0, 0.9)]),
                make_pano('p2', 0, 10, [(offset, 0, 0.9)])]
    merged, _, _ = fs.fuse(city(2.0), fs.FuseParams())
    assert len(merged) == 1
    split, _, _ = fs.fuse(city(3.9), fs.FuseParams())
    assert len(split) == 2


def test_parallel_rays_interpolate_never_extrapolate():
    # Two panos behind one another looking the same way: nearly parallel rays.
    # The fused position must sit between the member estimates (GLS mean), with
    # covariance still elongated along the ray.
    panos = [make_pano('p1', 0, -18, [(0, 0, 0.9)], heading_deg=0.0),
             make_pano('p2', 0, -12, [(0, 0.8, 0.9)], heading_deg=0.0)]
    sites, frame, _ = fs.fuse(panos, fs.FuseParams())
    assert len(sites) == 1
    site = sites[0]
    lo = min(enu_of(frame, 0, 0)[1], enu_of(frame, 0, 0.8)[1])
    hi = max(enu_of(frame, 0, 0)[1], enu_of(frame, 0, 0.8)[1])
    assert lo - 1e-6 <= site.n <= hi + 1e-6
    var_e, _, var_n = site.cov_p
    assert var_n > var_e  # along-ray (north) stays the uncertain axis


@pytest.mark.parametrize('crowd_source', ['mapillary', 'panoramax'])
def test_crowdsourced_records_get_the_wider_error_model(crowd_source):
    gsv = make_pano('g', 0, -10, [(0, 0, 0.9)], source='launch')
    mly = make_pano('m', 0, 10, [(0, 0, 0.9)], source=crowd_source)
    dets, _, _ = fs.project([gsv, mly], fs.FuseParams())
    cov = {d.pano_id: d.cov for d in dets}
    assert cov['m'][0] + cov['m'][2] > cov['g'][0] + cov['g'][2]


def _record_line(p):
    return json.dumps({
        'detections': [{'x_normalized': x, 'y_normalized': y, 'confidence': c}
                       for _, x, y, c in p.detections],
        'label_type': 'CurbRamp',
        'pano': {'panorama_id': p.pano_id, 'lat': p.lat, 'lng': p.lng,
                 'camera_heading': p.camera_heading,
                 'camera_pitch': p.camera_pitch, 'camera_roll': p.camera_roll,
                 'capture_date': p.capture_date, 'source': p.source}})


def _demo_city():
    panos = []
    for i in range(6):
        e = 25.0 * i
        panos.append(make_pano(f'a{i}', e, -10, [(e, 0, 0.9), (e + 3, 0, 0.6)]))
        panos.append(make_pano(f'b{i}', e, 10, [(e, 0, 0.7), (e + 1, 4, 0.3)]))
    return panos


def test_shuffled_input_produces_byte_identical_sites(tmp_path):
    lines = [_record_line(p) for p in _demo_city()]
    outputs = []
    for name, ordering in (('fwd', lines), ('rev', list(reversed(lines)))):
        src = tmp_path / f'{name}.jsonl'
        src.write_text('\n'.join(ordering) + '\n', encoding='utf-8')
        panos, skipped = fs.load_results(src)
        assert skipped == 0
        sites, frame, stats = fs.fuse(panos, fs.FuseParams())
        out = tmp_path / f'{name}_sites.jsonl'
        fs.write_sites(sites, frame, stats, fs.FuseParams(), out,
                       tmp_path / f'{name}_meta.json')
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1]


def test_load_results_needs_no_manifest_and_site_json_round_trips(tmp_path):
    src = tmp_path / 'results.jsonl'  # a bare run dir: no manifest.json anywhere
    src.write_text('\n'.join(_record_line(p) for p in _demo_city()) + '\n',
                   encoding='utf-8')
    panos, _ = fs.load_results(src)
    sites, frame, _ = fs.fuse(panos, fs.FuseParams())
    line = json.loads(json.dumps(fs.site_to_json(sites[0], frame)))
    assert line['site_id'] == 0
    assert line['n_members'] == len(sites[0].members)
    member = line['members'][0]
    src_pano = next(p for p in panos if p.pano_id == member['pano_id'])
    _, x, y, conf = src_pano.detections[member['det_index']]
    assert (member['x_normalized'], member['y_normalized']) == (x, y)
    assert member['confidence'] == conf
    assert isinstance(member['in_refit'], bool)


# --- camera height (issue #40)

def test_per_pano_heights_reunite_what_a_constant_splits():
    # Two rigs at different real heights see one ramp. Under the 2.6 m constant their
    # ground points disagree; with each pano's measured height they coincide.
    panos = [make_pano('low', -10, 0, [(0, 0, 0.9)], height=1.8),
             make_pano('high', 0, -12, [(0, 0, 0.9)], height=2.5)]
    fixed, _, _ = fs.fuse(panos, fs.FuseParams())
    per_pano, frame, stats = fs.fuse(panos, fs.FuseParams(camera_height_m=geo.PER_PANO))
    assert len(per_pano) == 1 and len(per_pano[0].pano_ids) == 2
    te, tn = enu_of(frame, 0, 0)
    assert math.hypot(per_pano[0].e - te, per_pano[0].n - tn) < 0.05
    assert stats['camera_heights'] == {'measured': 2, 'fallback': 0}
    spread = max(math.hypot(d.e - te, d.n - tn) for s in fixed for d, _ in s.members)
    assert spread > 2.0


def test_implied_heights_recover_the_true_rig_height():
    # Bearing-only triangulation does not depend on the height the raycast assumed.
    panos = [make_pano('a', -10, 0, [(0, 0, 0.9)], height=2.05),
             make_pano('b', 0, -12, [(0, 0, 0.9)], height=2.05)]
    sites, frame, _ = fs.fuse(panos, fs.FuseParams(camera_height_m=geo.PER_PANO))
    implied = fs.implied_heights(sites, frame, {p.pano_id: p for p in panos})
    assert implied['a'] == [pytest.approx(2.05)] and implied['b'] == [pytest.approx(2.05)]


def _line_with_block(p, **block):
    rec = json.loads(_record_line(p))
    rec['pano'].update(block)
    return json.dumps(rec)


def test_load_results_reads_heights_from_the_block_or_the_harvested_index(tmp_path):
    a, b, c = (make_pano(n, 0, i * 20.0, [(5, i * 20.0, 0.9)]) for i, n in enumerate('abc'))
    src = tmp_path / 'results.jsonl'
    src.write_text('\n'.join([
        _line_with_block(a, camera_height_m=1.9, camera_height_spread_m=0.1,
                         camera_height_status='measured'),
        # a post-#40 null is final, even though the index below measured this pano
        _line_with_block(b, camera_height_m=None, camera_height_status='synthetic_ground'),
        _record_line(c),                               # pre-#40: no keys, so the index
    ]) + '\n', encoding='utf-8')
    (tmp_path / 'depth').mkdir()
    (tmp_path / 'depth' / 'index.csv').write_text(
        'panorama_id,degenerate,camera_height_m,ground_tilt_deg,height_spread_m\n'
        'b,0,2.2,1.5,0.1\n'
        'c,0,2.1,1.4,0.2\n', encoding='utf-8')
    by_id = {p.pano_id: p for p in fs.load_results(src)[0]}
    assert (by_id['a'].camera_height_m, by_id['a'].camera_height_spread_m) == (1.9, 0.1)
    assert by_id['b'].camera_height_m is None
    assert (by_id['c'].camera_height_m, by_id['c'].camera_height_spread_m) == (2.1, 0.2)


def test_harvested_stand_in_grounds_are_not_heights(tmp_path):
    path = tmp_path / 'index.csv'
    path.write_text('panorama_id,degenerate,camera_height_m,ground_tilt_deg,height_spread_m\n'
                    'stand_in,0,2.5000,0.000,0.0000\n'
                    'degenerate,1,2.5000,0.000,0.0000\n'
                    'wrong_plane,0,0.0407,11.871,0.0000\n'
                    'real,0,1.9154,1.482,0.2318\n', encoding='utf-8')
    assert fs.load_depth_index(path) == {'real': (1.9154, 0.2318)}
