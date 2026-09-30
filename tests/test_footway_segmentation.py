"""footway_segmentation (#47 step 2): the equirect <-> perspective tile mapping, the
stitch vote, the class collapse and the pre-registered verdict. Pure functions only; the
segmenter itself is never loaded."""
import math

import numpy as np
import pytest

import footway_segmentation as fw


def test_tile_centre_lands_at_heading_and_pitch():
    size = 64
    for h, p in ((0, 0), (45, 0), (180, 0), (270, -35), (90, -35)):
        x, y = fw.tile_pixel_to_equirect(h, p, size=size)
        # the four centre pixels straddle the optical axis
        cx = np.mean(np.unwrap(x[31:33, 31:33].ravel() * 2 * math.pi)) / (2 * math.pi)
        assert (cx - (0.5 + h / 360)) % 1.0 == pytest.approx(0, abs=1e-3) or \
            (cx - (0.5 + h / 360)) % 1.0 == pytest.approx(1, abs=1e-3)
        assert y[31:33, 31:33].mean() == pytest.approx(0.5 - p / 180, abs=1e-3)


def test_tile_right_is_image_right_and_top_is_up():
    x, y = fw.tile_pixel_to_equirect(0, 0, size=64)
    assert x[32, 60] > x[32, 4]          # right side of the tile = larger x_norm
    assert y[4, 32] < y[60, 32]          # top of the tile = smaller y_norm (up)
    assert x[32, 63] == pytest.approx(0.5 + 45 / 360, abs=0.01)   # 90 deg FOV


@pytest.mark.parametrize('h,p', [(0, 0), (135, 0), (180, -35), (315, -35)])
def test_round_trip_tile_to_equirect_to_tile(h, p):
    size = 128
    x, y = fw.tile_pixel_to_equirect(h, p, size=size)
    col, row, valid, angle = fw.equirect_to_tile(x, y, h, p, size=size)
    assert valid.all()
    rr, cc = np.mgrid[0:size, 0:size]
    assert (col == cc).all() and (row == rr).all()
    assert angle[size // 2, size // 2] < math.radians(1)


def test_equirect_to_tile_rejects_behind_and_outside():
    col, row, valid, _ = fw.equirect_to_tile(np.array([0.0, 0.5, 0.5]),
                                             np.array([0.5, 0.5, 0.05]), 0, 0)
    assert list(valid) == [False, True, False]   # behind; centre; 72 deg up


def test_sample_bilinear_wraps_the_seam():
    img = np.zeros((4, 8), dtype=np.float64)
    img[:, 0] = 10.0          # first column
    img[:, 7] = 20.0          # last column
    # exactly on the seam: halfway between the last and the first column
    assert fw.sample_bilinear(img, np.array([0.0]), np.array([0.5]))[0] == pytest.approx(15)
    assert fw.sample_bilinear(img, np.array([1.0]), np.array([0.5]))[0] == pytest.approx(15)


def test_render_then_stitch_recovers_equirect_labels():
    """A labelled equirect rendered to tiles and voted back returns its own labels
    wherever a tile sees the pixel (nearest-pixel label transfer, small grid)."""
    gw, gh = 128, 64
    x, y = fw.grid_centres(gw, gh)
    truth = ((np.floor(x * 8) + 8 * np.floor(y * 4)) % 250).astype(np.uint8)  # 32 blocks
    tiles = {}
    for name, h, p in fw.tile_specs():
        tx, ty = fw.tile_pixel_to_equirect(h, p, size=256)
        c = np.minimum((tx * gw).astype(int), gw - 1)
        r = np.minimum((ty * gh).astype(int), gh - 1)
        tiles[name] = truth[r, c]
    geom = fw.StitchGeometry(gw, gh, tile_size=256)
    lab, votes = fw.vote(tiles, geom)
    seen = geom.n_cover > 0
    # block edges can flip by one pixel under nearest sampling; the interior must match
    assert (lab[seen] == truth[seen]).mean() > 0.97
    assert (lab[~seen] == fw.NONE_LABEL).all()
    # only the zenith (above the pitch-0 row, irrelevant below the horizon) and the far
    # nadir go unseen; the whole band from 40 deg up to 70 deg down is covered
    rows = (np.nonzero((~seen).any(axis=1))[0] + 0.5) / gh
    assert ((rows < 0.5 - 40 / 180) | (rows > 0.5 + 70 / 180)).all()


def test_vote_tie_goes_to_the_nearest_tile():
    class G:
        pass
    g = G()
    g.names = ['a', 'b']
    g.shape = (1, 1)
    g.flat = np.array([[0], [0]])
    g.valid = np.array([[True], [True]])
    g.order = np.array([[1], [0]])          # tile b is nearer in angle
    lab, votes = fw.vote({'a': np.array([[3]], np.uint8), 'b': np.array([[7]], np.uint8)}, g)
    assert lab[0, 0] == 7 and votes[0, 0] == 1


def test_collapse_covers_every_vistas_name_and_refuses_unknowns():
    names = list(fw.COLLAPSE)
    table = fw.collapse_table(dict(enumerate(names)))
    assert fw.SEG_GROUPS[table[names.index('Curb Cut')]] == fw.WALK
    assert fw.SEG_GROUPS[table[names.index('Car')]] == fw.OBJECT
    assert fw.SEG_GROUPS[table[fw.NONE_LABEL]] == fw.NONE
    with pytest.raises(SystemExit):
        fw.collapse_table({0: 'Sidewalk', 1: 'Hoverboard'})


def test_verdict_boundaries():
    assert fw.verdict(700, 70, 42, 30)['rule_c'] == 'SUPPORTED'
    assert fw.verdict(700, 70, 42, 10)['rule_c'] == 'NOT SUPPORTED'     # gap too small
    v = fw.verdict(700, 70, 42, 14)                                       # gap 0.233
    assert v['rule_c'] == 'NOT SUPPORTED' or v['false_lo'] > v['true_hi']
    assert fw.verdict(700, 70, 42, 30)['underpowered'] is True           # n=42 CI ~0.27 wide
    assert fw.verdict(700, 70, 4000, 3000)['underpowered'] is False
