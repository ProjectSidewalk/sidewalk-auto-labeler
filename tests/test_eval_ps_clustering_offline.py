"""eval_ps_clustering's offline machinery (issue #106): the blocked PS partition, label
synthesis from results.jsonl, region assignment and the unplaceable-label attach rule.

Needs pandas/scipy/haversine (analysis-only, not in requirements-test.txt), so the file
skips where they are missing (CI)."""
import pytest

pytest.importorskip('pandas')
pytest.importorskip('scipy')
pytest.importorskip('haversine')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eval_ps_clustering as epc  # noqa: E402

LAT0, LNG0 = 37.54, -77.43


def _partition_sets(df, arr):
    groups = {}
    for lid, c in zip(df['label_id'].to_numpy(), arr):
        groups.setdefault(int(c), set()).add(int(lid))
    return {frozenset(v) for v in groups.values()}


def test_blocked_partition_equals_dense():
    rng = np.random.default_rng(106)
    n = 300
    # clumps of labels along a few streets, so blocks are neither all singletons nor one
    centres = rng.uniform(0, 400, size=(40, 2))
    pts = centres[rng.integers(0, 40, n)] + rng.normal(0, 4.0, size=(n, 2))
    lat = LAT0 + pts[:, 1] / 111320.0
    lng = LNG0 + pts[:, 0] / (111320.0 * np.cos(np.radians(LAT0)))
    panos = [f'p{i // 3}' for i in range(n)]
    panos[1] = panos[0]            # same (user, pano) pair 0.1 m apart: cannot-link
    lat[1], lng[1] = lat[0] + 1e-6, lng[0]
    df = pd.DataFrame({'label_id': np.arange(n), 'user_id': 'ai', 'pano_id': panos,
                       'region_id': rng.integers(0, 3, n), 'lat': lat, 'lng': lng})
    ts = [0.0025, 0.0075, 0.0125, 0.015]
    for per_region in (True, False):
        dense = epc.ps_partition(df, ts, per_region, blocked=False)
        stats = {}
        blocked = epc.ps_partition(df, ts, per_region, blocked=True, stats=stats)
        assert stats['largest_block'] < n
        for t in ts:
            assert _partition_sets(df, dense[t]) == _partition_sets(df, blocked[t])
            assert blocked[t][0] != blocked[t][1]     # the cannot-link held
