"""detectors.batching.Batcher with a fake model (issue #2). Torch-free: runs in CI.

The real-model equivalence check is tests/test_curb_ramp_batching_equivalence.py (opt-in).
"""
import threading
import time

import pytest

from detectors.batching import (Batcher, BatcherClosedError, BatcherDeadError,
                                format_stats, make_batcher)


class FakeModel:
    """run_batch stand-in: doubles each item, records batch sizes, optionally gated."""

    def __init__(self, gate=None, fail_on=None):
        self.batches = []
        self.gate = gate
        self.fail_on = fail_on
        self.entered = threading.Event()

    def __call__(self, items):
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), 'test gate never released'
        self.batches.append(list(items))
        if self.fail_on is not None and self.fail_on in items:
            raise ValueError('forward failed')
        return [x * 2 for x in items]


def submit_all(batcher, items):
    """Submit each item from its own thread; returns {item: result or exception}."""
    out = {}

    def call(x):
        try:
            out[x] = batcher.submit(x)
        except Exception as e:
            out[x] = e
    threads = [threading.Thread(target=call, args=(x,)) for x in items]
    for t in threads:
        t.start()
    return out, threads


def join(threads):
    for t in threads:
        t.join(10)
        assert not t.is_alive()


def test_batch_fills_to_batch_size_without_waiting_out_the_bound():
    model = FakeModel()
    with Batcher(model, batch_size=4, batch_wait_s=5.0) as b:
        t0 = time.monotonic()
        out, threads = submit_all(b, [1, 2, 3, 4])
        join(threads)
        assert time.monotonic() - t0 < 4.0  # a full batch never waits out the 5 s bound
        assert out == {1: 2, 2: 4, 3: 6, 4: 8}
        assert [sorted(x) for x in model.batches] == [[1, 2, 3, 4]]
        st = b.stats()
    assert (st['forward_passes'], st['images'], st['mean_batch_fill']) == (1, 4, 1.0)


def test_lone_item_goes_after_the_wait_bound():
    model = FakeModel()
    with Batcher(model, batch_size=8, batch_wait_s=0.05) as b:
        t0 = time.monotonic()
        assert b.submit(21) == 42
        elapsed = time.monotonic() - t0
    assert 0.05 <= elapsed < 2.0
    assert model.batches == [[21]]


def test_results_map_back_to_their_callers_under_concurrency():
    model = FakeModel()
    with Batcher(model, batch_size=3, batch_wait_s=0.02) as b:
        out, threads = submit_all(b, list(range(40)))
        join(threads)
        st = b.stats()
    assert out == {i: 2 * i for i in range(40)}
    assert all(1 <= len(batch) <= 3 for batch in model.batches)
    assert sorted(x for batch in model.batches for x in batch) == list(range(40))
    assert st['images'] == 40 and st['forward_passes'] == len(model.batches)
    assert 'forward pass(es), 40 image(s)' in format_stats(st, wall_seconds=2.0, panos=40)


def test_a_failing_forward_fails_only_its_batch():
    model = FakeModel(fail_on=13)
    with Batcher(model, batch_size=4, batch_wait_s=0.0) as b:
        with pytest.raises(ValueError, match='forward failed'):
            b.submit(13)
        assert b.submit(5) == 10  # the consumer survived
        st = b.stats()
    assert st['failed_batches'] == 1 and st['forward_passes'] == 2


def test_close_drains_pending_items_then_refuses():
    gate = threading.Event()
    model = FakeModel(gate=gate)
    b = Batcher(model, batch_size=2, batch_wait_s=0.0)
    first, t_first = submit_all(b, [100])
    assert model.entered.wait(5)          # the consumer is now stuck inside a forward
    rest, t_rest = submit_all(b, [1, 2, 3])
    deadline = time.monotonic() + 5
    while b._queue.qsize() < 3 and time.monotonic() < deadline:
        time.sleep(0.01)                  # all three queued behind the stuck batch
    closer = threading.Thread(target=b.close)
    closer.start()
    gate.set()
    join(t_first + t_rest + [closer])
    assert first == {100: 200} and rest == {1: 2, 2: 4, 3: 6}
    assert not b.alive
    with pytest.raises(BatcherClosedError):
        b.submit(7)
    b.close()  # idempotent


def test_a_dead_consumer_fails_its_waiters_instead_of_hanging():
    class Boom(BaseException):
        """Escapes the per-batch `except Exception`, killing the consumer thread."""

    def run_batch(items):
        raise Boom()
    b = Batcher(run_batch, batch_size=2, batch_wait_s=0.0)
    with pytest.raises(BatcherDeadError):
        b.submit(1)
    with pytest.raises(BatcherDeadError):
        b.submit(2)  # refused up front, not queued forever


def test_batch_size_one_starts_no_consumer_thread():
    before = threading.active_count()
    assert make_batcher(FakeModel(), batch_size=1) is None
    assert threading.active_count() == before
    with pytest.raises(ValueError):
        Batcher(FakeModel(), batch_size=1)
    with pytest.raises(ValueError):
        make_batcher(FakeModel(), batch_size=0)
