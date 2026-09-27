"""Micro-batching for the detector's forward pass (issue #2). Stdlib only, no torch.

Many fetch threads call ``CurbRampDetector.detect`` at once. With ``batch_size == 1`` each
call takes the inference lock and runs its own forward pass. With ``batch_size > 1`` the
callers instead hand their preprocessed tensors to a ``Batcher``: one consumer thread
collects up to ``batch_size`` items, waiting at most ``batch_wait_s`` after the first one
arrives (so a lone pano is never stalled for long), runs ONE forward over all of them and
hands each caller its own output.

The forward pass itself is injected (``run_batch``), so this module knows nothing about
torch and the test suite can drive it with a fake model (tests/test_batching.py).

Usage::

    def run_batch(items):            # list in -> list of the same length out
        return [x * 2 for x in items]

    with Batcher(run_batch, batch_size=4, batch_wait_s=0.1) as b:
        b.submit(21)                 # -> 42, from whichever thread calls it
        b.stats()                    # -> {'forward_passes': 1, 'images': 1, ...}

Failure semantics:

- an exception raised by ``run_batch`` (e.g. CUDA out-of-memory) is re-raised in every
  caller of THAT batch; the consumer keeps going and later batches are unaffected;
- if the consumer thread itself dies, every pending and future caller gets
  ``BatcherDeadError`` instead of waiting forever;
- ``close()`` drains: everything submitted before it is still run; ``submit`` after it
  raises ``BatcherClosedError``. The consumer is a daemon thread, so an unclosed batcher
  never keeps the process alive at exit.
"""
import queue
import threading
import time

DEFAULT_BATCH_WAIT_S = 0.1

# How often a waiting caller re-checks that the consumer is still alive.
_LIVENESS_POLL_S = 1.0

_STOP = object()


class BatcherClosedError(RuntimeError):
    """submit() after close()."""


class BatcherDeadError(RuntimeError):
    """The consumer thread is gone, so this item will never be run."""


class ForwardStats:
    """Thread-safe tally of forward passes; shared by the batched and unbatched paths.

    ``snapshot()`` keys: forward_passes, images, batch_size, mean_batch (images per
    forward), mean_batch_fill (mean_batch / batch_size; 1.0 = every batch full),
    forward_seconds (wall time inside the forward, i.e. the serialized part),
    failed_batches.
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self.forward_passes = 0
        self.images = 0
        self.forward_seconds = 0.0
        self.failed_batches = 0

    def record(self, n_images, seconds, failed=False):
        with self._lock:
            self.forward_passes += 1
            self.images += n_images
            self.forward_seconds += seconds
            self.failed_batches += bool(failed)

    def snapshot(self):
        with self._lock:
            mean = self.images / self.forward_passes if self.forward_passes else 0.0
            return {'forward_passes': self.forward_passes, 'images': self.images,
                    'batch_size': self.batch_size, 'mean_batch': round(mean, 3),
                    'mean_batch_fill': round(mean / self.batch_size, 3),
                    'forward_seconds': round(self.forward_seconds, 3),
                    'failed_batches': self.failed_batches}


BATCH_SIZE_HELP = (
    "images per forward pass (default 1: unbatched, the pre-#2 path). Each queued image is "
    "one 3x2048x4096 float32 tensor (~100 MB) until its batch runs, and the stacked batch "
    "is a second copy on the device; that is on top of the up-to-<workers> decoded panos "
    "already in RAM. A lone pano waits at most "
    f"{DEFAULT_BATCH_WAIT_S} s for its batch to fill. The end-of-pass detector line reports "
    "the mean batch fill and panos/s, so measure before raising it.")


def report_detector(detector, wall_seconds=None, panos=None):
    """Print the detector's stats line; silent for a detector without stats() (test stubs)."""
    stats = getattr(detector, 'stats', None)
    if callable(stats):
        print('-> ' + format_stats(stats(), wall_seconds, panos))


def format_stats(stats, wall_seconds=None, panos=None):
    """One human-readable line for the end of a pass, e.g.
    ``detector: 12 forward pass(es), 40 image(s), mean batch 3.33/4 (fill 0.83), 51.2 s in
    forward; 0.78 panos/s over 51.6 s``."""
    line = (f"detector: {stats['forward_passes']} forward pass(es), {stats['images']} "
            f"image(s), mean batch {stats['mean_batch']}/{stats['batch_size']} "
            f"(fill {stats['mean_batch_fill']}), {stats['forward_seconds']} s in forward")
    if stats['failed_batches']:
        line += f", {stats['failed_batches']} failed batch(es)"
    if wall_seconds is not None and panos is not None:
        rate = panos / wall_seconds if wall_seconds > 0 else 0.0
        line += f"; {rate:.3f} panos/s over {wall_seconds:.1f} s"
    return line


def make_batcher(run_batch, batch_size, batch_wait_s=DEFAULT_BATCH_WAIT_S):
    """A started Batcher for ``batch_size > 1``; None for 1, which starts no thread and
    leaves the caller on its unbatched path."""
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    return Batcher(run_batch, batch_size, batch_wait_s) if batch_size > 1 else None


class _Pending:
    __slots__ = ('item', 'done', 'result', 'error')

    def __init__(self, item):
        self.item = item
        self.done = threading.Event()
        self.result = None
        self.error = None

    def fail(self, error):
        self.error = error
        self.done.set()


class Batcher:
    """Collects items from many threads into batches for one ``run_batch`` consumer.

    ``run_batch(items)`` must return a sequence with one output per item, in order.
    """

    def __init__(self, run_batch, batch_size, batch_wait_s=DEFAULT_BATCH_WAIT_S,
                 name='detector-batcher'):
        if batch_size < 2:
            raise ValueError('a Batcher needs batch_size >= 2; batch_size 1 runs unbatched')
        if batch_wait_s < 0:
            raise ValueError('batch_wait_s must be >= 0')
        self._run_batch = run_batch
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_s
        self._stats = ForwardStats(batch_size)
        self._queue = queue.Queue()
        self._state_lock = threading.Lock()  # orders submit() against close()
        self._closed = False
        self._death = None                   # the exception that killed the consumer
        self._thread = threading.Thread(target=self._consume, name=name, daemon=True)
        self._thread.start()

    # --------------------------------------------------------------- caller side
    def submit(self, item):
        """Queue ``item``, block until its batch has run, return its output (or raise)."""
        pending = _Pending(item)
        with self._state_lock:
            if self._death is not None:
                raise BatcherDeadError('detector batcher consumer died') from self._death
            if self._closed:
                raise BatcherClosedError('detector batcher is closed')
            self._queue.put(pending)
        while not pending.done.wait(_LIVENESS_POLL_S):
            if not self._thread.is_alive() and not pending.done.is_set():
                raise BatcherDeadError('detector batcher consumer exited without '
                                       'running this item') from self._death
        if pending.error is not None:
            raise pending.error
        return pending.result

    def close(self, timeout=None):
        """Stop accepting items, run everything already queued, then stop the consumer.
        Idempotent."""
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(_STOP)  # FIFO: lands behind every accepted item
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    @property
    def alive(self):
        return self._thread.is_alive()

    def stats(self):
        return self._stats.snapshot()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------- consumer side
    def _next_batch(self):
        """Block for the first item, then fill until batch_size or the wait bound.
        Returns (batch, stop_seen)."""
        first = self._queue.get()
        if first is _STOP:
            return [], True
        batch = [first]
        deadline = time.monotonic() + self.batch_wait_s
        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            try:
                nxt = (self._queue.get(timeout=remaining) if remaining > 0
                       else self._queue.get_nowait())
            except queue.Empty:
                break
            if nxt is _STOP:
                return batch, True
            batch.append(nxt)
        return batch, False

    def _run(self, batch):
        t0 = time.perf_counter()
        try:
            outputs = list(self._run_batch([p.item for p in batch]))
            if len(outputs) != len(batch):
                raise RuntimeError(f'run_batch returned {len(outputs)} outputs for '
                                   f'{len(batch)} items')
        except Exception as e:  # noqa: BLE001 -- goes to every caller of this batch
            self._stats.record(len(batch), time.perf_counter() - t0, failed=True)
            for p in batch:
                p.fail(e)
            return
        self._stats.record(len(batch), time.perf_counter() - t0)
        for p, out in zip(batch, outputs):
            p.result = out
            p.done.set()

    def _consume(self):
        batch = []
        try:
            stop = False
            while not stop:
                batch, stop = self._next_batch()
                if batch:
                    self._run(batch)
                batch = []
        except BaseException as e:  # the consumer itself broke: nobody may wait forever
            with self._state_lock:
                self._death = e
                self._closed = True
            dead = BatcherDeadError('detector batcher consumer died')
            dead.__cause__ = e
            for p in batch:
                p.fail(dead)
            while True:
                try:
                    p = self._queue.get_nowait()
                except queue.Empty:
                    break
                if p is not _STOP:
                    p.fail(dead)
