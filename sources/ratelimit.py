"""A small thread-safe rate limiter for imagery sources (issue #161).

Panoramax is run by volunteers and public agencies on modest servers, and its maintainers
asked for at most about one full-resolution download per second. Before #161 the source
made two requests per pano from 50 threads with no limit at all, so the pace was set by how
fast the servers could answer -- which is exactly what hurt them. A limiter per *kind* of
request (pictures, metadata, tiles) puts the pace in our hands instead.

`RateLimiter(per_second, burst=1)` is a token bucket written in the GCRA form: one
"theoretical arrival time" advances by 1/per_second per request, and a request may go once
the clock has caught up with it (less `burst - 1` intervals of credit). `wait()` blocks the
calling thread until its turn and returns how long it slept. Every attempt pays, retries
included -- a retry is a request to the server like any other.

`pause(seconds)` holds back *every* waiter until the pause ends. A 429 or 503 is the server
saying it is overloaded, and the answer to that is for the whole pool to back off, not just
the one thread that happened to receive it. Sleeping threads re-check after they wake, so a
pause that starts while they sleep still applies to them.

The clock and the sleep are injectable so the tests can drive the limiter with a fake clock
and never actually sleep:

    >>> t = [0.0]
    >>> lim = RateLimiter(2.0, clock=lambda: t[0],
    ...                   sleep=lambda s: t.__setitem__(0, t[0] + s))
    >>> [round(lim.wait(), 2) for _ in range(3)]   # 2/s: the second and third wait 0.5 s each
    [0.0, 0.5, 0.5]
    >>> lim.pause(10); round(lim.wait(), 2)
    10.0

A limiter built with `per_second=None` (or <= 0) is unlimited: `wait()` returns at once.
Limits are per *process*: two processes against the same server each get the full rate.
"""
import threading
import time


class RateLimiter:
    def __init__(self, per_second, burst=1, clock=time.monotonic, sleep=time.sleep):
        self.per_second = per_second if per_second and per_second > 0 else None
        self.burst = max(1, int(burst))
        self._interval = 1.0 / self.per_second if self.per_second else 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tat = None            # theoretical arrival time of the next request
        self._paused_until = None
        self.requests = 0           # requests let through (for the manifest's run entry)
        self.waited_s = 0.0         # total time threads spent waiting here
        self.pauses = 0             # pause() calls (429/503 answers)

    @property
    def unlimited(self):
        return self.per_second is None

    def wait(self):
        """Blocks until this caller may send one request; returns the seconds slept."""
        slept = 0.0
        while True:
            with self._lock:
                now = self._clock()
                ready = now
                if self._tat is not None and not self.unlimited:
                    ready = max(ready, self._tat - (self.burst - 1) * self._interval)
                if self._paused_until is not None:
                    ready = max(ready, self._paused_until)
                if now >= ready:
                    if not self.unlimited:
                        self._tat = max(self._tat if self._tat is not None else now, now) \
                            + self._interval
                    self.requests += 1
                    self.waited_s += slept
                    return slept
                delay = ready - now
            # Sleep outside the lock, then re-check: a pause may have begun meanwhile.
            self._sleep(delay)
            slept += delay

    def pause(self, seconds):
        """Holds every waiter back for `seconds` from now (never shortens a longer pause)."""
        if seconds is None or seconds <= 0:
            return
        with self._lock:
            until = self._clock() + seconds
            if self._paused_until is None or until > self._paused_until:
                self._paused_until = until
            self.pauses += 1

    def stats(self):
        return {'per_second': self.per_second, 'requests': self.requests,
                'waited_s': round(self.waited_s, 1), 'pauses': self.pauses}
