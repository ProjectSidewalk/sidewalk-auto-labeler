# Draft reply to sidewalk-auto-labeler#161 (for Jon to edit and post; NOT posted)

Written as from the maintainer: no tool mentions, no legal framing. The numbers come from
the run manifests, the archive index and PR #173's study. Check the one marked [CHECK]
before posting, and fill in the PR link once it is merged, if you prefer to link a merge.

---

Hi Christian,

Thank you for raising this, and I'm sorry, both for the load we put on the Panoramax
servers and for taking three weeks to reply. You're right that we should have contacted you
before running anything at this scale.

**What we downloaded.** All of it was full-resolution (`hd`) pictures, each with one STAC
item request, from up to 50 parallel workers and with no rate limit:

- **Bayonne**: about 28,500 HD pictures for a first pass at 10 m spacing, then 50,409 for a
  denser 5 m pass, and then the same pictures again to build our local archive of the
  imagery we had processed [CHECK the dates: late September to early October 2026].
- **Lyon**: 86,942 HD pictures for a first pass at 10 m spacing, another 87,583 on
  7 October to densify it to 5 m, and then, on 8 and 9 October, a full re-download of all
  174,405 pictures (at about 1.3 per second) for the archive. That last pass ran after you
  had opened this issue. We hadn't seen it yet, and it made things worse at exactly the
  wrong time.

**What we've changed** (https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/173):

- Every request to Panoramax now goes through a rate limiter: **at most one picture
  download per second**, and one metadata request per second, per process. Tile requests
  are limited to 5 per second. A 429 or 503 now pauses all of our workers for as long as
  your server asks (Retry-After), not just the worker that got the response.
- Metadata is batched through `/search?ids=` at 100 pictures per request, so a city the
  size of Lyon now costs about 1,750 metadata requests instead of 174,000.
- We now use at most 4 workers for Panoramax, down from 50.
- The archive step uses the same limits.

**On SD vs HD.** Thank you for the suggestion; we measured it properly before answering.
On 600 Lyon pictures across nine camera models, we ran our detector on `sd.jpg` and on
the original. On SD it found only about 63% of what it found on HD, and about 15% of its
SD detections weren't found on HD at all. Every camera model lost a similar share. Most of
the loss comes from the detector being less confident at the lower resolution: the model
was trained on full-resolution panoramas. So for now we still detect on HD, at one picture
per second. SD is available as an option, and we'll switch to it once we have a model that
works at that resolution. The full write-up is in the PR.

**Going forward**, we'll get in touch with you before any further large run or archive
(anything over a few thousand pictures), and we're happy to schedule these at times that
suit your servers. Would a bulk channel be better for you, such as a dump, a mirror, or a
particular instance or time window for this kind of batch work? We'd gladly use whatever is
easiest on your side.

Thanks again for Panoramax, and for telling us directly.

Jon Froehlich (Project Sidewalk, University of Washington)
