# Draft reply to sidewalk-auto-labeler#161 (for Jon to edit and post; NOT posted)

Written as from the maintainer: no tool mentions, no legal framing. The numbers come from
the run manifests, the archive index and PR #173's study. The Bayonne dates were resolved from the run records (10 m pass merged 2026-10-04, 5 m
densify + archive 2026-10-07). Fill in the PR link once it is merged, if you prefer to link a merge.

Timing against the issue (opened 2026-10-07 14:52Z = 07:52 PT), from memory notes
`dispatch-2026-10-07.md` and `lyon-panoramax-run.md`: the Bayonne 5 m archive was still
running at 12:13 PT and finished 19:13Z; the Lyon densify (Hyak job 41504145) had an ETA of
~22:30Z and its results.jsonl landed on the file server at 00:18Z on 8 Oct, when the Lyon
archive started. So parts of both 7 October pulls ran after the issue was opened, and the
draft now concedes that. [CHECK] The 7 Oct notes call the Bayonne archive "of 21,885
panos" and later "50,409/50,409 verified": if only the 21,885 new pictures were downloaded
(the 10 m ones already archived), say "the 21,900 new pictures" instead of "all 50,409".

---

Hi Christian,

Thank you for raising this, and I'm sorry, both for the load we put on the Panoramax
servers and for taking three weeks to reply. You're right that we should have contacted you
before running anything at this scale.

**What we downloaded.** All of it was full-resolution (`hd`) pictures, each with one STAC
item request, from up to 50 parallel workers and with no rate limit:

- **Bayonne**: about 28,500 HD pictures for a first pass at 10 m spacing in late September,
  about 21,900 more on 7 October to densify it to 5 m, and then, the same day, all 50,409
  again to build our local archive of the imagery we had processed.
- **Lyon**: about 87,000 HD pictures for a first pass at 10 m spacing, about 87,500 more
  on 7 October to densify it to 5 m (174,405 in all), and then, on 8 and 9 October, a
  full re-download of all 174,405 pictures (at about 1.3 per second) for the archive.

Parts of the two 7 October pulls, and all of the Lyon archive pass, ran after you had
opened this issue. We hadn't seen it yet, and it made things worse at exactly the wrong
time.

**What we've changed** (https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/173):

- Every request to Panoramax now goes through a rate limiter: **at most one picture
  download per second**, and one metadata request per second, per process. Tile requests
  are limited to 5 per second. A 429 or 503 now pauses all of our workers for as long as
  your server asks (Retry-After), up to five minutes, not just the worker that got the
  response.
- Metadata is batched through `/search?ids=` at 100 pictures per request, so a city the
  size of Lyon now costs about 1,750 metadata requests instead of 174,000.
- That limit of one picture per second applies everywhere we download from Panoramax,
  including the archive step.

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
