"""Imagery-source registry.

Each source module provides the same interface, consumed by main.py:

- NAME: canonical source name (matches Project Sidewalk's pano_source enum value
  for the provider, e.g. 'gsv', 'mapillary', 'panoramax').
- COVERAGE_TILE_ZOOM: Slippy-map zoom level for the coverage scan.
- prepare(): fail fast (before the model loads) if the source can't run,
  e.g. a missing API token.
- fetch_panos_for_tile(tile_x, tile_y, area_shape) -> {pano_id: (lat, lon, ...)}
  for panos inside the area, or None if the tile failed after retries. main.py
  reads only the first two tuple elements; sources may append extra data.
- thin_panos(panos) -> panos (OPTIONAL): spatial dedup applied to the combined
  scan result before processing, for sources with near-duplicate coverage.
- fetch_pano(pano_id, lat, lon) ->
  {'status': 'success', 'pano': <JSONL pano dict>, 'image': <PIL image>} or
  {'status': 'skipped'|'failure', 'reason': str}. 'skipped' is deterministic
  (main.py caches it so it's never retried); 'failure' is retryable (left
  uncached so the next run retries it).
- fetch_pano_by_id(pano_id, area_shape) (OPTIONAL): same contract as fetch_pano
  for a pano known only by id (a dangling link target), positioned from its own
  metadata; outside-the-area is a deterministic skip. Providing this hook enables
  main.py's post-run gap-fill phase (link-graph closure, issue #32) — only
  meaningful for sources whose pano records carry links (GSV).

Modules are imported lazily so one source's dependencies (e.g. the MVT decoder
for Mapillary) aren't required to run another.
"""

SOURCE_NAMES = ('gsv', 'mapillary', 'panoramax')

# Every source normalizes imagery to the detector's expected input size (see
# detectors/curb_ramp.py). Lives here — not in panorama.py — so sources that
# don't use streetlevel don't import it (its pyexiv2 dependency needs a newer
# glibc than some clusters have).
TARGET_IMAGE_SIZE = (4096, 2048)


def get_source(name):
    if name == 'gsv':
        from sources import gsv
        return gsv
    if name == 'mapillary':
        from sources import mapillary
        return mapillary
    if name == 'panoramax':
        from sources import panoramax
        return panoramax
    raise ValueError(f"Unknown imagery source: {name!r} (expected one of {SOURCE_NAMES})")
