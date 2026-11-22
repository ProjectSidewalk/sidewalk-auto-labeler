#!/bin/bash
# Native-res (max zoom) archive of the Bend GSV run onto the makelab pool.
# 78,560 panos; GSV native fetch stitches hundreds of tiles per pano, so expect
# many hours. Resumable (existing files skipped) and self-verifying (index.csv +
# decayed.txt + 1:1 reconcile). GSV panos DO expire as Google refreshes coverage,
# which is why this exists. Same driver shape as archive_richmond/annapolis.
set -u
cd /projects/makeabilitylab/sidewalk-auto-labeler || exit 1
city=bend
echo "=== START $city $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
.venv/bin/python scripts/export_benchmark.py "runs/$city/results.jsonl" \
    --out "runs/$city/panos" >> "runs/$city/fetch.log" 2>&1
rc=$?
echo "=== END $city rc=$rc $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
echo "BEND_DONE rc=$rc $(date -u +%FT%TZ)" > "runs/${city}_archive.done"