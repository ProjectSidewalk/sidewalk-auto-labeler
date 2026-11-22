#!/bin/bash
# Native-res archive of Morgantown + Budapest onto the makelab pool.
# Sequential (avoids doubling Mapillary rate-limit contention), resumable
# (export_benchmark skips panos already present), self-verifying (writes
# index.csv + decayed.txt and reconciles 1:1 per city).
set -u
cd /projects/makeabilitylab/sidewalk-auto-labeler || exit 1
for city in morgantown budapest_district5; do
  echo "=== START $city $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
  .venv/bin/python scripts/export_benchmark.py "runs/$city/results.jsonl" \
      --out "runs/$city/panos" >> "runs/$city/fetch.log" 2>&1
  rc=$?
  echo "=== END $city rc=$rc $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
done
echo "ALL_DONE $(date -u +%FT%TZ)" > runs/archive_all.done
