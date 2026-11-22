#!/bin/bash
# Native-res archive of the Annapolis run onto the makelab pool.
# Resumable (export_benchmark skips panos already present), self-verifying (writes
# index.csv + decayed.txt beside panos/ and reconciles 1:1). Same shape as the
# archive_both.sh that did Morgantown + Budapest.
set -u
cd /projects/makeabilitylab/sidewalk-auto-labeler || exit 1
city=annapolis
echo "=== START $city $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
.venv/bin/python scripts/export_benchmark.py "runs/$city/results.jsonl" \
    --out "runs/$city/panos" >> "runs/$city/fetch.log" 2>&1
rc=$?
echo "=== END $city rc=$rc $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
echo "ANNAPOLIS_DONE rc=$rc $(date -u +%FT%TZ)" > "runs/${city}_archive.done"
