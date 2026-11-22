#!/bin/bash
# Native-res archive of the Richmond run onto the makelab pool.
# Pixels were fetched 2026-07-22 (9,091 native 11000x5500 panos, 23 GB) but never
# reconciled; this pass fetches anything missing and writes index.csv + decayed.txt.
set -u
cd /projects/makeabilitylab/sidewalk-auto-labeler || exit 1
city=richmond
echo "=== START $city $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
.venv/bin/python scripts/export_benchmark.py "runs/$city/results.jsonl" \
    --out "runs/$city/panos" >> "runs/$city/fetch.log" 2>&1
rc=$?
echo "=== END $city rc=$rc $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
echo "RICHMOND_DONE rc=$rc $(date -u +%FT%TZ)" > "runs/${city}_archive.done"