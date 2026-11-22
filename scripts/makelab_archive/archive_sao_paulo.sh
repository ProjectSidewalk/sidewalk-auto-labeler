#!/bin/bash
# Native-res archive of the São Paulo (Brás + Sé/Cambuci/Bom Retiro) run onto the
# makelab pool. Resumable (export_benchmark skips panos already present),
# self-verifying (writes index.csv + decayed.txt beside panos/ and reconciles 1:1).
# Same shape as archive_gainesville.sh / archive_paterson.sh.
set -u
cd /projects/makeabilitylab/sidewalk-auto-labeler || exit 1
city=sao_paulo
echo "=== START $city $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
.venv/bin/python scripts/export_benchmark.py "runs/$city/results.jsonl" \
    --out "runs/$city/panos" >> "runs/$city/fetch.log" 2>&1
rc=$?
echo "=== END $city rc=$rc $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
echo "SAO_PAULO_DONE rc=$rc $(date -u +%FT%TZ)" > "runs/${city}_archive.done"
