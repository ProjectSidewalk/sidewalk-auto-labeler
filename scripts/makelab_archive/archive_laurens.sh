#!/bin/bash
# Native-res archive of the Laurens run onto the makelab pool -- BOTH imagery arms.
# Laurens is the only city with a GSV and a Mapillary run over one footprint
# (area hash d8dd392b...), so both are archived side by side and the GSV arm is
# what RampNet#151 needs to separate ramp geometry from rig and season.
# Resumable (export_benchmark skips panos already present), self-verifying
# (writes index.csv + decayed.txt beside panos/ and reconciles 1:1).
set -u
cd /projects/makeabilitylab/sidewalk-auto-labeler || exit 1
for city in laurens laurens_gsv; do
    echo "=== START $city $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
    .venv/bin/python scripts/export_benchmark.py "runs/$city/results.jsonl" \
        --out "runs/$city/panos" >> "runs/$city/fetch.log" 2>&1
    rc=$?
    echo "=== END $city rc=$rc $(date -u +%FT%TZ) ===" >> "runs/$city/fetch.log"
    echo "${city}_DONE rc=$rc $(date -u +%FT%TZ)" > "runs/${city}_archive.done"
done
