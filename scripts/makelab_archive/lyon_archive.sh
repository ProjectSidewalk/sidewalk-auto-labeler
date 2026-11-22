#!/bin/bash
# Native-res archive of the Lyon Panoramax run, 5 m (issues #146, #159), per the makelab archive protocol.
cd /projects/makeabilitylab/sidewalk-auto-labeler || exit 1
PY=/homes/gws/jonf/sidewalk-auto-labeler-py312/.venv/bin/python
SCRIPT=/homes/gws/jonf/sidewalk-auto-labeler-py312/scripts/export_benchmark.py
echo "start $(date -u +%FT%TZ)" >> runs/lyon/fetch.log
"$PY" "$SCRIPT" runs/lyon/results.jsonl --out runs/lyon/panos >> runs/lyon/fetch.log 2>&1
rc=$?
echo "rc=$rc $(date -u +%FT%TZ)" > runs/lyon_archive.done
echo "end rc=$rc $(date -u +%FT%TZ)" >> runs/lyon/fetch.log
exit $rc
