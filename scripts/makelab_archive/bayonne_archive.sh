#!/bin/bash
# Native-res archive of the Bayonne Panoramax run (issue #57), per the makelab archive protocol.
cd /projects/makeabilitylab/sidewalk-auto-labeler || exit 1
PY=/homes/gws/jonf/sidewalk-auto-labeler-py312/.venv/bin/python
SCRIPT=/homes/gws/jonf/sidewalk-auto-labeler-py312/scripts/export_benchmark.py
echo "start $(date -u +%FT%TZ)" >> runs/bayonne/fetch.log
"$PY" "$SCRIPT" runs/bayonne/results.jsonl --out runs/bayonne/panos >> runs/bayonne/fetch.log 2>&1
rc=$?
echo "rc=$rc $(date -u +%FT%TZ)" > runs/bayonne_archive.done
echo "end rc=$rc $(date -u +%FT%TZ)" >> runs/bayonne/fetch.log
exit $rc
