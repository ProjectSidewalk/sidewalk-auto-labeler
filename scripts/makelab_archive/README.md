# makelab2 native-res archive drivers (verbatim copies)

These are the shell drivers that built the native-resolution pano archives on makelab2 under
`/projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/`. Until 2026-11-22 they lived
only on the lab share, beside the archive, so an archive could not be re-run exactly from
this repo. They are copied here **byte for byte**. `SHA256SUMS` holds the hashes read on the
host, and `.gitattributes` marks the files `-text` so a CRLF checkout cannot rewrite them.

Each driver is a thin wrapper around one command,
`scripts/export_benchmark.py runs/<city>/results.jsonl --out runs/<city>/panos`, appended to
`runs/<city>/fetch.log`, with a `runs/<city>_archive.done` marker holding the exit code.
`export_benchmark.py` does the work: resumable (panos already present are skipped),
verifying (writes `index.csv` + `decayed.txt` and reconciles records 1:1), and non-zero
unless the archive matches the records. The commands to run it by hand are in
[docs/commands.md](../../docs/commands.md) ("Ground truth lives in RampNet; benchmark bundles
and native-res archives").

**Don't run these from a checkout.** They `cd` to the absolute share path, and each calls a
specific interpreter. The first eight use `.venv/bin/python scripts/export_benchmark.py`
relative to the share copy of the repo. That copy's files now carry mtimes up to 2026-08-31,
so the July archives ran an earlier state of it.
The Panoramax two use the Python 3.12 checkout `/homes/gws/jonf/sidewalk-auto-labeler-py312`.
To reproduce an archive exactly, run the same `export_benchmark.py` command from the commit
listed below.

| Driver | Cities | Host mtime (UTC) | Code it ran |
|---|---|---|---|
| `archive_both.sh` | morgantown, budapest_district5 | 2026-07-25 04:37 | share copy's `.venv` |
| `archive_annapolis.sh` | annapolis | 2026-07-28 01:52 | share copy's `.venv` |
| `archive_richmond.sh` | richmond | 2026-07-28 22:32 | share copy's `.venv` |
| `archive_bend.sh` | bend | 2026-07-29 00:01 | share copy's `.venv` |
| `archive_paterson.sh` | paterson | 2026-07-29 20:51 | share copy's `.venv` |
| `archive_gainesville.sh` | gainesville | 2026-07-30 02:53 | share copy's `.venv` |
| `archive_sao_paulo.sh` | sao_paulo | 2026-08-01 13:49 | share copy's `.venv` |
| `archive_laurens.sh` | laurens, laurens_gsv | 2026-08-31 20:02 | share copy's `.venv` |
| `bayonne_archive.sh` | bayonne (10 m pass 2026-10-01, 5 m pass 2026-10-07) | 2026-10-01 05:50 | py312 checkout at `632e819` |
| `lyon_archive.sh` | lyon (2026-10-08 to 10-09) | 2026-10-08 00:18 | py312 checkout at `632e819` |

**The Panoramax two ran `632e819`.** The py312 checkout's reflog shows a clone at `660ccd5`
on 2026-09-28 and one fast-forward to `632e819` on 2026-09-30, and no move after that. On
2026-11-22 its working tree differed from `632e819` only in two run manifests
(`runs/bayonne/manifest.json`, `runs/vancouver/manifest.json`), and `scripts/export_benchmark.py`
was unmodified. For the first eight, the state of the share copy each one ran is not recorded
as a commit; each city's archive `manifest.json` / `fetch.log` is the record of what it did.

**Provenance and trust.** The files were read from the share through `recycle` on
2026-11-22, while the makelab hosts were campus-only after the 2026-11-21 makelab1 incident.
Every host mtime above is before the incident (the newest is 2026-10-08). An mtime can be
reset, so that is weak evidence; each file was also read in full, and each does only what is
described above. `~/verify_archives.sh` in the NFS home is unrelated: it verifies the klone
departed-user tarballs, not these archives.
