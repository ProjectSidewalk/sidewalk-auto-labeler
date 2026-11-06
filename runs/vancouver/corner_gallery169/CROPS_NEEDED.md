# Crops needed: 281 of 281

The crops of this bundle could not be cut here: they come from 279 panos of the Project
Sidewalk pano store on makelab2 (`/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa`), and no local copy of those panos was found. The
gallery must not be rated until the crops exist (`render` refuses while `images.sha256` lacks
them). Everything else (items, snapshot, aerials, report) is built.

## Option A, one command (recommended)

From this worktree on the Windows desktop, with the makelab2 helper working
(`D:\Git\dotfiles\wsl-ssh.ps1 makelab2 check`) and the analysis extras installed
(`pip install pandas scipy haversine`; `split_figures` needs them), re-run the same build
without `--no-crops`.
It keeps `items.jsonl` (same unit list), cuts the crops on makelab2 with the crop worker
(`site_explorer.CROP_WORKER`, staged under `/homes/gws/jonf/.sidewalk_explorer/<hash>` and
removed afterwards), pulls them back as one tarball, then rewrites `images.sha256`,
`snapshot.json` and `report.md` and deletes this file:

```bash
python scripts/corner_gallery.py build --items-from runs/vancouver/corner_labels169/na_discount_sw/sample.csv \
    --corners-full ../posdetect241/runs/vancouver/corner_inventory_posdetect241/corners_full.jsonl \
    --bundle runs/vancouver/corner_gallery169
python scripts/corner_gallery.py render --bundle runs/vancouver/corner_gallery169 --rater jonf
```

## Option B, by hand on makelab2

`crop_job.json` here is the crop worker's job (`panos_root` = the store, each item's `path` is
`<id[:2]>/<id>.jpg`). On makelab2:

```bash
mkdir -p ~/crops169 && cd ~/crops169
# copy crop_job.json here, and the CROP_WORKER string of scripts/site_explorer.py as crop.py
/projects/makeabilitylab/sidewalk-auto-labeler/.venv/bin/python crop.py crop_job.json out
tar -czf crops169.tar.gz -C out .
```

Copy `crops169.tar.gz` back, unpack it into `runs/vancouver/corner_gallery169/crops/`, then run the two commands of
option A (the build finds the crops on disk, cuts nothing, and writes the manifest).
