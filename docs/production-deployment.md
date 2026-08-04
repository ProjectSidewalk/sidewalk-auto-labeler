# Production deployment

How a city actually gets labeled, end to end, on the machines we really use. The
[`README.md`](../README.md) explains *what* the pipeline is and documents every flag;
[`makeability-quickstart.md`](makeability-quickstart.md) is the RA-facing happy path for a
single-box run on the lab GPU server. This document covers the part neither of them does:
the **multi-host production path** — a Slurm GPU cluster (UW Hyak, `klone`), where every
city run from Richmond onward was actually detected, and the lab file server, where all of
their imagery is archived.

> **Paths and account names below are placeholders.** `<netid>`, `<group>`, and
> `<slurm-account>` are per-user and per-lab; substitute your own. Nothing in this
> document is a credential, and none is needed to read it — but don't copy the paths
> literally.

---

## There is no automated deployment — and that is deliberate

Worth stating up front, because it is the single most load-bearing fact about operating
this repo, and it is invisible from the code:

- **No webhooks.** Unlike [`SidewalkWebpage`](https://github.com/ProjectSidewalk/SidewalkWebpage),
  this repo has no webhook of any kind configured on GitHub.
- **No CI/CD deploy.** The only GitHub Actions workflow is
  [`tests.yml`](../.github/workflows/tests.yml) — it runs `pytest` and ships nothing.
  There are no Actions secrets, no environments, no deployments.
- **No branch protection, no rulesets.**
- **No cron jobs and no systemd timers** on any of the hosts.
- **No server-side checkout.** The Project Sidewalk server never pulls this code; it only
  ever receives HTTP POSTs from [`send_to_ps.py`](../send_to_ps.py), which a human runs.

Every stage below is launched by hand. Nothing observes a branch, a tag, or a push.
The practical consequences:

- Merging to the default branch deploys **nothing**. A city run keeps using whatever
  commit is checked out on the host it runs on, until someone updates that checkout.
- Conversely, **renaming the default branch cannot break production.** The only place the
  default branch name appears in this repo is the `push:` trigger in `tests.yml`.
- The flip side: **the checkouts drift.** They routinely sit on whichever
  `add-<city>-area` branch ran last, which is rarely the one you want next. Always check
  `git -C <repo> log --oneline -1` before you launch, and see
  [The branch guard](#the-branch-guard-copy-it-into-every-sbatch) below.

---

## Where each stage runs

| Stage | Host | Why there |
| :--- | :--- | :--- |
| Scope the area (`--scan-only`) | laptop | No GPU needed; it just counts panos |
| Smoke test (25–100 panos) | laptop / lab GPU box | Catch a bad geometry before burning cluster hours |
| **Full detection run** | **Hyak cluster, `ckpt-g2`, 1× L40S** | Free (checkpoint partition), fastest GPU we have access to |
| Native-res imagery archive | **lab file server** | Hundreds of GB per city — far past the cluster's quota (see below) |
| Benchmark bundle → GT review | lab server → [RampNet](https://github.com/ProjectSidewalk/RampNet) | Only this repo can fetch pixels; scoring lives in RampNet |
| Multi-view fusion | laptop | CPU-only, no network — reads a finished `runs/<city>/` |
| Submit to Project Sidewalk | laptop | Just HTTP POSTs to the city's server |

The lab GPU box also has a GPU (A40) and can do the whole detection run itself — that is
what `makeability-quickstart.md` describes, and it is the right choice for a small city or
when the cluster is congested. Hyak is faster and doesn't compete with the lab's
[`sidewalk-ai-api`](https://github.com/ProjectSidewalk/sidewalk-ai-api) service for the A40.

### Compute on the cluster, storage on the lab server

These are two different machines for a reason, and it is not preference — the cluster
**physically cannot hold the output**:

| Cluster storage | Quota | In use |
| :--- | :--- | :--- |
| Shared group scratch | 1 TB, and 1M files | **~87% full, ~80% of the file quota** |
| Per-user home | 10 GB | **100% full** |

A single city's native-res archive runs from 23 GB (Richmond) to 1.2 TB (Bend). Bend alone
exceeds the entire shared group quota; Clovis (846 GB) would consume nearly all of it. The
file-count quota bites just as hard — one pano is one file, so a 35k-pano city is 35k files
against a 1M shared ceiling.

That scratch space is **shared across the whole lab**, so filling it doesn't just fail your
job, it breaks everyone else's. Cluster storage is a **staging area, not a destination**:
finish the run, copy the (small) run directory off, archive the (huge) imagery on the lab
file server, and clean up. Treat anything left on the cluster as transient.

---

## 1. Scope and smoke-test locally

Never send an unscoped geometry to the cluster.

```bash
python main.py example_geojson/<city>.geojson --name <city> --scan-only
```

This loads no model and processes nothing — it reports the pano count and a runtime
estimate. Note that the estimate's throughput constant is calibrated to a local RTX 3070
and is **pessimistic by ~3.5× for an L40S** (São Paulo: estimated 9.4 h, actual 2.7 h).
Use it for order-of-magnitude scoping, not for `--time`.

Then do a real smoke run of a few dozen panos before committing cluster hours — it is the
cheapest way to catch a geometry that is empty, inverted, or in the wrong hemisphere.

The GeoJSON must be a **bare geometry object**, not a `Feature`/`FeatureCollection` —
`shape()` and the area hash both consume it directly. City geometries live on their own
`add-<city>-area` branch until merged.

---

## 2. Get the code onto the cluster — via a git bundle

The cluster has no credentials for our GitHub org, and a new city's geometry usually lives
on an unpushed branch. A **git bundle** solves both: it is a single file containing exactly
the refs you choose, and `git` treats it as a remote.

Throughout this section:

```bash
SAL_ROOT=/gscratch/<group>/<netid>/sidewalk-auto-labeler   # cluster work dir
```

First time (creates `repo/` and the conda env, ~15 min, mostly the CUDA torch wheel):

```bash
# locally — pack the branches you need
git bundle create sal.bundle --all
scp sal.bundle klone:$SAL_ROOT/sal.bundle

# on the cluster — idempotent setup
bash $SAL_ROOT/hyak_sal_setup.sh
```

> ⚠️ **`hyak_sal_setup.sh` ends with a hardcoded `git checkout mapillary-source`** — a
> stale branch from the first Richmond run. A fresh setup therefore leaves you on the
> *wrong* branch. Check out the branch you actually want before submitting, and rely on
> the branch guard below to catch it if you forget.

Updating an existing checkout — an **incremental** bundle is a couple of KB rather than
14 MB, since it carries only the new commits:

```bash
# locally — everything on the city branch the cluster doesn't have yet
git bundle create <city>2.bundle <last-commit-cluster-has>..add-<city>-area
```

```bash
# on the cluster
cd $SAL_ROOT/repo
git fetch ../<city>2.bundle 'refs/heads/*:refs/remotes/origin/*'
git checkout add-<city>-area && git merge --ff-only origin/add-<city>-area
```

Note that `repo`'s `origin` is a **local bundle file**, not GitHub. Nothing on the cluster
tracks the GitHub repo, so nothing on the cluster is affected by anything done to it —
including renaming a branch.

---

## 3. Submit the run (Slurm)

Never run detection on a login node. Hyak reaps heavy login-node processes, and that reap
also kills any shared SSH control master, so a violation breaks other people's sessions too.

The template below is the most refined of our run scripts and the one to copy for a new
city. Every non-obvious line earned its place:

```bash
#!/bin/bash
#SBATCH --job-name=sal-<city>
#SBATCH --account=<slurm-account>
#SBATCH --partition=ckpt-g2          # free checkpoint partition — preemptible
#SBATCH --gpus=l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --requeue                    # REQUIRED on ckpt: restart after preemption
#SBATCH --open-mode=append           # so a requeue doesn't truncate the log
#SBATCH --output=%x_%j.log
set -euo pipefail
WORKDIR="$SAL_ROOT"
export HF_HOME=$WORKDIR/hf_home
export HF_HUB_OFFLINE=1              # see "HF_HUB_OFFLINE" below
source "$WORKDIR/miniforge3/etc/profile.d/conda.sh"
conda activate sal
cd "$WORKDIR/repo"

git rev-parse --abbrev-ref HEAD | grep -qx add-<city>-area \
  || { echo "FATAL: repo is not on add-<city>-area"; exit 1; }

python -c "import requests; requests.get('https://maps.googleapis.com', timeout=10); print('network OK')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python main.py example_geojson/<city>.geojson --name <city>
```

```bash
cd $SAL_ROOT && sbatch sal_<city>.sbatch
```

### Why `ckpt-g2` is safe here

`ckpt` is free but **preemptible** — your job can be killed at any moment and requeued.
That is normally a serious constraint, and here it is nearly free, because the pipeline is
checkpointed at the pano level: `runs/<city>/already_processed.txt` is appended and flushed
line by line, so a requeued job re-runs the ~1 min coverage rescan and then picks up exactly
where it stopped. Gainesville survived a real preemption with zero lost work.

`--requeue` and `--open-mode=append` are what make that true. Without them a preemption
ends the run silently, or the restart truncates the log you would use to diagnose it.

### The branch guard: copy it into every sbatch

The `git rev-parse | grep -qx` line is not ceremony. A requeue can land hours later on a
tree that someone (or a past you) left on a different branch. A run directory is bound to
one geometry hash and one imagery source, so the wrong branch would abort mid-run — or, if
the geometry file merely doesn't exist there, waste the allocation. Failing in the first
second is much cheaper.

### `HF_HUB_OFFLINE=1`

The RampNet weights (~690 MB) are already in `$HF_HOME`. Without this, one run hung for
**over 11 minutes** at startup on an unauthenticated Hub call checking for updates (5 s of
CPU, RSS stuck at 590 MB — it looks like a crash, not a stall). Offline mode uses the cache
directly: same revision, no network dependency, and it fails fast instead of hanging if the
cache is ever incomplete.

### The network preflight

Compute nodes are not guaranteed to reach the imagery APIs. One `requests.get` costs a
second and turns "silently zero results for two hours" into an immediate, obvious failure.

### GSV runs don't end when the progress bar does

A GSV run finishes with an automatic **gap-fill phase** (issue #32): link-target panos that
the run's own records reference but the tile scan never enumerated get fetched by id and
kept if in-area, iterating until the view graph is closed. Two operational consequences:

- Wall clock runs past what `--scan-only` estimated — that estimate covers the main pass
  only. Size `--time` with headroom.
- A run made before this existed can be retrofitted without re-detecting anything:
  `python main.py runs/<city>/area.geojson --name <city> --gap-fill-only`
  (add `--scan-only` to just count the missing targets first).

Each pass is recorded as a `phase: gap_fill` entry in the manifest's run history, so you
can tell from the manifest alone whether a given run's graph was closed.

### Monitoring

```bash
squeue -u $USER
tail -20 $SAL_ROOT/sal-<city>_<jobid>.log
sacct -j <jobid> --format=JobID,JobName%20,Elapsed,State
```

### Measured throughput

São Paulo (Brás + Sé/Cambuci/Bom Retiro, 13.95 km², GSV) — `ckpt-g2`, one L40S, 16 CPUs,
64 GB:

| | |
| :--- | :--- |
| Wall clock | **2 h 39 m** (`COMPLETED`, no preemption) |
| Panos processed | 22,540 (9 failed → retried on a later pass) |
| Effective rate | **≈ 2.4 panos/sec**, including the coverage scan |

For comparison, the lab A40 runs ~1.7 panos/sec while sharing the card with
`sidewalk-ai-api`. Cluster run logs for every city are kept beside the sbatch files in
`$SAL_ROOT`.

---

## 4. Get the run directory off the cluster

Do this promptly — cluster scratch is shared, quota-tight, and not where results live. The
run directory is the deliverable, and it is small (São Paulo's `results.jsonl` is 41 MB for
22.7k panos):

```
runs/<city>/results.jsonl        # one JSON line per successfully processed pano
runs/<city>/manifest.json        # geometry hash, model provenance, source, storage floor
runs/<city>/area.geojson         # exact geometry used
runs/<city>/already_processed.txt
```

Copy `results.jsonl`, `manifest.json`, and `area.geojson` to wherever the next stage runs —
the lab file server for archiving, and your laptop for fusion and submission (`scp`/`rsync`).

Keep `already_processed.txt` with the copy you might resume from; without it a rerun
re-downloads everything.

---

## 5. Archive the imagery at native resolution (lab file server)

This is the step that must happen **soon after** the run: GSV panoramas decay — Google
retires imagery, and once it is gone the pixels are unrecoverable. Bend lost 8 panos this
way. The archive is also RampNet's only source of native-res imagery, since this repo is
the only one that can fetch pixels.

It runs on the lab file server, not the cluster, for the quota reasons in
[Compute on the cluster, storage on the lab server](#compute-on-the-cluster-storage-on-the-lab-server).
The per-city driver scripts live **only on that host**, not in this repo. They are three
lines around one command:

```bash
cd <lab-server>/sidewalk-auto-labeler
.venv/bin/python scripts/export_benchmark.py "runs/<city>/results.jsonl" \
    --out "runs/<city>/panos" >> "runs/<city>/fetch.log" 2>&1
```

Run it under `tmux` — a city takes hours and hundreds of GB. It is **resumable** (panos
already on disk are skipped) and **self-verifying**: it writes `index.csv` and
`decayed.txt` beside `panos/` and reconciles them 1:1 against `results.jsonl`, so
"finished" means "provably complete", and any decayed pano is named explicitly.

Archive sizes to plan for: Richmond 23 GB, Budapest 30 GB, Morgantown 38 GB, São Paulo
369 GB, Paterson 495 GB, Gainesville 626 GB, Clovis 846 GB, Bend 1.2 TB.

---

## 6. Downstream: GT, fusion, submission

**Benchmark bundle for RampNet** — a spatially de-clustered sample rather than the whole
city, for human GT review:

```bash
python scripts/export_benchmark.py runs/<city>/results.jsonl \
    --bundle ../RampNet/benchmark/<city> --sample 100 --empty-sample 25
```

Scoring and GT tooling live in RampNet (`rampnet.validation`, `scripts/gt_gallery.py`) —
see [`docs/adding_a_benchmark_city.md`](https://github.com/ProjectSidewalk/RampNet/blob/main/docs/adding_a_benchmark_city.md)
there. Use `--records-only` when the pixels will be copied from an existing full-city
archive instead of re-fetched.

**Multi-view fusion** (optional today — the stage-4 promotion path is not wired into
submission yet):

```bash
python scripts/fuse_sites.py runs/<city>          # → sites.jsonl
python scripts/eval_sites.py <city>               # → runs/<city>/fusion_eval/report.md
python scripts/site_explorer.py <city>            # visual review
```

**Submit.** This requires a Project Sidewalk instance for the city to already exist — see
the README's "Prerequisites" and [`bend-onboarding.md`](bend-onboarding.md).

```bash
python send_to_ps.py runs/<city>/results.jsonl --dry-run
python send_to_ps.py runs/<city>/results.jsonl --endpoint https://<server>/ai/submitLabelsOnPano
```

Always `--dry-run` first. Remember that `results.jsonl` stores candidates down to the
storage floor (0.10), not beliefs — `send_to_ps.py` filters at `OPERATIONAL_CONFIDENCE`
(0.55) via `--min-confidence`. Submitting the raw file without that filter would push
thousands of sub-threshold detections into a live city.

---

## Gotchas, collected

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| Job dies in ~49 s on the cluster with a `pyexiv2` import error | The cluster is Rocky 8 (glibc 2.28); published `pyexiv2` wheels bundle a libexiv2 needing glibc ≥ 2.29 | Already fixed on the default branch (`sources/gsv.py` installs a stub). Any city branch cut before that fix must merge or cherry-pick it |
| Startup hangs > 11 min at 590 MB RSS | Unauthenticated HuggingFace Hub update check | `export HF_HUB_OFFLINE=1` |
| Run aborts complaining about the geometry or source | Run directory is bound to one geometry hash + source + storage floor; the checkout is on the wrong branch | Check out the right branch; use the branch guard |
| Fresh cluster setup is on `mapillary-source` | `hyak_sal_setup.sh` hardcodes that checkout | Check out the branch you want after setup |
| "Metadata unavailable" errors pile up mid-run | Google is soft-throttling the host IP | Lower `--processing-concurrency` to 10–20; the run is resumable and failures retry |
| Job fails writing output, or the whole lab's jobs start failing | Shared scratch quota (1 TB / 1M files) is near its ceiling | Archive to the lab file server and clean up; never archive imagery on the cluster |
| Files vanish between cluster sessions | `/tmp` is node-local and login nodes are load-balanced | Stage to shared group scratch, never `/tmp` |
| GPU util ~100% but power well under cap on the lab box | Time-slicing with `sidewalk-ai-api` | Nothing to do; let it run |
