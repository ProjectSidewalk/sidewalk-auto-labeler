# Running the auto-labeler on a Makeability Lab server (RA quickstart)

Lab-facing happy-path for running `sidewalk-auto-labeler` on our GPU server. This is the
"how do I actually launch a city run" guide; for what the pipeline *is* and full flag/output
details, see the top-level [`README.md`](../README.md).

**TL;DR:** SSH to `makelab2`, run the job inside `tmux`, launch it, detach, walk away. The
run is checkpointed and resumable, so a dropped SSH connection does **not** kill it.

---

## Which machine? Use `makelab2` (it has the GPU)

Detection is GPU-bound. `makelab2` has an NVIDIA A40; `makelab1` has **no GPU**, so a city
run there would fall back to CPU and take *weeks* instead of hours. Use `makelab2`.

### Makeability Lab servers

| Component | `makelab1.cs.washington.edu` | `makelab2.cs.washington.edu` |
| :--- | :--- | :--- |
| **Hardware model** | Supermicro SYS-6029U-E1CR4 | Supermicro SYS-620U-TNR |
| **OS** | Rocky Linux 9.8 (kernel 5.14.0, x86-64) | Rocky Linux 9.8 (kernel 5.14.0, x86-64) |
| **CPU** | 2× Intel Xeon Silver 4214 @ 2.20 GHz — 48 logical cores | 2× Intel Xeon Silver 4310 @ 2.10 GHz — 48 logical cores |
| **System RAM** | 187 GB | 188 GB |
| **GPU** | none | 1× NVIDIA A40, 48 GB VRAM |
| **NVIDIA driver / CUDA** | n/a | 595.80 / CUDA 13.2 |

> Specs captured 2026-07-02. To re-check on any lab machine, one command dumps everything:
> ```bash
> hostnamectl; echo "---"; lscpu; echo "---"; free -h; echo "---"; nvidia-smi
> ```
> (`nvidia-smi: command not found` means the machine has no NVIDIA GPU / drivers.)

### Before you launch: check who else is on the box

These are shared lab servers. Check current load so you don't stomp on someone's job:

```bash
nvidia-smi          # GPU: is another process already holding VRAM / running at high util?
free -h             # RAM headroom
htop                # CPU load and per-user processes (q to quit)
```

The A40 has 48 GB of VRAM; this pipeline uses only a few GB at a time (see the concurrency
note below), so it coexists fine with other jobs as long as the GPU isn't already saturated.

**You'll usually be sharing the A40 with the Project Sidewalk AI API service.** Expect to see
a long-running process like this in `nvidia-smi` (user `saugstad`, ~9 GB VRAM):

```
saugstad  /opt/conda/bin/python3.11 /opt/conda/bin/gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 main:app
```

That's [`sidewalk-ai-api`](https://github.com/ProjectSidewalk/sidewalk-ai-api) — the live
service behind Project Sidewalk's AI tag-suggestion / validation features. **Do not kill it**
(it's a semi-production endpoint, and it isn't yours). It's a single-worker gunicorn server
that reserves ~9 GB of VRAM continuously but only competes for GPU *compute* in bursts, when a
request actually comes in — so between requests the card is effectively yours. To identify any
mystery process before assuming it's safe to touch:

```bash
ps -o user=,etime=,cmd= -p <PID>
```

---

## First-time setup on makelab2

```bash
ssh jonf@makelab2.cs.washington.edu

git clone https://github.com/ProjectSidewalk/sidewalk-auto-labeler.git
cd sidewalk-auto-labeler

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Linux x86-64, `pip install -r requirements.txt` pulls a **CUDA-enabled** PyTorch wheel by
default, so the A40 is used automatically — no extra index-url step needed. (`conda env
create -f environment.yml` also works here since that export is linux-64; the venv path above
is simpler and is what we've run in practice.)

On the **first run**, the RampNet model (~360 MB) downloads from HuggingFace. You'll see a
HuggingFace warning that `modeling.py` "may contain malicious code" — this is the standard
`trust_remote_code=True` disclaimer for any model that ships custom architecture code; the
`projectsidewalk/rampnet-model` repo is ours and is safe to load.

---

## Running a city

`example_geojson/bend.geojson` (and `chicago`, `vancouver`) already ship in the repo — no file
transfer needed. For a **new** city, `scp` its GeoJSON into `example_geojson/` first (it must
be a bare geometry object — see the README's "Deploying to a new city").

### 1. Scope it first (always)

```bash
python main.py example_geojson/bend.geojson --name bend --scan-only
```

No model load, nothing processed — just the pano count and a runtime estimate, so you know
whether you're committing to hours or days. Bend is ~78k panos.

### 2. Launch inside tmux (so it survives SSH drops)

A city run is many hours (Bend ≈ half a day to a day on the A40). Your SSH connection **will**
drop over that span — laptop sleeps, Wi-Fi blips. Run inside `tmux` so the job keeps going on
the server regardless.

```bash
tmux new -s autolabeler          # start a persistent session

# tmux may drop you into a fresh shell — reactivate the venv inside it:
source .venv/bin/activate
python main.py example_geojson/bend.geojson --name bend
```

Once you see the `Processing New Panoramas` progress bar advancing, detach and let it run:
**press `Ctrl-b`, release, then press `d`.** You're back at your normal shell; the job runs on.

Reattach any time (after SSH'ing back in) to check progress:

```bash
tmux attach -t autolabeler
```

### About `--processing-concurrency` (don't drop it to avoid "GPU OOM")

You may see advice to lower `--processing-concurrency` to a handful "or the GPU will run out of
memory." **That's not how this pipeline works.** GPU inference is serialized by a lock
(`detectors/curb_ramp.py` — `with self._inference_lock, torch.no_grad():`), so only **one**
forward pass runs at a time no matter the concurrency. VRAM use is that of a single inference
(a few GB), so there is **no OOM risk** from a higher setting on a 48 GB A40.

What the flag actually controls: the number of parallel **download/pre-process** threads
feeding the GPU (default `50`). Throughput is GPU-bound (~1–2 panos/sec on the A40), so the
default keeps the GPU fed on our 48-core box — just use the default unless you have a reason
not to. The companion `--coverage-concurrency` (default `100`) scales the initial
tile-scanning phase.

### Is it slow because the GPU is starved, or because you're sharing it?

If a run feels slow, watch the card in a second SSH session and read **GPU-Util *and* power
together** — they distinguish the two very different causes:

```bash
watch -n 2 nvidia-smi
```

| What you see | Meaning | What to do |
| :--- | :--- | :--- |
| **Util low** (~20–50%) | GPU is *starved* — waiting on Street View downloads | Raise `--processing-concurrency` (it's resumable, so restart freely) |
| **Util ~100%, power near cap** (~250–300 W) | You're genuinely GPU-bound and have the card to yourself | Nothing to do — that's the A40's real rate for this model |
| **Util ~100%, power well under cap** (e.g. ~150 W) | Two jobs *time-slicing* the GPU — the card is always "busy" but neither runs flat-out | You're sharing with the AI API service (above); just let it run, or coordinate |

The middle and bottom rows look identical on util alone — **power draw is what tells them
apart.** In practice this pipeline is GPU-bound, so raising concurrency rarely helps; if you're
sharing with `sidewalk-ai-api`, ~1.7 panos/sec is a normal rate and there's no safe knob to
turn beyond letting the run finish (it's checkpointed, so leaving it overnight is free).

### 3. If the connection drops mid-run — don't panic

`Broken pipe` / `Operation timed out` just means SSH died, not the job. SSH back in,
`tmux attach -t autolabeler`, and the progress bar picks up where it was. Even if the tmux
session itself were lost, the run is **checkpointed**: re-running the same command skips
already-processed panos (tracked in `runs/bend/already_processed.txt`) and continues. See the
README's "Caching & resumability."

---

## After the run: spot-check, score, submit

All output lands in `runs/bend/`. The QA and submission steps are the same as anywhere and are
documented in the README — briefly:

```bash
python scripts/spot_check_gallery.py runs/bend      # visual QA + validation UI
python scripts/score_validation.py runs/bend        # precision/recall + threshold sweep
python send_to_ps.py runs/bend/results.jsonl --dry-run   # then --endpoint to submit for real
```

Submission requires a Project Sidewalk instance for the city to **already exist** — see the
README "Prerequisites" section and [`bend-onboarding.md`](bend-onboarding.md).
