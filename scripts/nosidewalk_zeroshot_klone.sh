#!/bin/bash
# Klone (UW Hyak) jobs for scripts/nosidewalk_zeroshot.py, as run on 2026-11-27.
# See docs/overnight/nosidewalk_zeroshot.md. Usage, from a login node:
#
#   W=/gscratch/scrubbed/$USER/nosidewalk-zeroshot-2026-11   # code/ = a copy of this repo
#   sbatch --export=ALL,W=$W,STEP=gsv     scripts/nosidewalk_zeroshot_klone.sh
#   sbatch --export=ALL,W=$W,STEP=segment --gpus=1 --mem=64G scripts/nosidewalk_zeroshot_klone.sh
#
# Environments: PY_GPU has torch + transformers (Mask2Former); PY_NET is a venv over it
# with streetlevel==0.12.10 added (note: pip pulls a second torch in via pyequilib; the
# GPU step therefore uses PY_GPU). Delete macOS `._*` files from any upload into views/
# before segmenting: footway_segmentation.py globs *.jpg and they are not images.
#SBATCH --job-name=nszs
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --requeue
set -euo pipefail
: "${W:?set W}" "${STEP:?set STEP=gsv|segment}"
PY_GPU=${PY_GPU:-/gscratch/makelab/jonf/envs/sidewalkcv2/bin/python}
PY_NET=${PY_NET:-/gscratch/scrubbed/jfroehli/envs/nosidewalk-zs/bin/python}
export HF_HOME=${HF_HOME:-/gscratch/scrubbed/$USER/hf}
export PROJ_DATA=${PROJ_DATA:-/gscratch/makelab/jonf/envs/sidewalkcv2/share/proj}
cd "$W/code"
case "$STEP" in
  gsv)
    "$PY_NET" -u scripts/nosidewalk_zeroshot.py gsv-views --work "$W/work" --workers 8 ;;
  segment)
    find "$W/work/views" -name '._*' -delete
    "$PY_GPU" -u scripts/footway_segmentation.py segment --in "$W/work/views" \
        --out "$W/work/labels" --fp16 --batch-size 4 \
        --note "nosidewalk_zeroshot lateral views (90 deg FOV, pitch -30, 1024 px)"
    "$PY_GPU" -u scripts/nosidewalk_zeroshot.py bands --work "$W/work" ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
