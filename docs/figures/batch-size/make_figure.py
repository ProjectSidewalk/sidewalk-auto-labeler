"""Draw docs/figures/batch-size/batch_size_a40.png from data/passes.csv (issue #2).

The CSV is the quiet-GPU table from the results comment on issue #2: makelab2's A40,
300 Vancouver store panos, ``detect_from_store.py --workers 16``, two passes per size.
Peak VRAM was already recorded per batch size, so both passes of batch 1 and 4 repeat it.

Usage::

    python docs/figures/batch-size/make_figure.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CARD_GB = 48.0          # NVIDIA A40
OTHER_JOBS_GB = 28.0    # other jobs seen on the card earlier that day (issue #2)
BLUE = '#2a78d6'        # dataviz reference palette, categorical slot 1
INK, MUTED, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def main():
    rows = list(csv.DictReader(open(HERE / 'data' / 'passes.csv', newline='')))
    bs = [int(r['batch_size']) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_xticks([1, 4, 8])
        ax.set_xlim(0, 9)
        ax.set_xlabel('batch size (panos per forward pass)', color=INK)
        ax.tick_params(colors=MUTED)
        ax.grid(axis='y', color='#e6e5e1', linewidth=0.8)
        ax.set_axisbelow(True)

    ax1.scatter(bs, [float(r['panos_per_s']) for r in rows], s=46, color=BLUE,
                edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax1.set_ylim(0, 1.6)
    ax1.set_ylabel('throughput (panos/s)', color=INK)
    ax1.set_title('Throughput is flat at 1.27-1.28 panos/s', color=INK, fontsize=11, loc='left')
    ax1.annotate('forward pass = 99% of wall time at every size', xy=(0.3, 0.35),
                 color=MUTED, fontsize=9)

    ax2.scatter(bs, [float(r['peak_vram_gb']) for r in rows], s=46, color=BLUE,
                edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax2.axhline(CARD_GB, color=INK, linewidth=1.2)
    ax2.text(0.3, CARD_GB + 1, 'A40 capacity, 48 GB', color=INK, fontsize=9)
    free = CARD_GB - OTHER_JOBS_GB
    ax2.axhline(free, color=MUTED, linewidth=1.2, linestyle='--')
    ax2.text(0.3, free - 3, f'room left beside ~{OTHER_JOBS_GB:.0f} GB of other jobs',
             color=MUTED, fontsize=9)
    ax2.set_ylim(0, 54)
    ax2.set_ylabel('peak GPU memory (GB)', color=INK)
    ax2.set_title('Memory grows with batch size', color=INK, fontsize=11, loc='left')

    fig.suptitle('Batching on the A40 buys no throughput (2 passes per size, quiet GPU)',
                 color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    out = HERE / 'batch_size_a40.png'
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(out)


if __name__ == '__main__':
    main()
