# Vancouver split examples (#56)

Inventory ramps that one clustering arm splits and another does not, drawn by
`scripts/split_figures.py` (seed 56, 3 per group). Each figure shows the same ramp under
the deployed clusters, the server rule re-run (ps @ 7.5 m) and fusion + bearing attach.

Every cluster is placed at the mean of its labels' **server** positions and assigned to the
nearest inventory ramp within 5 m. In this frame every cluster is placed, so no arm gains by
dropping unplaceable clusters. Split rates here (covered ramps with >= 2 clusters, divided by
covered ramps): deployed 0.196, ps @ 7.5 m 0.138, fusion + attach 0.120, the `placement:
server`, `label_set: all` rows of `runs/vancouver/inventory_clustering/report.md`. The
raycast-frame rows there (0.255 / 0.200 / 0.092 for `fusion_server`) score only clusters with
a raycastable label, and fusion loses more of those (3,810 vs 1,524 for ps @ 7.5 m). On one
label set (the `label_set: mapped` table) each arm reads best in the frame it clusters in,
so neither frame is the arms' common ground. The inventory is descriptive for Vancouver
(#56); these figures illustrate it and decide nothing.

| group | visible-pool ramps | figures |
|---|---:|---|
| server rule splits, fusion does not | 596 | `fix_*.png` |
| fusion splits, server rule does not | 418 | `break_*.png` |
| deployed splits, fresh server rule does not (stale clusters) | 678 | `stale_*.png` |
| server rule and fusion both split | 728 | `both_*.png` |

`groups.csv` lists each drawn ramp's position and cluster count per arm. The ground truth
that would replace the inventory for cities without one is RampNet#224.

Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community (Esri World Imagery).
