# vancouver: clustering scored against cluster-review GT (RampNet#224)

Protocol and decision rule: RampNet docs/cluster_review_protocol.md (pre-registered); rubric: RampNet benchmark/RUBRICS.md §6.

- bundle: `cluster_review` of vancouver: 80 units (30 pilot), 1406 labels; label snapshot sha256 `57c31c73dc75a6139b2694fdc5e7c0823bf0d0504348f1e1f38d3681b713098d` (64847 features, fetched 2026-09-28T23:01:58+00:00)
- deployed clusters sha256 `d8a1e7065e2cf97ff7da0ab50eba7e754b2ac67c4286ecbc8d528f8ff6bec1c3`; results.jsonl sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`; fusion frame `auto`
- arms rebuilt: deployed 18684, ps @ 7.5 m 17451, ps @ 10 m 16034, ps @ 12.5 m 15421, ps @ 15 m 15080, fusion_server 18414, fusion_server+attach 16101 clusters
- scored label set: 1406 labels in the bundle's windows; 0 human label(s) excluded from every arm; labels an arm does not hold, scored as its singletons: deployed 1, ps @ 7.5 m 0, ps @ 10 m 0, ps @ 12.5 m 0, ps @ 15 m 0, fusion_server 34, fusion_server+attach 34

## NO GT YET

`assignments.json` does not exist in the bundle: no reviewer has exported an assignment, so there is nothing to score and no number below is a result. Review the pilot in RampNet `scripts/cluster_review_gallery.py`, save the export into the bundle, and re-run this command.

- self-consistency: `fusion_server+attach` scored against an assignment derived from itself (in memory, never written) over all 80 units: split 0/330, merge 0/245 -- OK
