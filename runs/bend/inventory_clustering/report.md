# bend: label clustering vs the city curb-ramp inventory (#106 Part 2)

- inventory: 12504 kept ramps (LifeCycleStatus IN ('I')) of 14822, `inventory.geojson` sha256 `30acadb2abdf5dfd06adaa32705814518e8af505ee9a166db07d6a34cb7bb8ba`, fetched 2026-09-26T14:28:53+00:00 from https://services5.arcgis.com/JisFYcK2mIVg9ueP/arcgis/rest/services/sCurbRamps/FeatureServer/0
- results `results.jsonl` sha256 `1307faa8041acbbf4cba78fd53979e2215511b8371c018f427c356f6b0e26153`
- regions: none (no server), so per-region == citywide

## tier 0.3, auto frame

- visible pool: 12065 of 12504 inventory ramps within 20 m of one of 78560 panos (439 excluded)
- 51529 labels synthesized; raycast placed 49701 detections at or above the storage floor
- camera height mode `auto`: auto -> gsv-per-rig
- 78560 of 78560 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 1.96 m, 6 of 6 measured -> 2.5 m
  - 2008: median 1.78 m, 9 of 9 measured -> 2.5 m
  - 2009: median 1.83 m, 13 of 19 measured -> 2.5 m
  - 2011: median n/a, 0 of 8 measured -> 2.5 m
  - 2012: median 2.36 m, 166 of 3603 measured -> 2.5 m
  - 2015: median 2.37 m, 77 of 206 measured -> 2.5 m
  - 2017: median 2.39 m, 72 of 740 measured -> 2.5 m
  - 2018: median 2.33 m, 264 of 1451 measured -> 2.5 m
  - 2019: median 2.34 m, 1541 of 1666 measured -> 2.5 m
  - 2021: median 2.38 m, 909 of 1007 measured -> 2.5 m
  - 2024: median 2.37 m, 59928 of 65897 measured -> 2.5 m
  - 2025: median 2.35 m, 2881 of 3948 measured -> 2.5 m
- fusion_server+attach: 1792 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 14650 | 51529 | 14476 | 0.880 / 0.896 / 0.899 | 0.093 (1009/10808) | 0.096 | 0.016 (170/10780) | 1.36 |
| ps @ 10 m | 13915 | 51529 | 13838 | 0.873 / 0.891 / 0.894 | 0.057 (618/10752) | 0.058 | 0.020 (213/10680) | 1.29 |
| ps @ 12.5 m | 13633 | 51529 | 13585 | 0.863 / 0.885 / 0.889 | 0.050 (538/10683) | 0.051 | 0.023 (243/10634) | 1.28 |
| ps @ 15 m | 13423 | 51529 | 13389 | 0.851 / 0.876 / 0.881 | 0.051 (534/10566) | 0.051 | 0.025 (262/10618) | 1.27 |
| ps_citywide @ 7.5 m | 14650 | 51529 | 14476 | 0.880 / 0.896 / 0.899 | 0.093 (1009/10808) | 0.096 | 0.016 (170/10780) | 1.36 |
| ps @ 7.5 m (server centroid) | 14650 | 51529 | 14650 | 0.875 / 0.896 / 0.901 | 0.057 (616/10815) | 0.058 | 0.016 (170/10780) | 1.35 |
| fusion | 14058 | 49701 | 14058 | 0.885 / 0.897 / 0.900 | 0.048 (515/10820) | 0.049 | 0.031 (316/10324) | 1.30 |
| fusion_server+attach | 14094 | 51529 | 14058 | 0.885 / 0.897 / 0.900 | 0.048 (515/10820) | 0.049 | 0.031 (316/10324) | 1.30 |

## tier 0.3, 2.6 m frame

- visible pool: 12065 of 12504 inventory ramps within 20 m of one of 78560 panos (439 excluded)
- 51529 labels synthesized; raycast placed 49701 detections at or above the storage floor
- fusion_server+attach: 1792 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 14650 | 51529 | 14476 | 0.876 / 0.893 / 0.896 | 0.098 (1055/10775) | 0.101 | 0.020 (220/10788) | 1.36 |
| ps @ 10 m | 13915 | 51529 | 13838 | 0.870 / 0.889 / 0.892 | 0.061 (654/10722) | 0.062 | 0.025 (265/10680) | 1.30 |
| ps @ 12.5 m | 13633 | 51529 | 13585 | 0.861 / 0.883 / 0.887 | 0.054 (570/10651) | 0.054 | 0.028 (295/10631) | 1.28 |
| ps @ 15 m | 13423 | 51529 | 13389 | 0.848 / 0.873 / 0.878 | 0.054 (564/10531) | 0.054 | 0.029 (313/10616) | 1.27 |
| ps_citywide @ 7.5 m | 14650 | 51529 | 14476 | 0.876 / 0.893 / 0.896 | 0.098 (1055/10775) | 0.101 | 0.020 (220/10788) | 1.36 |
| ps @ 7.5 m (server centroid) | 14650 | 51529 | 14650 | 0.875 / 0.896 / 0.901 | 0.057 (616/10815) | 0.058 | 0.020 (220/10788) | 1.35 |
| fusion | 14187 | 49701 | 14187 | 0.884 / 0.897 / 0.900 | 0.048 (524/10817) | 0.050 | 0.034 (346/10287) | 1.31 |
| fusion_server+attach | 14223 | 51529 | 14187 | 0.884 / 0.897 / 0.900 | 0.048 (524/10817) | 0.050 | 0.034 (346/10287) | 1.31 |

## tier 0.55, auto frame

- visible pool: 12065 of 12504 inventory ramps within 20 m of one of 78560 panos (439 excluded)
- 51529 labels synthesized; raycast placed 49701 detections at or above the storage floor
- camera height mode `auto`: auto -> gsv-per-rig
- 78560 of 78560 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 1.96 m, 6 of 6 measured -> 2.5 m
  - 2008: median 1.78 m, 9 of 9 measured -> 2.5 m
  - 2009: median 1.83 m, 13 of 19 measured -> 2.5 m
  - 2011: median n/a, 0 of 8 measured -> 2.5 m
  - 2012: median 2.36 m, 166 of 3603 measured -> 2.5 m
  - 2015: median 2.37 m, 77 of 206 measured -> 2.5 m
  - 2017: median 2.39 m, 72 of 740 measured -> 2.5 m
  - 2018: median 2.33 m, 264 of 1451 measured -> 2.5 m
  - 2019: median 2.34 m, 1541 of 1666 measured -> 2.5 m
  - 2021: median 2.38 m, 909 of 1007 measured -> 2.5 m
  - 2024: median 2.37 m, 59928 of 65897 measured -> 2.5 m
  - 2025: median 2.35 m, 2881 of 3948 measured -> 2.5 m
- fusion_server+attach: 1792 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 14650 | 51529 | 14476 | 0.880 / 0.896 / 0.899 | 0.093 (1009/10808) | 0.096 | 0.016 (170/10780) | 1.36 |
| ps @ 10 m | 13915 | 51529 | 13838 | 0.873 / 0.891 / 0.894 | 0.057 (618/10752) | 0.058 | 0.020 (213/10680) | 1.29 |
| ps @ 12.5 m | 13633 | 51529 | 13585 | 0.863 / 0.885 / 0.889 | 0.050 (538/10683) | 0.051 | 0.023 (243/10634) | 1.28 |
| ps @ 15 m | 13423 | 51529 | 13389 | 0.851 / 0.876 / 0.881 | 0.051 (534/10566) | 0.051 | 0.025 (262/10618) | 1.27 |
| ps_citywide @ 7.5 m | 14650 | 51529 | 14476 | 0.880 / 0.896 / 0.899 | 0.093 (1009/10808) | 0.096 | 0.016 (170/10780) | 1.36 |
| ps @ 7.5 m (server centroid) | 14650 | 51529 | 14650 | 0.875 / 0.896 / 0.901 | 0.057 (616/10815) | 0.058 | 0.016 (170/10780) | 1.35 |
| fusion | 14058 | 49701 | 14058 | 0.885 / 0.897 / 0.900 | 0.048 (515/10820) | 0.049 | 0.031 (316/10324) | 1.30 |
| fusion_server+attach | 14094 | 51529 | 14058 | 0.885 / 0.897 / 0.900 | 0.048 (515/10820) | 0.049 | 0.031 (316/10324) | 1.30 |

## tier 0.55, 2.6 m frame

- visible pool: 12065 of 12504 inventory ramps within 20 m of one of 78560 panos (439 excluded)
- 51529 labels synthesized; raycast placed 49701 detections at or above the storage floor
- fusion_server+attach: 1792 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 14650 | 51529 | 14476 | 0.876 / 0.893 / 0.896 | 0.098 (1055/10775) | 0.101 | 0.020 (220/10788) | 1.36 |
| ps @ 10 m | 13915 | 51529 | 13838 | 0.870 / 0.889 / 0.892 | 0.061 (654/10722) | 0.062 | 0.025 (265/10680) | 1.30 |
| ps @ 12.5 m | 13633 | 51529 | 13585 | 0.861 / 0.883 / 0.887 | 0.054 (570/10651) | 0.054 | 0.028 (295/10631) | 1.28 |
| ps @ 15 m | 13423 | 51529 | 13389 | 0.848 / 0.873 / 0.878 | 0.054 (564/10531) | 0.054 | 0.029 (313/10616) | 1.27 |
| ps_citywide @ 7.5 m | 14650 | 51529 | 14476 | 0.876 / 0.893 / 0.896 | 0.098 (1055/10775) | 0.101 | 0.020 (220/10788) | 1.36 |
| ps @ 7.5 m (server centroid) | 14650 | 51529 | 14650 | 0.875 / 0.896 / 0.901 | 0.057 (616/10815) | 0.058 | 0.020 (220/10788) | 1.35 |
| fusion | 14187 | 49701 | 14187 | 0.884 / 0.897 / 0.900 | 0.048 (524/10817) | 0.050 | 0.034 (346/10287) | 1.31 |
| fusion_server+attach | 14223 | 51529 | 14187 | 0.884 / 0.897 / 0.900 | 0.048 (524/10817) | 0.050 | 0.034 (346/10287) | 1.31 |

covered = pool ramps with a cluster assigned (nearest ramp within r) / pool; split = covered ramps with >= 2 clusters / covered; merge = clusters whose members fall on >= 2 distinct ramps with >= 2 members each / clusters with >= 2 assigned members; clusters/covered = all clusters / covered pool ramps. Every cluster is placed at the mean of its members' raycast positions in the frame named, except the `(server centroid)` row. Rule and read: docs/ps-clustering-eval.md, "City-inventory scoring".
