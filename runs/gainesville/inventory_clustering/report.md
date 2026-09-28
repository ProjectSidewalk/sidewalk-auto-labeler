# gainesville: label clustering vs the city curb-ramp inventory (#106 Part 2)

- inventory: 3208 kept ramps (LIFECYCLE IN ('Active')) of 7248, `inventory.geojson` sha256 `5ddc4e17548faff50175bd3fba5b4cd32cf944cec77585a3a5f075a4ed8c8018`, fetched 2026-09-26T14:28:58+00:00 from https://services2.arcgis.com/Zzhtlau4ccHkQgTu/arcgis/rest/services/PublicWorksInfrastructure_AGO/FeatureServer/3
- results `results.jsonl` sha256 `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8`
- regions: nearest street of the server's street network (the server's insert rule); `streets.geojson`: 27448 features, sha256 `39b778bb4a85bb8a3fa42f89adfff9ffc2b0615be403534b3bab3dd03b602ed3`, 2026-09-28T17:26:41+00:00 (0.0 days old at run time), from https://sidewalk-gainesville.cs.washington.edu/v3/api/streets?filetype=geojson

## tier 0.3, auto frame

- visible pool: 3107 of 3208 inventory ramps within 20 m of one of 37435 panos (101 excluded)
- 24072 labels synthesized; raycast placed 37056 detections at or above the storage floor
- camera height mode `auto`: auto -> gsv-per-rig
- 37435 of 37435 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 2.15 m, 153 of 299 measured -> 2.5 m
  - 2008: median 2.42 m, 9 of 14 measured -> 2.5 m
  - 2011: median 2.12 m, 136 of 358 measured -> 2.5 m
  - 2014: median 2.29 m, 44 of 108 measured -> 2.5 m
  - 2015: median 1.94 m, 225 of 606 measured -> 2.5 m
  - 2016: median 2.13 m, 110 of 549 measured -> 2.5 m
  - 2017: median 2.24 m, 17 of 96 measured -> 2.5 m
  - 2018: median 2.07 m, 160 of 1059 measured -> 2.5 m
  - 2019: median 2.14 m, 64 of 304 measured -> 2.5 m
  - 2021: median 2.30 m, 334 of 368 measured -> 2.5 m
  - 2022: median 2.28 m, 2224 of 2592 measured -> 2.5 m
  - 2023: median 2.30 m, 1619 of 2140 measured -> 2.5 m
  - 2024: median 2.31 m, 2286 of 3679 measured -> 2.5 m
  - 2025: median 2.29 m, 1127 of 1497 measured -> 2.5 m
  - 2026: median 1.76 m, 21950 of 23766 measured -> 2 m
- fusion_server+attach: 2739 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 7936 | 24072 | 7548 | 0.785 / 0.864 / 0.889 | 0.312 (837/2685) | 0.377 | 0.009 (29/3117) | 2.96 |
| ps @ 10 m | 7215 | 24072 | 6986 | 0.775 / 0.857 / 0.883 | 0.258 (688/2664) | 0.294 | 0.014 (42/3052) | 2.71 |
| ps @ 12.5 m | 6820 | 24072 | 6666 | 0.764 / 0.850 / 0.874 | 0.236 (623/2640) | 0.259 | 0.019 (57/3009) | 2.58 |
| ps @ 15 m | 6571 | 24072 | 6448 | 0.750 / 0.843 / 0.870 | 0.222 (581/2618) | 0.242 | 0.021 (64/3000) | 2.51 |
| ps_citywide @ 7.5 m | 7781 | 24072 | 7405 | 0.782 / 0.863 / 0.889 | 0.283 (758/2682) | 0.339 | 0.009 (29/3078) | 2.90 |
| ps @ 7.5 m (server centroid) | 7936 | 24072 | 7936 | 0.789 / 0.871 / 0.897 | 0.268 (724/2706) | 0.296 | 0.009 (29/3117) | 2.93 |
| fusion | 7151 | 21176 | 7151 | 0.784 / 0.863 / 0.888 | 0.211 (565/2681) | 0.237 | 0.017 (48/2763) | 2.67 |
| fusion_server+attach | 7307 | 24072 | 7150 | 0.784 / 0.863 / 0.888 | 0.211 (565/2681) | 0.237 | 0.017 (48/2763) | 2.73 |

## tier 0.3, 2.6 m frame

- visible pool: 3107 of 3208 inventory ramps within 20 m of one of 37435 panos (101 excluded)
- 24072 labels synthesized; raycast placed 37056 detections at or above the storage floor
- fusion_server+attach: 2721 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 7936 | 24072 | 7548 | 0.684 / 0.833 / 0.884 | 0.288 (746/2587) | 0.357 | 0.024 (69/2867) | 3.07 |
| ps @ 10 m | 7215 | 24072 | 6986 | 0.678 / 0.827 / 0.880 | 0.230 (592/2569) | 0.264 | 0.027 (76/2806) | 2.81 |
| ps @ 12.5 m | 6820 | 24072 | 6666 | 0.673 / 0.820 / 0.872 | 0.207 (526/2547) | 0.225 | 0.032 (87/2761) | 2.68 |
| ps @ 15 m | 6571 | 24072 | 6448 | 0.661 / 0.814 / 0.869 | 0.195 (492/2528) | 0.209 | 0.034 (93/2755) | 2.60 |
| ps_citywide @ 7.5 m | 7781 | 24072 | 7405 | 0.681 / 0.830 / 0.883 | 0.268 (692/2579) | 0.329 | 0.025 (71/2844) | 3.02 |
| ps @ 7.5 m (server centroid) | 7936 | 24072 | 7936 | 0.789 / 0.871 / 0.897 | 0.268 (724/2706) | 0.296 | 0.024 (69/2867) | 2.93 |
| fusion | 8967 | 21176 | 8967 | 0.707 / 0.851 / 0.898 | 0.331 (875/2643) | 0.367 | 0.016 (43/2633) | 3.39 |
| fusion_server+attach | 9142 | 24072 | 8967 | 0.707 / 0.851 / 0.898 | 0.331 (875/2643) | 0.367 | 0.016 (43/2633) | 3.46 |

## tier 0.55, auto frame

- visible pool: 3107 of 3208 inventory ramps within 20 m of one of 37435 panos (101 excluded)
- 15573 labels synthesized; raycast placed 37056 detections at or above the storage floor
- camera height mode `auto`: auto -> gsv-per-rig
- 37435 of 37435 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 2.15 m, 153 of 299 measured -> 2.5 m
  - 2008: median 2.42 m, 9 of 14 measured -> 2.5 m
  - 2011: median 2.12 m, 136 of 358 measured -> 2.5 m
  - 2014: median 2.29 m, 44 of 108 measured -> 2.5 m
  - 2015: median 1.94 m, 225 of 606 measured -> 2.5 m
  - 2016: median 2.13 m, 110 of 549 measured -> 2.5 m
  - 2017: median 2.24 m, 17 of 96 measured -> 2.5 m
  - 2018: median 2.07 m, 160 of 1059 measured -> 2.5 m
  - 2019: median 2.14 m, 64 of 304 measured -> 2.5 m
  - 2021: median 2.30 m, 334 of 368 measured -> 2.5 m
  - 2022: median 2.28 m, 2224 of 2592 measured -> 2.5 m
  - 2023: median 2.30 m, 1619 of 2140 measured -> 2.5 m
  - 2024: median 2.31 m, 2286 of 3679 measured -> 2.5 m
  - 2025: median 2.29 m, 1127 of 1497 measured -> 2.5 m
  - 2026: median 1.76 m, 21950 of 23766 measured -> 2 m
- fusion_server+attach: 1336 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 5096 | 15573 | 4950 | 0.723 / 0.798 / 0.818 | 0.190 (470/2478) | 0.211 | 0.009 (23/2451) | 2.06 |
| ps @ 10 m | 4742 | 15573 | 4649 | 0.708 / 0.787 / 0.809 | 0.153 (375/2444) | 0.165 | 0.012 (29/2424) | 1.94 |
| ps @ 12.5 m | 4567 | 15573 | 4504 | 0.693 / 0.774 / 0.800 | 0.140 (338/2406) | 0.148 | 0.017 (42/2405) | 1.90 |
| ps @ 15 m | 4454 | 15573 | 4398 | 0.676 / 0.761 / 0.790 | 0.134 (318/2365) | 0.141 | 0.023 (54/2399) | 1.88 |
| ps_citywide @ 7.5 m | 4955 | 15573 | 4822 | 0.720 / 0.795 / 0.817 | 0.159 (392/2471) | 0.174 | 0.011 (27/2425) | 2.01 |
| ps @ 7.5 m (server centroid) | 5096 | 15573 | 5096 | 0.721 / 0.802 / 0.825 | 0.161 (400/2492) | 0.169 | 0.009 (23/2451) | 2.04 |
| fusion | 4729 | 14154 | 4729 | 0.724 / 0.798 / 0.819 | 0.120 (298/2478) | 0.128 | 0.015 (34/2275) | 1.91 |
| fusion_server+attach | 4812 | 15573 | 4729 | 0.724 / 0.798 / 0.819 | 0.120 (298/2478) | 0.128 | 0.015 (34/2275) | 1.94 |

## tier 0.55, 2.6 m frame

- visible pool: 3107 of 3208 inventory ramps within 20 m of one of 37435 panos (101 excluded)
- 15573 labels synthesized; raycast placed 37056 detections at or above the storage floor
- fusion_server+attach: 1324 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 5096 | 15573 | 4950 | 0.605 / 0.759 / 0.814 | 0.165 (390/2358) | 0.188 | 0.018 (41/2223) | 2.16 |
| ps @ 10 m | 4742 | 15573 | 4649 | 0.598 / 0.752 / 0.805 | 0.128 (300/2335) | 0.140 | 0.020 (45/2204) | 2.03 |
| ps @ 12.5 m | 4567 | 15573 | 4504 | 0.587 / 0.743 / 0.797 | 0.117 (269/2307) | 0.123 | 0.023 (51/2191) | 1.98 |
| ps @ 15 m | 4454 | 15573 | 4398 | 0.577 / 0.733 / 0.788 | 0.108 (246/2277) | 0.114 | 0.025 (55/2191) | 1.96 |
| ps_citywide @ 7.5 m | 4955 | 15573 | 4822 | 0.603 / 0.754 / 0.810 | 0.145 (339/2344) | 0.162 | 0.020 (44/2205) | 2.11 |
| ps @ 7.5 m (server centroid) | 5096 | 15573 | 5096 | 0.721 / 0.802 / 0.825 | 0.161 (400/2492) | 0.169 | 0.018 (41/2223) | 2.04 |
| fusion | 6252 | 14154 | 6252 | 0.638 / 0.779 / 0.828 | 0.247 (597/2420) | 0.266 | 0.013 (28/2088) | 2.58 |
| fusion_server+attach | 6347 | 15573 | 6252 | 0.638 / 0.779 / 0.828 | 0.247 (597/2420) | 0.266 | 0.013 (28/2088) | 2.62 |

covered = pool ramps with a cluster assigned (nearest ramp within r) / pool; split = covered ramps with >= 2 clusters / covered; merge = clusters whose members fall on >= 2 distinct ramps with >= 2 members each / clusters with >= 2 assigned members; clusters/covered = all clusters / covered pool ramps. Every cluster is placed at the mean of its members' raycast positions in the frame named, except the `(server centroid)` row. Rule and read: docs/ps-clustering-eval.md, "City-inventory scoring".
