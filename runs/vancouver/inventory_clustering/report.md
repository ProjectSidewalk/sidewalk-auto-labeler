# vancouver: label clustering vs the city curb-ramp inventory (#106 Part 2)

- inventory: 11355 kept ramps (STATUS IN ('Available')) of 17614, `inventory.geojson` sha256 `c4d2497995f7b6261333c3859a668348cd62a36367593cef2dfdfa6b7e020d87`, fetched 2026-09-28T23:11:26+00:00 from https://services.arcgis.com/oNvpY90qsPDizwkN/arcgis/rest/services/COV_TransCurbRamp/FeatureServer/0
- results `results.jsonl` sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`
- regions: nearest street of the server's street network (the server's insert rule); `streets.geojson`: 12567 features, sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`, 2026-09-30T01:39:49+00:00 (4.8 days old at run time), from https://sidewalk-vancouver.cs.washington.edu/v3/api/streets?filetype=geojson; 11783 open streets kept (the server snaps to open streets only)

## Server-label arms (#56; the inventory is DESCRIPTIVE there -- the confirmatory fragmentation test is the near-cluster proxy in ps_clustering_eval/report.md)

- `raw_labels.geojson`: 64847 features, sha256 `57c31c73dc75a6139b2694fdc5e7c0823bf0d0504348f1e1f38d3681b713098d`, 2026-09-28T23:01:58+00:00 (5.9 days old at run time), from https://sidewalk-vancouver.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- `clusters.geojson`: 18684 features, sha256 `d8a1e7065e2cf97ff7da0ab50eba7e754b2ac67c4286ecbc8d528f8ff6bec1c3`, 2026-09-30T01:39:55+00:00 (4.8 days old at run time), from https://sidewalk-vancouver.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson
- 64814 labels of the AI account, 50252 of them map pixel-exactly to a stored detection of the rebuilt run (the rest are placed only at their server position); 33 human labels are in `deployed` and `fusion_server` but not in `ps @ t`, as in eval_ps_clustering.py
- matched label set (`label_set: mapped`): 50252 mapped AI + 33 human labels; the 14562 unmapped AI labels are left out of every arm

### auto frame (fusion_server association; server-placed rows of the fixed arms are written once)

- visible pool: 10685 of 11355 inventory ramps within 20 m of one of 28830 panos
- fusion_server input: 50252 mapped AI + 14562 unmapped AI (at 0.55) + 33 human labels on 28862 panos (25242 positioned by inversion); 2313 labels attached by bearing
- camera height mode `auto`: auto -> gsv-per-rig
- 28830 of 28830 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2011: median 2.28 m, 22 of 197 measured -> 2.5 m
  - 2012: median 2.41 m, 4 of 310 measured -> 2.5 m
  - 2014: median 2.34 m, 40 of 722 measured -> 2.5 m
  - 2015: median 2.38 m, 8 of 277 measured -> 2.5 m
  - 2016: median 2.41 m, 18 of 319 measured -> 2.5 m
  - 2017: median 2.43 m, 15 of 258 measured -> 2.5 m
  - 2018: median 2.24 m, 46 of 953 measured -> 2.5 m
  - 2019: median 2.32 m, 968 of 1663 measured -> 2.5 m
  - 2021: median 2.37 m, 506 of 926 measured -> 2.5 m
  - 2022: median 2.35 m, 1624 of 3434 measured -> 2.5 m
  - 2023: median 2.37 m, 4562 of 8653 measured -> 2.5 m
  - 2024: median 2.35 m, 5482 of 10621 measured -> 2.5 m
  - 2025: median 2.36 m, 277 of 497 measured -> 2.5 m

| arm | placement | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---|---:|---:|---:|---|---|---:|---|---:|
| deployed | server | 18684 | 64918 | 18684 | 0.841 / 0.903 / 0.916 | 0.196 (1894/9647) | 0.219 | 0.020 (207/10370) | 1.94 |
| deployed | raycast | 18684 | 64918 | 16790 | 0.833 / 0.877 / 0.887 | 0.255 (2390/9366) | 0.304 | 0.014 (136/9585) | 1.99 |
| ps @ 7.5 m | server | 17451 | 64814 | 17451 | 0.838 / 0.902 / 0.915 | 0.138 (1329/9637) | 0.145 | 0.019 (194/10087) | 1.81 |
| ps @ 7.5 m | raycast | 17451 | 64814 | 15927 | 0.835 / 0.878 / 0.888 | 0.200 (1875/9383) | 0.223 | 0.014 (130/9362) | 1.86 |
| ps @ 10 m | server | 16034 | 64814 | 16034 | 0.819 / 0.893 / 0.909 | 0.109 (1045/9546) | 0.115 | 0.025 (244/9937) | 1.68 |
| ps @ 12.5 m | server | 15421 | 64814 | 15421 | 0.807 / 0.885 / 0.903 | 0.103 (970/9458) | 0.108 | 0.028 (276/9894) | 1.63 |
| ps @ 15 m | server | 15080 | 64814 | 15080 | 0.796 / 0.876 / 0.897 | 0.101 (946/9364) | 0.107 | 0.032 (314/9858) | 1.61 |
| ps_citywide @ 7.5 m | server | 17310 | 64814 | 17310 | 0.837 / 0.901 / 0.915 | 0.127 (1225/9629) | 0.133 | 0.020 (198/10044) | 1.80 |
| fusion_server | server | 18484 | 64847 | 18484 | 0.832 / 0.900 / 0.915 | 0.201 (1932/9619) | 0.241 | 0.039 (373/9487) | 1.92 |
| fusion_server | raycast | 18484 | 64847 | 14674 | 0.832 / 0.874 / 0.884 | 0.092 (861/9338) | 0.096 | 0.024 (208/8645) | 1.98 |
| fusion_server+attach | server | 16171 | 64847 | 16171 | 0.821 / 0.897 / 0.912 | 0.120 (1150/9582) | 0.130 | 0.041 (389/9548) | 1.69 |

Same label set (`label_set: mapped`: 50285 labels, the unmapped AI labels left out of every arm), each arm placed both ways, r = 5 m. fusion_server here: 50252 AI + 33 human labels.

| arm | clusters | split, raycast placement | split, server placement | placed (raycast / server) | covered (raycast / server) | merge (raycast / server) |
|---|---:|---:|---:|---|---|---|
| ps @ 7.5 m | 15580 | 0.163 (1518/9285) | 0.098 (905/9257) | 15293 / 15580 | 0.869 / 0.866 | 0.020 / 0.019 |
| fusion_server | 16532 | 0.090 (840/9332) | 0.160 (1483/9256) | 14641 / 16532 | 0.873 / 0.866 | 0.024 / 0.031 |
| fusion_server+attach | 14702 | 0.090 (840/9332) | 0.093 (861/9212) | 14641 / 14702 | 0.873 / 0.862 | 0.024 / 0.032 |

### 2.6 m frame (fusion_server association; server-placed rows of the fixed arms are written once)

- visible pool: 10685 of 11355 inventory ramps within 20 m of one of 28830 panos
- fusion_server input: 50252 mapped AI + 14562 unmapped AI (at 0.55) + 33 human labels on 28862 panos (25242 positioned by inversion); 2315 labels attached by bearing

| arm | placement | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---|---:|---:|---:|---|---|---:|---|---:|
| deployed | raycast | 18684 | 64918 | 16790 | 0.829 / 0.874 / 0.885 | 0.262 (2448/9340) | 0.316 | 0.016 (155/9609) | 2.00 |
| ps @ 7.5 m | raycast | 17451 | 64814 | 15927 | 0.831 / 0.876 / 0.886 | 0.208 (1949/9355) | 0.234 | 0.016 (150/9378) | 1.87 |
| fusion_server | server | 18424 | 64847 | 18424 | 0.830 / 0.898 / 0.913 | 0.205 (1970/9598) | 0.249 | 0.045 (424/9486) | 1.92 |
| fusion_server | raycast | 18424 | 64847 | 14654 | 0.832 / 0.873 / 0.884 | 0.088 (821/9332) | 0.092 | 0.029 (250/8606) | 1.97 |
| fusion_server+attach | server | 16109 | 64847 | 16109 | 0.818 / 0.895 / 0.910 | 0.127 (1210/9559) | 0.139 | 0.046 (441/9543) | 1.69 |

Same label set (`label_set: mapped`: 50285 labels, the unmapped AI labels left out of every arm), each arm placed both ways, r = 5 m. fusion_server here: 50252 AI + 33 human labels.

| arm | clusters | split, raycast placement | split, server placement | placed (raycast / server) | covered (raycast / server) | merge (raycast / server) |
|---|---:|---:|---:|---|---|---|
| ps @ 7.5 m | 15580 | 0.172 (1592/9260) | 0.098 (905/9257) | 15293 / 15580 | 0.867 / 0.866 | 0.023 / 0.019 |
| fusion_server | 16502 | 0.085 (791/9330) | 0.166 (1534/9239) | 14611 / 16502 | 0.873 / 0.865 | 0.029 / 0.034 |
| fusion_server+attach | 14672 | 0.085 (791/9330) | 0.100 (920/9192) | 14611 / 14672 | 0.873 / 0.860 | 0.029 / 0.036 |

### per-pano frame (fusion_server association; server-placed rows of the fixed arms are written once)

- visible pool: 10685 of 11355 inventory ramps within 20 m of one of 28830 panos
- fusion_server input: 50252 mapped AI + 14562 unmapped AI (at 0.55) + 33 human labels on 28862 panos (25242 positioned by inversion); 2283 labels attached by bearing
- camera height mode `per-pano`
- 13572 of 28830 panos took a measured height; 15258 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 629}
- 13572 heights read from depth/index.csv; spread definition: measured_planes_only

| arm | placement | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---|---:|---:|---:|---|---|---:|---|---:|
| deployed | raycast | 18684 | 64918 | 16791 | 0.826 / 0.876 / 0.887 | 0.249 (2328/9361) | 0.297 | 0.016 (156/9558) | 2.00 |
| ps @ 7.5 m | raycast | 17451 | 64814 | 15928 | 0.827 / 0.877 / 0.888 | 0.195 (1826/9376) | 0.217 | 0.016 (153/9322) | 1.86 |
| fusion_server | server | 18867 | 64847 | 18867 | 0.835 / 0.902 / 0.916 | 0.210 (2027/9637) | 0.265 | 0.038 (361/9571) | 1.96 |
| fusion_server | raycast | 18867 | 64847 | 14989 | 0.829 / 0.876 / 0.886 | 0.100 (937/9358) | 0.110 | 0.026 (229/8659) | 2.02 |
| fusion_server+attach | server | 16584 | 64847 | 16584 | 0.824 / 0.899 / 0.914 | 0.133 (1279/9601) | 0.156 | 0.039 (380/9633) | 1.73 |

Same label set (`label_set: mapped`: 50285 labels, the unmapped AI labels left out of every arm), each arm placed both ways, r = 5 m. fusion_server here: 50252 AI + 33 human labels.

| arm | clusters | split, raycast placement | split, server placement | placed (raycast / server) | covered (raycast / server) | merge (raycast / server) |
|---|---:|---:|---:|---|---|---|
| ps @ 7.5 m | 15580 | 0.160 (1483/9281) | 0.098 (905/9257) | 15294 / 15580 | 0.869 / 0.866 | 0.023 / 0.019 |
| fusion_server | 16823 | 0.098 (918/9356) | 0.169 (1572/9284) | 14959 / 16823 | 0.876 / 0.869 | 0.026 / 0.031 |
| fusion_server+attach | 15020 | 0.098 (918/9356) | 0.105 (968/9242) | 14959 / 15020 | 0.876 / 0.865 | 0.026 / 0.032 |

## Synthesized arms from the rebuilt run (#56: EXPLORATORY here, never the deployed labels' fusion)

## tier 0.3, auto frame

- visible pool: 10685 of 11355 inventory ramps within 20 m of one of 28830 panos (670 excluded)
- 73478 labels synthesized; raycast placed 85466 detections at or above the storage floor
- camera height mode `auto`: auto -> gsv-per-rig
- 28830 of 28830 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2011: median 2.28 m, 22 of 197 measured -> 2.5 m
  - 2012: median 2.41 m, 4 of 310 measured -> 2.5 m
  - 2014: median 2.34 m, 40 of 722 measured -> 2.5 m
  - 2015: median 2.38 m, 8 of 277 measured -> 2.5 m
  - 2016: median 2.41 m, 18 of 319 measured -> 2.5 m
  - 2017: median 2.43 m, 15 of 258 measured -> 2.5 m
  - 2018: median 2.24 m, 46 of 953 measured -> 2.5 m
  - 2019: median 2.32 m, 968 of 1663 measured -> 2.5 m
  - 2021: median 2.37 m, 506 of 926 measured -> 2.5 m
  - 2022: median 2.35 m, 1624 of 3434 measured -> 2.5 m
  - 2023: median 2.37 m, 4562 of 8653 measured -> 2.5 m
  - 2024: median 2.35 m, 5482 of 10621 measured -> 2.5 m
  - 2025: median 2.36 m, 277 of 497 measured -> 2.5 m
- fusion_server+attach: 3600 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 20205 | 73478 | 19654 | 0.887 / 0.928 / 0.937 | 0.274 (2720/9913) | 0.315 | 0.022 (253/11300) | 2.04 |
| ps @ 10 m | 18317 | 73478 | 18046 | 0.878 / 0.923 / 0.934 | 0.194 (1917/9866) | 0.209 | 0.030 (328/10894) | 1.86 |
| ps @ 12.5 m | 17456 | 73478 | 17310 | 0.872 / 0.920 / 0.931 | 0.164 (1611/9833) | 0.172 | 0.035 (378/10759) | 1.78 |
| ps @ 15 m | 16996 | 73478 | 16897 | 0.862 / 0.912 / 0.926 | 0.151 (1475/9748) | 0.158 | 0.039 (421/10696) | 1.74 |
| ps_citywide @ 7.5 m | 20061 | 73478 | 19520 | 0.887 / 0.927 / 0.937 | 0.267 (2647/9907) | 0.304 | 0.023 (254/11269) | 2.02 |
| ps @ 7.5 m (server centroid) | 20205 | 73478 | 20205 | 0.857 / 0.926 / 0.940 | 0.165 (1636/9892) | 0.173 | 0.022 (253/11300) | 2.04 |
| fusion | 17650 | 69812 | 17650 | 0.889 / 0.927 / 0.936 | 0.126 (1245/9901) | 0.133 | 0.035 (344/9879) | 1.78 |
| fusion_server+attach | 17716 | 73478 | 17650 | 0.889 / 0.927 / 0.936 | 0.126 (1245/9901) | 0.133 | 0.035 (344/9879) | 1.79 |

## tier 0.3, 2.6 m frame

- visible pool: 10685 of 11355 inventory ramps within 20 m of one of 28830 panos (670 excluded)
- 73478 labels synthesized; raycast placed 85466 detections at or above the storage floor
- fusion_server+attach: 3597 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 20205 | 73478 | 19654 | 0.886 / 0.926 / 0.935 | 0.286 (2831/9893) | 0.335 | 0.027 (303/11399) | 2.04 |
| ps @ 10 m | 18317 | 73478 | 18046 | 0.879 / 0.921 / 0.932 | 0.203 (1999/9845) | 0.222 | 0.035 (382/10982) | 1.86 |
| ps @ 12.5 m | 17456 | 73478 | 17310 | 0.873 / 0.918 / 0.929 | 0.169 (1660/9814) | 0.179 | 0.040 (431/10813) | 1.78 |
| ps @ 15 m | 16996 | 73478 | 16897 | 0.863 / 0.911 / 0.924 | 0.155 (1514/9738) | 0.163 | 0.045 (478/10737) | 1.75 |
| ps_citywide @ 7.5 m | 20061 | 73478 | 19520 | 0.887 / 0.925 / 0.935 | 0.279 (2757/9888) | 0.324 | 0.027 (304/11369) | 2.03 |
| ps @ 7.5 m (server centroid) | 20205 | 73478 | 20205 | 0.857 / 0.926 / 0.940 | 0.165 (1636/9892) | 0.173 | 0.027 (303/11399) | 2.04 |
| fusion | 17551 | 69812 | 17551 | 0.889 / 0.926 / 0.935 | 0.119 (1176/9898) | 0.125 | 0.037 (361/9854) | 1.77 |
| fusion_server+attach | 17620 | 73478 | 17551 | 0.889 / 0.926 / 0.935 | 0.119 (1176/9898) | 0.125 | 0.037 (361/9854) | 1.78 |

## tier 0.3, per-pano frame

- visible pool: 10685 of 11355 inventory ramps within 20 m of one of 28830 panos (670 excluded)
- 73478 labels synthesized; raycast placed 85546 detections at or above the storage floor
- camera height mode `per-pano`
- 13572 of 28830 panos took a measured height; 15258 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 629}
- 13572 heights read from depth/index.csv; spread definition: measured_planes_only
- fusion_server+attach: 3545 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 20205 | 73478 | 19660 | 0.883 / 0.929 / 0.937 | 0.268 (2664/9922) | 0.308 | 0.024 (267/11271) | 2.04 |
| ps @ 10 m | 18317 | 73478 | 18048 | 0.873 / 0.924 / 0.934 | 0.192 (1898/9868) | 0.207 | 0.031 (338/10890) | 1.86 |
| ps @ 12.5 m | 17456 | 73478 | 17312 | 0.867 / 0.920 / 0.931 | 0.162 (1594/9831) | 0.171 | 0.035 (377/10747) | 1.78 |
| ps @ 15 m | 16996 | 73478 | 16898 | 0.857 / 0.912 / 0.925 | 0.150 (1462/9745) | 0.157 | 0.039 (421/10681) | 1.74 |
| ps_citywide @ 7.5 m | 20061 | 73478 | 19526 | 0.883 / 0.928 / 0.937 | 0.261 (2592/9916) | 0.297 | 0.024 (267/11239) | 2.02 |
| ps @ 7.5 m (server centroid) | 20205 | 73478 | 20205 | 0.857 / 0.926 / 0.940 | 0.165 (1636/9892) | 0.173 | 0.024 (267/11271) | 2.04 |
| fusion | 18094 | 69865 | 18094 | 0.886 / 0.928 / 0.937 | 0.136 (1345/9916) | 0.146 | 0.034 (338/9953) | 1.82 |
| fusion_server+attach | 18162 | 73478 | 18094 | 0.886 / 0.928 / 0.937 | 0.136 (1345/9916) | 0.146 | 0.034 (338/9953) | 1.83 |

## tier 0.55, auto frame

- visible pool: 10685 of 11355 inventory ramps within 20 m of one of 28830 panos (670 excluded)
- 60094 labels synthesized; raycast placed 85466 detections at or above the storage floor
- camera height mode `auto`: auto -> gsv-per-rig
- 28830 of 28830 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2011: median 2.28 m, 22 of 197 measured -> 2.5 m
  - 2012: median 2.41 m, 4 of 310 measured -> 2.5 m
  - 2014: median 2.34 m, 40 of 722 measured -> 2.5 m
  - 2015: median 2.38 m, 8 of 277 measured -> 2.5 m
  - 2016: median 2.41 m, 18 of 319 measured -> 2.5 m
  - 2017: median 2.43 m, 15 of 258 measured -> 2.5 m
  - 2018: median 2.24 m, 46 of 953 measured -> 2.5 m
  - 2019: median 2.32 m, 968 of 1663 measured -> 2.5 m
  - 2021: median 2.37 m, 506 of 926 measured -> 2.5 m
  - 2022: median 2.35 m, 1624 of 3434 measured -> 2.5 m
  - 2023: median 2.37 m, 4562 of 8653 measured -> 2.5 m
  - 2024: median 2.35 m, 5482 of 10621 measured -> 2.5 m
  - 2025: median 2.36 m, 277 of 497 measured -> 2.5 m
- fusion_server+attach: 2080 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 16200 | 60094 | 15906 | 0.855 / 0.896 / 0.906 | 0.174 (1669/9577) | 0.190 | 0.021 (212/9907) | 1.69 |
| ps @ 10 m | 14930 | 60094 | 14807 | 0.844 / 0.889 / 0.900 | 0.111 (1052/9500) | 0.115 | 0.028 (265/9630) | 1.57 |
| ps @ 12.5 m | 14415 | 60094 | 14351 | 0.836 / 0.883 / 0.895 | 0.092 (868/9438) | 0.095 | 0.031 (292/9553) | 1.53 |
| ps @ 15 m | 14135 | 60094 | 14091 | 0.824 / 0.873 / 0.888 | 0.086 (799/9332) | 0.088 | 0.034 (322/9516) | 1.51 |
| ps_citywide @ 7.5 m | 16065 | 60094 | 15773 | 0.854 / 0.895 / 0.905 | 0.164 (1572/9568) | 0.178 | 0.022 (213/9867) | 1.68 |
| ps @ 7.5 m (server centroid) | 16200 | 60094 | 16200 | 0.832 / 0.895 / 0.909 | 0.100 (958/9560) | 0.102 | 0.021 (212/9907) | 1.69 |
| fusion | 14865 | 57972 | 14865 | 0.859 / 0.897 / 0.906 | 0.077 (736/9587) | 0.079 | 0.033 (303/9121) | 1.55 |
| fusion_server+attach | 14907 | 60094 | 14865 | 0.859 / 0.897 / 0.906 | 0.077 (736/9587) | 0.079 | 0.033 (303/9121) | 1.55 |

## tier 0.55, 2.6 m frame

- visible pool: 10685 of 11355 inventory ramps within 20 m of one of 28830 panos (670 excluded)
- 60094 labels synthesized; raycast placed 85466 detections at or above the storage floor
- fusion_server+attach: 2078 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 16200 | 60094 | 15906 | 0.854 / 0.895 / 0.905 | 0.182 (1737/9563) | 0.200 | 0.026 (255/9968) | 1.69 |
| ps @ 10 m | 14930 | 60094 | 14807 | 0.844 / 0.888 / 0.899 | 0.116 (1096/9483) | 0.121 | 0.032 (310/9671) | 1.57 |
| ps @ 12.5 m | 14415 | 60094 | 14351 | 0.836 / 0.882 / 0.894 | 0.094 (890/9422) | 0.098 | 0.035 (335/9579) | 1.53 |
| ps @ 15 m | 14135 | 60094 | 14091 | 0.825 / 0.872 / 0.887 | 0.087 (813/9316) | 0.090 | 0.039 (368/9535) | 1.52 |
| ps_citywide @ 7.5 m | 16065 | 60094 | 15773 | 0.853 / 0.894 / 0.904 | 0.172 (1641/9556) | 0.188 | 0.025 (253/9928) | 1.68 |
| ps @ 7.5 m (server centroid) | 16200 | 60094 | 16200 | 0.832 / 0.895 / 0.909 | 0.100 (958/9560) | 0.102 | 0.026 (255/9968) | 1.69 |
| fusion | 14820 | 57972 | 14820 | 0.857 / 0.895 / 0.905 | 0.074 (709/9568) | 0.076 | 0.034 (314/9102) | 1.55 |
| fusion_server+attach | 14864 | 60094 | 14820 | 0.857 / 0.895 / 0.905 | 0.074 (709/9568) | 0.076 | 0.034 (314/9102) | 1.55 |

## tier 0.55, per-pano frame

- visible pool: 10685 of 11355 inventory ramps within 20 m of one of 28830 panos (670 excluded)
- 60094 labels synthesized; raycast placed 85546 detections at or above the storage floor
- camera height mode `per-pano`
- 13572 of 28830 panos took a measured height; 15258 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 629}
- 13572 heights read from depth/index.csv; spread definition: measured_planes_only
- fusion_server+attach: 2043 labels attached by bearing

| arm | clusters | labels | placed | covered r3 / r5 / r8 | split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |
|---|---:|---:|---:|---|---|---:|---|---:|
| ps @ 7.5 m | 16200 | 60094 | 15909 | 0.851 / 0.897 / 0.907 | 0.169 (1620/9586) | 0.183 | 0.024 (235/9917) | 1.69 |
| ps @ 10 m | 14930 | 60094 | 14808 | 0.841 / 0.889 / 0.900 | 0.108 (1029/9504) | 0.113 | 0.030 (292/9643) | 1.57 |
| ps @ 12.5 m | 14415 | 60094 | 14352 | 0.833 / 0.883 / 0.895 | 0.090 (851/9437) | 0.094 | 0.033 (318/9561) | 1.53 |
| ps @ 15 m | 14135 | 60094 | 14091 | 0.821 / 0.873 / 0.888 | 0.084 (780/9333) | 0.087 | 0.037 (353/9520) | 1.51 |
| ps_citywide @ 7.5 m | 16065 | 60094 | 15776 | 0.851 / 0.896 / 0.906 | 0.159 (1522/9577) | 0.172 | 0.024 (235/9878) | 1.68 |
| ps @ 7.5 m (server centroid) | 16200 | 60094 | 16200 | 0.832 / 0.895 / 0.909 | 0.100 (958/9560) | 0.102 | 0.024 (235/9917) | 1.69 |
| fusion | 15191 | 58007 | 15191 | 0.856 / 0.897 / 0.907 | 0.088 (848/9588) | 0.093 | 0.032 (297/9178) | 1.58 |
| fusion_server+attach | 15235 | 60094 | 15191 | 0.856 / 0.897 / 0.907 | 0.088 (848/9588) | 0.093 | 0.032 (297/9178) | 1.59 |

covered = pool ramps with a cluster assigned (nearest ramp within r) / pool; split = covered ramps with >= 2 clusters / covered; merge = clusters whose members fall on >= 2 distinct ramps with >= 2 members each / clusters with >= 2 assigned members; clusters/covered = all clusters / covered pool ramps. Every cluster is placed at the mean of its members' raycast positions in the frame named, except the `(server centroid)` row. Rule and read: docs/ps-clustering-eval.md, "City-inventory scoring".
