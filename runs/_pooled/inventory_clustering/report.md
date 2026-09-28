# City-inventory clustering score: verdict (#106 Part 2)

Pre-registered rule (docs/ps-clustering-eval.md, "City-inventory scoring"): `fusion` vs `ps @ 7.5 m` at r = 5 m, tier 0.30, common `auto` frame, in both cities; the 2.6 m frame must not reverse it.

**NOT ESTABLISHED**

- gainesville auto: split 0.312 -> 0.211 (-0.101), merge 0.009 -> 0.017 (+0.008), covered 0.864 -> 0.863 (-0.001) -- meets the rule
- gainesville 2.6: split 0.288 -> 0.331 (+0.043), merge 0.024 -> 0.016 (-0.008), covered 0.833 -> 0.851 (+0.018) -- REVERSES
- bend auto: split 0.093 -> 0.048 (-0.046), merge 0.016 -> 0.031 (+0.015), covered 0.896 -> 0.897 (+0.001) -- does NOT meet it
- bend 2.6: split 0.098 -> 0.048 (-0.049), merge 0.020 -> 0.034 (+0.013), covered 0.893 -> 0.897 (+0.003) -- REVERSES

| city | tier | frame | arm | covered r5 | split r5 | merge r5 | clusters |
|---|---|---|---|---|---|---|---:|
| gainesville | 0.3 | auto | ps @ 7.5 m | 0.864 | 0.312 | 0.009 | 7936 |
| gainesville | 0.3 | auto | ps @ 10 m | 0.857 | 0.258 | 0.014 | 7215 |
| gainesville | 0.3 | auto | ps @ 12.5 m | 0.850 | 0.236 | 0.019 | 6820 |
| gainesville | 0.3 | auto | ps @ 15 m | 0.843 | 0.222 | 0.021 | 6571 |
| gainesville | 0.3 | auto | ps_citywide @ 7.5 m | 0.863 | 0.283 | 0.009 | 7781 |
| gainesville | 0.3 | auto | ps @ 7.5 m (server centroid) | 0.871 | 0.268 | 0.009 | 7936 |
| gainesville | 0.3 | auto | fusion | 0.863 | 0.211 | 0.017 | 7151 |
| gainesville | 0.3 | auto | fusion_server+attach | 0.863 | 0.211 | 0.017 | 7307 |
| gainesville | 0.3 | 2.6 | ps @ 7.5 m | 0.833 | 0.288 | 0.024 | 7936 |
| gainesville | 0.3 | 2.6 | ps @ 10 m | 0.827 | 0.230 | 0.027 | 7215 |
| gainesville | 0.3 | 2.6 | ps @ 12.5 m | 0.820 | 0.207 | 0.032 | 6820 |
| gainesville | 0.3 | 2.6 | ps @ 15 m | 0.814 | 0.195 | 0.034 | 6571 |
| gainesville | 0.3 | 2.6 | ps_citywide @ 7.5 m | 0.830 | 0.268 | 0.025 | 7781 |
| gainesville | 0.3 | 2.6 | ps @ 7.5 m (server centroid) | 0.871 | 0.268 | 0.024 | 7936 |
| gainesville | 0.3 | 2.6 | fusion | 0.851 | 0.331 | 0.016 | 8967 |
| gainesville | 0.3 | 2.6 | fusion_server+attach | 0.851 | 0.331 | 0.016 | 9142 |
| gainesville | 0.55 | auto | ps @ 7.5 m | 0.798 | 0.190 | 0.009 | 5096 |
| gainesville | 0.55 | auto | ps @ 10 m | 0.787 | 0.153 | 0.012 | 4742 |
| gainesville | 0.55 | auto | ps @ 12.5 m | 0.774 | 0.140 | 0.017 | 4567 |
| gainesville | 0.55 | auto | ps @ 15 m | 0.761 | 0.134 | 0.023 | 4454 |
| gainesville | 0.55 | auto | ps_citywide @ 7.5 m | 0.795 | 0.159 | 0.011 | 4955 |
| gainesville | 0.55 | auto | ps @ 7.5 m (server centroid) | 0.802 | 0.161 | 0.009 | 5096 |
| gainesville | 0.55 | auto | fusion | 0.798 | 0.120 | 0.015 | 4729 |
| gainesville | 0.55 | auto | fusion_server+attach | 0.798 | 0.120 | 0.015 | 4812 |
| gainesville | 0.55 | 2.6 | ps @ 7.5 m | 0.759 | 0.165 | 0.018 | 5096 |
| gainesville | 0.55 | 2.6 | ps @ 10 m | 0.752 | 0.128 | 0.020 | 4742 |
| gainesville | 0.55 | 2.6 | ps @ 12.5 m | 0.743 | 0.117 | 0.023 | 4567 |
| gainesville | 0.55 | 2.6 | ps @ 15 m | 0.733 | 0.108 | 0.025 | 4454 |
| gainesville | 0.55 | 2.6 | ps_citywide @ 7.5 m | 0.754 | 0.145 | 0.020 | 4955 |
| gainesville | 0.55 | 2.6 | ps @ 7.5 m (server centroid) | 0.802 | 0.161 | 0.018 | 5096 |
| gainesville | 0.55 | 2.6 | fusion | 0.779 | 0.247 | 0.013 | 6252 |
| gainesville | 0.55 | 2.6 | fusion_server+attach | 0.779 | 0.247 | 0.013 | 6347 |
| bend | 0.3 | auto | ps @ 7.5 m | 0.896 | 0.093 | 0.016 | 14650 |
| bend | 0.3 | auto | ps @ 10 m | 0.891 | 0.057 | 0.020 | 13915 |
| bend | 0.3 | auto | ps @ 12.5 m | 0.885 | 0.050 | 0.023 | 13633 |
| bend | 0.3 | auto | ps @ 15 m | 0.876 | 0.051 | 0.025 | 13423 |
| bend | 0.3 | auto | ps_citywide @ 7.5 m | 0.896 | 0.093 | 0.016 | 14650 |
| bend | 0.3 | auto | ps @ 7.5 m (server centroid) | 0.896 | 0.057 | 0.016 | 14650 |
| bend | 0.3 | auto | fusion | 0.897 | 0.048 | 0.031 | 14058 |
| bend | 0.3 | auto | fusion_server+attach | 0.897 | 0.048 | 0.031 | 14094 |
| bend | 0.3 | 2.6 | ps @ 7.5 m | 0.893 | 0.098 | 0.020 | 14650 |
| bend | 0.3 | 2.6 | ps @ 10 m | 0.889 | 0.061 | 0.025 | 13915 |
| bend | 0.3 | 2.6 | ps @ 12.5 m | 0.883 | 0.054 | 0.028 | 13633 |
| bend | 0.3 | 2.6 | ps @ 15 m | 0.873 | 0.054 | 0.029 | 13423 |
| bend | 0.3 | 2.6 | ps_citywide @ 7.5 m | 0.893 | 0.098 | 0.020 | 14650 |
| bend | 0.3 | 2.6 | ps @ 7.5 m (server centroid) | 0.896 | 0.057 | 0.020 | 14650 |
| bend | 0.3 | 2.6 | fusion | 0.897 | 0.048 | 0.034 | 14187 |
| bend | 0.3 | 2.6 | fusion_server+attach | 0.897 | 0.048 | 0.034 | 14223 |
| bend | 0.55 | auto | ps @ 7.5 m | 0.896 | 0.093 | 0.016 | 14650 |
| bend | 0.55 | auto | ps @ 10 m | 0.891 | 0.057 | 0.020 | 13915 |
| bend | 0.55 | auto | ps @ 12.5 m | 0.885 | 0.050 | 0.023 | 13633 |
| bend | 0.55 | auto | ps @ 15 m | 0.876 | 0.051 | 0.025 | 13423 |
| bend | 0.55 | auto | ps_citywide @ 7.5 m | 0.896 | 0.093 | 0.016 | 14650 |
| bend | 0.55 | auto | ps @ 7.5 m (server centroid) | 0.896 | 0.057 | 0.016 | 14650 |
| bend | 0.55 | auto | fusion | 0.897 | 0.048 | 0.031 | 14058 |
| bend | 0.55 | auto | fusion_server+attach | 0.897 | 0.048 | 0.031 | 14094 |
| bend | 0.55 | 2.6 | ps @ 7.5 m | 0.893 | 0.098 | 0.020 | 14650 |
| bend | 0.55 | 2.6 | ps @ 10 m | 0.889 | 0.061 | 0.025 | 13915 |
| bend | 0.55 | 2.6 | ps @ 12.5 m | 0.883 | 0.054 | 0.028 | 13633 |
| bend | 0.55 | 2.6 | ps @ 15 m | 0.873 | 0.054 | 0.029 | 13423 |
| bend | 0.55 | 2.6 | ps_citywide @ 7.5 m | 0.893 | 0.098 | 0.020 | 14650 |
| bend | 0.55 | 2.6 | ps @ 7.5 m (server centroid) | 0.896 | 0.057 | 0.020 | 14650 |
| bend | 0.55 | 2.6 | fusion | 0.897 | 0.048 | 0.034 | 14187 |
| bend | 0.55 | 2.6 | fusion_server+attach | 0.897 | 0.048 | 0.034 | 14223 |
