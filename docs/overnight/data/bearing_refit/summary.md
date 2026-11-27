Rule 1: B beats A in 1 of 3 open cities -> FAIL
Rule 2: height-robust in 3 of 6 open (city, height) cells -> FAIL

| city | refs (sites, box) | med abs az A | med abs az B | D = A - B [95% CI] | p90 A / B | rule 1 | med row A / B px | med A-B dist m |
|---|---|---:|---:|---|---|---|---|---:|
| richmond | 183 (183, 175) | 3.29 | 2.48 | 0.81 [0.36, 1.40] | 11.17 / 11.67 | pass | 4.2 / 4.9 | 0.80 |
| laurens | 76 (70, 0) | 4.48 | 3.19 | 1.30 [-0.52, 2.56] | 19.20 / 21.91 | fail | 5.1 / 6.8 | 1.52 |
| bayonne | 53 (53, 0) | 9.57 | 15.24 | -5.67 [-8.70, -0.25] | 27.59 / 27.06 | fail | 7.6 / 9.7 | 1.37 |
| paterson | 194 (194, 64) | 2.41 | 1.97 | 0.44 [-0.01, 1.20] | 8.56 / 6.70 | fail | 3.2 / 3.7 | 0.69 |

| city | h m | sites | lost | changed | B unsolvable | med shift A frozen | med shift A re-assoc | med shift B re-assoc | B = 0 share | B/A | rule 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| richmond | 2 | 893 | 5 | 642 | 27 | 1.41 | 1.49 | 0.164 | 0.31 | 0.116 | pass |
| richmond | 3.2 | 861 | 47 | 767 | 17 | 1.22 | 1.40 | 0.269 | 0.13 | 0.221 | pass |
| laurens | 2 | 85 | 1 | 83 | 21 | 1.55 | 2.18 | 0.771 | 0.27 | 0.497 | fail |
| laurens | 3.2 | 103 | 1 | 71 | 3 | 1.42 | 1.57 | 0.159 | 0.34 | 0.112 | pass |
| bayonne | 2 | 598 | 2 | 421 | 81 | 1.32 | 1.79 | 0.435 | 0.43 | 0.329 | fail |
| bayonne | 3.2 | 579 | 54 | 455 | 48 | 1.18 | 1.65 | 0.889 | 0.30 | 0.750 | fail |
| paterson | 2 | 4474 | 58 | 3158 | 19 | 1.12 | 1.25 | 0.131 | 0.30 | 0.117 | pass |
| paterson | 3.2 | 3206 | 1299 | 3043 | 46 | 0.94 | 1.49 | 0.286 | 0.07 | 0.303 | fail |

| city | GT-free views (sites) | med abs az A | med abs az B | D [95% CI] | p90 A / B | unsolvable views (A med) |
|---|---|---:|---:|---|---|---|
| richmond | 7078 (941) | 3.13 | 2.05 | 1.08 [0.96, 1.21] | 12.11 / 11.16 | 157 (6.10) |
| laurens | 556 (109) | 5.23 | 2.98 | 2.24 [1.10, 3.00] | 19.62 / 20.39 | 29 (11.01) |
| bayonne | 3054 (767) | 8.85 | 9.86 | -1.01 [-1.48, -0.53] | 29.19 / 34.58 | 572 (12.47) |
| paterson | 19717 (4552) | 2.41 | 1.53 | 0.88 [0.82, 0.93] | 8.26 / 6.39 | 241 (4.75) |

| city | subset | spread >= | n | med A | med B | D [95% CI] |
|---|---|---:|---:|---:|---:|---|
| richmond | box_member | 10 | 175 | 3.29 | 2.48 | 0.81 [0.31, 1.40] |
| richmond | peak_member | 10 | 8 | 4.01 | 1.96 | 2.05 [-1.23, 5.34] |
| richmond | missed | 10 | 21 | 6.17 | 6.81 | -0.64 [-8.37, 2.97] |
| richmond | member | 0 | 184 | 3.34 | 2.48 | 0.86 [0.35, 1.39] |
| richmond | member | 5 | 184 | 3.34 | 2.48 | 0.86 [0.35, 1.39] |
| richmond | member | 10 | 183 | 3.29 | 2.48 | 0.81 [0.36, 1.40] |
| richmond | member | 15 | 182 | 3.25 | 2.47 | 0.77 [0.32, 1.32] |
| richmond | member | 20 | 181 | 3.21 | 2.47 | 0.74 [0.32, 1.32] |
| richmond | member | 30 | 181 | 3.21 | 2.47 | 0.74 [0.32, 1.32] |
| laurens | box_member | 10 | 0 | — | — | — [—, —] |
| laurens | peak_member | 10 | 76 | 4.48 | 3.19 | 1.30 [-0.52, 2.56] |
| laurens | missed | 10 | 35 | 4.85 | 3.78 | 1.07 [-2.24, 3.79] |
| laurens | member | 0 | 78 | 4.50 | 3.19 | 1.31 [-0.47, 2.70] |
| laurens | member | 5 | 77 | 4.49 | 3.35 | 1.14 [-0.67, 2.50] |
| laurens | member | 10 | 76 | 4.48 | 3.19 | 1.30 [-0.52, 2.56] |
| laurens | member | 15 | 75 | 4.47 | 3.02 | 1.45 [-0.47, 2.61] |
| laurens | member | 20 | 75 | 4.47 | 3.02 | 1.45 [-0.47, 2.61] |
| laurens | member | 30 | 73 | 4.47 | 3.35 | 1.12 [-0.70, 2.38] |
| bayonne | box_member | 10 | 0 | — | — | — [—, —] |
| bayonne | peak_member | 10 | 53 | 9.57 | 15.24 | -5.67 [-8.70, -0.25] |
| bayonne | missed | 10 | 24 | 12.02 | 9.36 | 2.67 [-3.76, 5.62] |
| bayonne | member | 0 | 54 | 8.84 | 15.09 | -6.26 [-8.60, -0.57] |
| bayonne | member | 5 | 54 | 8.84 | 15.09 | -6.26 [-8.60, -0.57] |
| bayonne | member | 10 | 53 | 9.57 | 15.24 | -5.67 [-8.70, -0.25] |
| bayonne | member | 15 | 53 | 9.57 | 15.24 | -5.67 [-8.70, -0.25] |
| bayonne | member | 20 | 52 | 9.84 | 15.09 | -5.26 [-8.16, -0.28] |
| bayonne | member | 30 | 48 | 8.84 | 15.09 | -6.26 [-8.51, -0.29] |
| paterson | box_member | 10 | 64 | 3.05 | 2.28 | 0.77 [-0.71, 1.66] |
| paterson | peak_member | 10 | 130 | 2.35 | 1.70 | 0.65 [0.02, 1.18] |
| paterson | missed | 10 | 19 | 2.52 | 1.98 | 0.54 [-1.72, 1.92] |
| paterson | member | 0 | 196 | 2.41 | 2.01 | 0.40 [-0.01, 1.18] |
| paterson | member | 5 | 195 | 2.41 | 2.00 | 0.41 [-0.01, 1.20] |
| paterson | member | 10 | 194 | 2.41 | 1.97 | 0.44 [-0.01, 1.20] |
| paterson | member | 15 | 192 | 2.41 | 1.92 | 0.49 [0.04, 1.16] |
| paterson | member | 20 | 192 | 2.41 | 1.92 | 0.49 [0.04, 1.16] |
| paterson | member | 30 | 191 | 2.41 | 1.90 | 0.50 [0.02, 1.21] |

| city | parallax bin deg | n | med A | med B |
|---|---|---:|---:|---:|
| richmond | [10,15) | 1 | 15.82 | 25.59 |
| richmond | [15,30) | 1 | 9.01 | 5.25 |
| richmond | [30,60) | 14 | 3.56 | 2.32 |
| richmond | [60,90) | 167 | 3.18 | 2.48 |
| laurens | [10,15) | 1 | 7.23 | 40.62 |
| laurens | [15,30) | 2 | 9.72 | 0.75 |
| laurens | [30,60) | 12 | 4.97 | 1.59 |
| laurens | [60,90) | 61 | 4.19 | 3.71 |
| bayonne | [10,15) | 0 | — | — |
| bayonne | [15,30) | 5 | 10.10 | 21.01 |
| bayonne | [30,60) | 9 | 7.17 | 16.47 |
| bayonne | [60,90) | 39 | 12.87 | 14.95 |
| paterson | [10,15) | 2 | 3.94 | 7.47 |
| paterson | [15,30) | 1 | 5.27 | 5.44 |
| paterson | [30,60) | 33 | 3.57 | 2.46 |
| paterson | [60,90) | 158 | 2.35 | 1.64 |

| city | member refs | unit | B unsolvable (low spread / bad range) | A med az on unsolvable (p90) | A med az on unit |
|---|---:|---:|---|---|---:|
| richmond | 190 | 183 | 7 (1 / 6) | 8.49 (15.02) | 3.29 |
| laurens | 80 | 76 | 4 (2 / 2) | 8.93 (15.46) | 4.48 |
| bayonne | 62 | 53 | 9 (2 / 7) | 4.84 (20.40) | 9.57 |
| paterson | 196 | 194 | 2 (2 / 0) | 9.78 (15.05) | 2.41 |

| city | spread >= | sites >= 3 views | spread ok | B solvable |
|---|---:|---:|---:|---:|
| richmond | 0 | 945 | 945 | 926 |
| richmond | 5 | 945 | 943 | 925 |
| richmond | 10 | 945 | 942 | 925 |
| richmond | 15 | 945 | 939 | 923 |
| richmond | 20 | 945 | 933 | 917 |
| richmond | 30 | 945 | 915 | 901 |
| laurens | 0 | 109 | 109 | 107 |
| laurens | 5 | 109 | 109 | 107 |
| laurens | 10 | 109 | 109 | 107 |
| laurens | 15 | 109 | 109 | 107 |
| laurens | 20 | 109 | 109 | 107 |
| laurens | 30 | 109 | 107 | 105 |
| bayonne | 0 | 793 | 793 | 684 |
| bayonne | 5 | 793 | 793 | 684 |
| bayonne | 10 | 793 | 783 | 681 |
| bayonne | 15 | 793 | 774 | 678 |
| bayonne | 20 | 793 | 763 | 669 |
| bayonne | 30 | 793 | 724 | 641 |
| paterson | 0 | 4552 | 4552 | 4551 |
| paterson | 5 | 4552 | 4552 | 4551 |
| paterson | 10 | 4552 | 4552 | 4551 |
| paterson | 15 | 4552 | 4548 | 4548 |
| paterson | 20 | 4552 | 4544 | 4544 |
| paterson | 30 | 4552 | 4531 | 4531 |

EXPLORATORY (post hoc):

| city | units | group | n | med A | med B | share B better | D [95% CI] |
|---|---|---|---:|---:|---:|---:|---|
| richmond | gtfree | baseline [0,5) m | 43 | 3.76 | 1.91 | 0.70 | 1.85 [0.46, 3.44] |
| richmond | gtfree | baseline [5,10) m | 179 | 5.92 | 3.31 | 0.58 | 2.61 [0.97, 3.56] |
| richmond | gtfree | baseline [10,20) m | 849 | 4.10 | 2.40 | 0.62 | 1.70 [1.19, 2.29] |
| richmond | gtfree | baseline [20,1e+09) m | 6007 | 2.99 | 1.99 | 0.62 | 1.00 [0.86, 1.14] |
| richmond | gtfree | 1 sequence | 380 | 6.15 | 2.56 | 0.71 | 3.59 [2.88, 4.56] |
| richmond | gtfree | >= 2 sequences | 6698 | 3.04 | 2.02 | 0.61 | 1.02 [0.88, 1.14] |
| richmond | gtfree | all | 7078 | 3.13 | 2.05 | 0.62 | 1.08 [0.96, 1.21] |
| richmond | refs_member | baseline [0,5) m | 1 | 9.01 | 5.25 | 1.00 | 3.77 [3.77, 3.77] |
| richmond | refs_member | baseline [5,10) m | 3 | 2.94 | 2.46 | 0.67 | 0.47 [-9.77, 0.47] |
| richmond | refs_member | baseline [10,20) m | 27 | 4.13 | 3.97 | 0.56 | 0.16 [-3.62, 3.45] |
| richmond | refs_member | baseline [20,1e+09) m | 152 | 3.17 | 2.42 | 0.59 | 0.75 [0.26, 1.45] |
| richmond | refs_member | 1 sequence | 7 | 9.01 | 7.75 | 0.57 | 1.27 [-2.80, 5.86] |
| richmond | refs_member | >= 2 sequences | 176 | 3.12 | 2.46 | 0.59 | 0.66 [0.26, 1.29] |
| richmond | refs_member | all | 183 | 3.29 | 2.48 | 0.59 | 0.81 [0.36, 1.40] |
| laurens | gtfree | baseline [0,5) m | 7 | 5.58 | 7.92 | 0.43 | -2.34 [-11.99, 7.83] |
| laurens | gtfree | baseline [5,10) m | 46 | 6.43 | 1.81 | 0.72 | 4.62 [1.40, 9.78] |
| laurens | gtfree | baseline [10,20) m | 155 | 5.70 | 4.53 | 0.57 | 1.17 [-0.64, 3.26] |
| laurens | gtfree | baseline [20,1e+09) m | 348 | 4.93 | 2.91 | 0.66 | 2.02 [0.95, 2.90] |
| laurens | gtfree | 1 sequence | 56 | 5.71 | 2.76 | 0.61 | 2.96 [0.37, 4.88] |
| laurens | gtfree | >= 2 sequences | 500 | 5.20 | 3.07 | 0.64 | 2.14 [0.98, 2.97] |
| laurens | gtfree | all | 556 | 5.23 | 2.98 | 0.64 | 2.24 [1.10, 3.00] |
| laurens | refs_member | baseline [0,5) m | 0 | — | — | — | — [—, —] |
| laurens | refs_member | baseline [5,10) m | 4 | 4.38 | 0.59 | 0.75 | 3.79 [-4.41, 4.47] |
| laurens | refs_member | baseline [10,20) m | 24 | 5.08 | 2.57 | 0.58 | 2.51 [-0.90, 4.62] |
| laurens | refs_member | baseline [20,1e+09) m | 48 | 4.25 | 3.71 | 0.60 | 0.54 [-1.35, 2.15] |
| laurens | refs_member | 1 sequence | 6 | 7.59 | 4.96 | 0.67 | 2.62 [-25.94, 14.29] |
| laurens | refs_member | >= 2 sequences | 70 | 4.38 | 2.98 | 0.60 | 1.40 [-0.70, 2.72] |
| laurens | refs_member | all | 76 | 4.48 | 3.19 | 0.61 | 1.30 [-0.52, 2.56] |
| bayonne | gtfree | baseline [0,5) m | 111 | 12.30 | 14.07 | 0.50 | -1.76 [-8.33, 2.85] |
| bayonne | gtfree | baseline [5,10) m | 268 | 9.01 | 10.27 | 0.41 | -1.25 [-4.53, 0.25] |
| bayonne | gtfree | baseline [10,20) m | 1032 | 9.64 | 10.65 | 0.43 | -1.02 [-2.10, -0.02] |
| bayonne | gtfree | baseline [20,1e+09) m | 1643 | 8.06 | 9.30 | 0.46 | -1.24 [-1.74, -0.68] |
| bayonne | gtfree | 1 sequence | 536 | 8.72 | 8.70 | 0.46 | 0.03 [-1.40, 1.34] |
| bayonne | gtfree | >= 2 sequences | 2518 | 8.89 | 9.99 | 0.44 | -1.10 [-1.67, -0.71] |
| bayonne | gtfree | all | 3054 | 8.85 | 9.86 | 0.45 | -1.01 [-1.48, -0.53] |
| bayonne | refs_member | baseline [0,5) m | 3 | 7.54 | 59.19 | 0.33 | -51.66 [-140.53, 0.20] |
| bayonne | refs_member | baseline [5,10) m | 4 | 17.96 | 14.81 | 0.75 | 3.16 [-1.15, 43.32] |
| bayonne | refs_member | baseline [10,20) m | 14 | 10.83 | 15.67 | 0.50 | -4.83 [-9.59, 4.84] |
| bayonne | refs_member | baseline [20,1e+09) m | 32 | 7.77 | 13.16 | 0.47 | -5.39 [-9.53, 0.63] |
| bayonne | refs_member | 1 sequence | 7 | 7.54 | 8.03 | 0.71 | -0.49 [-9.42, 3.58] |
| bayonne | refs_member | >= 2 sequences | 46 | 12.20 | 16.64 | 0.46 | -4.44 [-9.32, -0.57] |
| bayonne | refs_member | all | 53 | 9.57 | 15.24 | 0.49 | -5.67 [-8.70, -0.25] |
| paterson | gtfree | baseline [0,5) m | 21 | 4.01 | 1.54 | 0.62 | 2.46 [-2.83, 5.15] |
| paterson | gtfree | baseline [5,10) m | 838 | 3.66 | 1.99 | 0.64 | 1.67 [1.34, 2.11] |
| paterson | gtfree | baseline [10,20) m | 5634 | 2.92 | 1.59 | 0.65 | 1.34 [1.22, 1.45] |
| paterson | gtfree | baseline [20,1e+09) m | 13224 | 2.19 | 1.49 | 0.61 | 0.70 [0.64, 0.76] |
| paterson | gtfree | 1 sequence | 0 | — | — | — | — [—, —] |
| paterson | gtfree | >= 2 sequences | 0 | — | — | — | — [—, —] |
| paterson | gtfree | all | 19717 | 2.41 | 1.53 | 0.62 | 0.88 [0.82, 0.93] |
| paterson | refs_member | baseline [0,5) m | 1 | 0.44 | 6.37 | 0.00 | -5.93 [-5.93, -5.93] |
| paterson | refs_member | baseline [5,10) m | 4 | 9.12 | 5.52 | 0.75 | 3.60 [-1.13, 8.34] |
| paterson | refs_member | baseline [10,20) m | 67 | 2.42 | 2.19 | 0.55 | 0.23 [-0.73, 1.32] |
| paterson | refs_member | baseline [20,1e+09) m | 122 | 2.35 | 1.64 | 0.63 | 0.70 [0.09, 1.64] |
| paterson | refs_member | 1 sequence | 0 | — | — | — | — [—, —] |
| paterson | refs_member | >= 2 sequences | 0 | — | — | — | — [—, —] |
| paterson | refs_member | all | 194 | 2.41 | 1.97 | 0.60 | 0.44 [-0.01, 1.20] |
