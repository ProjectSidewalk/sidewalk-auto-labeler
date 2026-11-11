# Corner precision gallery (#169): bundle report

Built 2026-11-06T13:59:49+00:00 by `scripts/corner_gallery.py build --items-from` at `bb41377e3c282680b97c6135e750a3cbf58fa69e`. Items sha256 `e7648a487a92ef508d488e40d8425de333f0d3719d2412a72a80138e15e26910`.

- Sample: `runs/vancouver/corner_labels169_split/na_discount_sw/sample.csv` (sha256 `14911083669d3fa9793754b926e85cd7d1a0348218d82b44acf1f91c0eaf04ce`), 100 corners drawn with seed 169 from 3070 proposals (random.Random(seed).sample(sorted proposal keys, n)).
- Proposals: `runs/vancouver/corner_labels169_split/na_discount_sw/proposals.jsonl` (sha256 `1562e076edaf7e5c37f6eada0a2e32275fa86f0f45f7caeb20d21d25abf1d4d0`), built at `30168fddeb6e51ab79760b63da845ec10fb26b12`, NA read `discount` (settled: True), sidewalk rule applied.
- 98 units, 100 corners; by stratum {'arterial': 34, 'residential': 66}. Only the sampled corners of a unit are shown and rated.
- Views: up to 6 per corner, chosen as in gallery 243 (newest-date slot, then distance order); landscape crops 120 x 60 deg at 896 x 448 px. The pano that carries the rule's label is among them at 100 of 100 corners. Corners with no view: 0.
- Crops missing: 0.
- Rubric: v2 (gallery 243 v1 plus the phantom-leg / two-landing paragraph). Reveal: the city inventory, after completion; the blind verdicts are scored.
- Score: `python scripts/corner_labels.py score --proposals <proposals> --sample <sample.csv> --bundle <this bundle> --verdicts <bundle>/verdicts__<rater>.json`.
- Split-wide corner table: 76 units carry phantom legs; 67 of 100 sampled corners are bounded by one (the far side of a T); 5 are still wider than 150 deg.

## Inputs

| input | sha256 |
|---|---|
| sample | `14911083669d3fa9793754b926e85cd7d1a0348218d82b44acf1f91c0eaf04ce` |
| sample_json | `388751db7c9e42b41be17455427ecaa25eb028e551b4e3465d5820ac0bb7bc60` |
| proposals | `1562e076edaf7e5c37f6eada0a2e32275fa86f0f45f7caeb20d21d25abf1d4d0` |
| labels_build_json | `f07a33df572f8e48f4a49571c861f417ff3bda020bd09ef3ff1e77b2c749732f` |
| build_json | `da61166a412ac18a0fa12dd4c97116cf8d56839f0dd69655c50cf02824bd199b` |
| corners_full | `b50da32e1bb0b86f5d93fcd0de0cead1be620d8ac39f309d3abd2461f87ce9fc` |
| inventory | `c4d2497995f7b6261333c3859a668348cd62a36367593cef2dfdfa6b7e020d87` |
| results | `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28` |
| extra:vancouver_posdetect241:results | `69f46b26110df05d1ef52a4431c4b230ceb157941b212fd3d553fd64bb86bf0e` |
| extra:vancouver_posdetect241_gsv:results | `953716d1d7f32f2680c101fb9a4d7ab0933939459704fbc62375042768ca57b2` |

## Units

| order | unit | type | sampled corners | of | phantom legs |
|---:|---|---|---|---:|---|
| 1 | `vancouver:res:n47333440` | residential | 3 | 4 | 12 |
| 2 | `vancouver:res:n47383713` | residential | 0 | 4 | 91 |
| 3 | `vancouver:res:n47277515` | residential | 0, 1 | 3 | - |
| 4 | `vancouver:res:n47242030` | residential | 0 | 4 | 61 |
| 5 | `vancouver:res:n47226535` | residential | 0 | 4 | 90 |
| 6 | `vancouver:art:n47236177` | arterial | 0 | 4 | 90 |
| 7 | `vancouver:art:n47342408` | arterial | 3 | 4 | - |
| 8 | `vancouver:art:n47277537` | arterial | 2 | 4 | 180 |
| 9 | `vancouver:art:n3072089610` | arterial | 2 | 4 | 270 |
| 10 | `vancouver:res:n47288872` | residential | 1 | 4 | - |
| 11 | `vancouver:res:n47279351` | residential | 2 | 4 | - |
| 12 | `vancouver:art:n47372480` | arterial | 2 | 4 | 359 |
| 13 | `vancouver:res:n47223700` | residential | 3 | 4 | 359 |
| 14 | `vancouver:res:n47258012` | residential | 3 | 4 | 90 |
| 15 | `vancouver:res:n47287088` | residential | 2 | 4 | 325 |
| 16 | `vancouver:art:n47311108` | arterial | 0 | 4 | 89 |
| 17 | `vancouver:art:n1972799943` | arterial | 0 | 2 | - |
| 18 | `vancouver:res:n47226954` | residential | 0 | 4 | 179 |
| 19 | `vancouver:res:n1375314821` | residential | 3 | 4 | 180 |
| 20 | `vancouver:res:n47293515` | residential | 3 | 4 | 347 |
| 21 | `vancouver:art:n47259620` | arterial | 1 | 4 | 180 |
| 22 | `vancouver:res:n47270720` | residential | 0 | 4 | 90 |
| 23 | `vancouver:res:n1326714827` | residential | 0 | 4 | 94 |
| 24 | `vancouver:res:n959123524` | residential | 1 | 4 | 180 |
| 25 | `vancouver:res:n47277531` | residential | 0 | 4 | 36 |
| 26 | `vancouver:res:n1326925816` | residential | 0 | 3 | - |
| 27 | `vancouver:res:n47298427` | residential | 1 | 4 | 271 |
| 28 | `vancouver:res:n47228143` | residential | 3 | 4 | 1 |
| 29 | `vancouver:res:n47240407` | residential | 1 | 4 | 283 |
| 30 | `vancouver:res:n47269362` | residential | 3 | 4 | 358 |
| 31 | `vancouver:art:n47337119` | arterial | 1 | 4 | 89 |
| 32 | `vancouver:res:n47249350` | residential | 0 | 4 | - |
| 33 | `vancouver:res:n47384886` | residential | 0 | 4 | 178 |
| 34 | `vancouver:res:n958723095` | residential | 0, 1 | 4 | 180 |
| 35 | `vancouver:res:n47407428` | residential | 1 | 4 | 91 |
| 36 | `vancouver:art:n47270820` | arterial | 3 | 4 | 270 |
| 37 | `vancouver:art:n47397494` | arterial | 0 | 4 | 0 |
| 38 | `vancouver:art:n47306286` | arterial | 1 | 4 | 270 |
| 39 | `vancouver:res:n47263777` | residential | 1 | 4 | 179 |
| 40 | `vancouver:art:n47351587` | arterial | 3 | 4 | 271 |
| 41 | `vancouver:res:n4616972801` | residential | 2 | 4 | 158 |
| 42 | `vancouver:art:n47301278` | arterial | 0 | 4 | 180 |
| 43 | `vancouver:res:n47278845` | residential | 1 | 4 | 270 |
| 44 | `vancouver:art:n47270034` | arterial | 0 | 4 | 269 |
| 45 | `vancouver:res:n47304590` | residential | 2 | 4 | 270 |
| 46 | `vancouver:art:n47257305` | arterial | 3 | 4 | 0 |
| 47 | `vancouver:res:n47227519` | residential | 1 | 4 | 106 |
| 48 | `vancouver:res:n47259878` | residential | 0 | 2 | - |
| 49 | `vancouver:res:n47277936` | residential | 3 | 4 | 360 |
| 50 | `vancouver:res:n47306242` | residential | 1 | 3 | - |
| 51 | `vancouver:art:n47231494` | arterial | 3 | 4 | - |
| 52 | `vancouver:res:n47285879` | residential | 3 | 4 | 351 |
| 53 | `vancouver:res:n47207366` | residential | 0 | 4 | - |
| 54 | `vancouver:res:n7004605714` | residential | 1 | 4 | 90 |
| 55 | `vancouver:art:n47321239` | arterial | 3 | 4 | 90 |
| 56 | `vancouver:art:n3641331150` | arterial | 2 | 4 | 358 |
| 57 | `vancouver:res:n47304528` | residential | 1 | 4 | 192 |
| 58 | `vancouver:res:n47255336` | residential | 2 | 4 | 180 |
| 59 | `vancouver:art:n47296495` | arterial | 0 | 4 | 1 |
| 60 | `vancouver:res:n1637998606` | residential | 3 | 4 | 18 |
| 61 | `vancouver:art:n47321074` | arterial | 0 | 4 | 0 |
| 62 | `vancouver:art:n1627708192` | arterial | 1 | 4 | 180 |
| 63 | `vancouver:res:n47259574` | residential | 3 | 4 | 148 |
| 64 | `vancouver:res:n47246605` | residential | 2 | 4 | 320 |
| 65 | `vancouver:art:n3811642641` | arterial | 1 | 2 | - |
| 66 | `vancouver:art:n47307735` | arterial | 2 | 4 | 180 |
| 67 | `vancouver:res:n47220759` | residential | 2 | 4 | 186 |
| 68 | `vancouver:res:n1375322788` | residential | 0 | 4 | 90 |
| 69 | `vancouver:res:n47225182` | residential | 1 | 4 | 360 |
| 70 | `vancouver:art:n47262047` | arterial | 1 | 4 | 91 |
| 71 | `vancouver:res:n47260796` | residential | 3 | 4 | 271 |
| 72 | `vancouver:art:n958723115` | arterial | 1 | 4 | 270 |
| 73 | `vancouver:art:n2966616369` | arterial | 1 | 4 | 90 |
| 74 | `vancouver:res:n47289231` | residential | 0 | 4 | 91 |
| 75 | `vancouver:res:n47283068` | residential | 3 | 4 | 347 |
| 76 | `vancouver:res:n47278142` | residential | 1 | 4 | 90 |
| 77 | `vancouver:res:n47241615` | residential | 2 | 4 | - |
| 78 | `vancouver:art:n47196855` | arterial | 3 | 4 | 34 |
| 79 | `vancouver:art:n1587312220` | arterial | 1 | 2 | - |
| 80 | `vancouver:art:n47226335` | arterial | 3 | 4 | 271 |
| 81 | `vancouver:res:n47368091` | residential | 3 | 4 | 89 |
| 82 | `vancouver:res:n923872755` | residential | 3 | 4 | 64 |
| 83 | `vancouver:res:n47253754` | residential | 0 | 4 | - |
| 84 | `vancouver:art:n47336088` | arterial | 1 | 4 | 180 |
| 85 | `vancouver:art:n11232645814` | arterial | 0 | 2 | - |
| 86 | `vancouver:res:n47360202` | residential | 0 | 4 | - |
| 87 | `vancouver:res:n47278927` | residential | 2 | 4 | 356 |
| 88 | `vancouver:art:n47236827` | arterial | 3 | 4 | 354 |
| 89 | `vancouver:res:n47316039` | residential | 0 | 4 | - |
| 90 | `vancouver:art:n3072092946` | arterial | 2 | 4 | 0 |
| 91 | `vancouver:res:n47267482` | residential | 0 | 4 | 90 |
| 92 | `vancouver:res:n47269572` | residential | 3 | 4 | - |
| 93 | `vancouver:res:n47232867` | residential | 0 | 4 | 299 |
| 94 | `vancouver:res:n47252453` | residential | 1 | 4 | - |
| 95 | `vancouver:res:n47292364` | residential | 2 | 4 | - |
| 96 | `vancouver:res:n4602349127` | residential | 0 | 6 | 241, 298 |
| 97 | `vancouver:res:n47242021` | residential | 0 | 4 | 105 |
| 98 | `vancouver:res:n47240416` | residential | 0 | 3 | - |
