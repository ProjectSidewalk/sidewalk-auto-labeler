# Corner precision gallery (#169): bundle report

Built 2026-11-06T04:08:42+00:00 by `scripts/corner_gallery.py build --items-from` at `5b93bfec0b9d08587c2f8895a88be2b449a002dd`. Items sha256 `1735f7bd2d17c45429d219f204a70423d8d0e0297f502f8c399488f9c2f04806`.

- Sample: `runs/vancouver/corner_labels169/na_discount_sw/sample.csv` (sha256 `de25b2896277c51df681c1ae1b8fe6277269d381950163334fbdd795b2bf44fc`), 100 corners drawn with seed 169 from 1821 proposals (random.Random(seed).sample(sorted proposal keys, n)).
- Proposals: `runs/vancouver/corner_labels169/na_discount_sw/proposals.jsonl` (sha256 `b64e95cd75f7fc017145b05978795639e21d1e8234164b68717e29ab32b93a44`), built at `80fc070659f7c4f1b7ae2570d96491296bad5f82`, NA read `discount` (settled: True), sidewalk rule applied.
- 99 units, 100 corners; by stratum {'arterial': 35, 'residential': 64, 'signalised': 1}. Only the sampled corners of a unit are shown and rated.
- Views: up to 3 per corner, chosen as in gallery 243. The pano that carries the rule's label is among them at 100 of 100 corners. Corners with no view: 0.
- Crops missing: 0.
- Rubric: gallery 243 v1, unchanged. Reveal: the city inventory, after completion; the blind verdicts are scored.
- Score: `python scripts/corner_labels.py score --proposals <proposals> --sample <sample.csv> --bundle <this bundle> --verdicts <bundle>/verdicts__<rater>.json`.

## Inputs

| input | sha256 |
|---|---|
| sample | `de25b2896277c51df681c1ae1b8fe6277269d381950163334fbdd795b2bf44fc` |
| sample_json | `44df79419422a03f599a18806173ce8e0a6e3f762696c21202392b8ee6266bdb` |
| proposals | `b64e95cd75f7fc017145b05978795639e21d1e8234164b68717e29ab32b93a44` |
| labels_build_json | `2987d6c45203b85f92f7920b318589e26273b27b96c92f906839bf6dc65e0e89` |
| build_json | `da61166a412ac18a0fa12dd4c97116cf8d56839f0dd69655c50cf02824bd199b` |
| corners_full | `ec20606098d0e8b34113b4b328e57d3737b7669d01a37f1d6d722faab4681f8d` |
| inventory | `c4d2497995f7b6261333c3859a668348cd62a36367593cef2dfdfa6b7e020d87` |
| results | `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28` |
| extra:vancouver_posdetect241:results | `69f46b26110df05d1ef52a4431c4b230ceb157941b212fd3d553fd64bb86bf0e` |
| extra:vancouver_posdetect241_gsv:results | `953716d1d7f32f2680c101fb9a4d7ab0933939459704fbc62375042768ca57b2` |

## Units

| order | unit | type | sampled corners | of |
|---:|---|---|---|---:|
| 1 | `vancouver:res:n47207271` | residential | 2 | 4 |
| 2 | `vancouver:res:n47238401` | residential | 0 | 3 |
| 3 | `vancouver:res:n47277515` | residential | 0 | 3 |
| 4 | `vancouver:art:n13538926834` | arterial | 0, 1 | 3 |
| 5 | `vancouver:art:n47372178` | arterial | 0 | 3 |
| 6 | `vancouver:sig:n1237720684` | signalised | 0 | 2 |
| 7 | `vancouver:res:n958723078` | residential | 2 | 3 |
| 8 | `vancouver:res:n47279351` | residential | 2 | 4 |
| 9 | `vancouver:res:n47208364` | residential | 3 | 4 |
| 10 | `vancouver:art:n47305285` | arterial | 2 | 3 |
| 11 | `vancouver:art:n1972799943` | arterial | 0 | 2 |
| 12 | `vancouver:res:n959123136` | residential | 1 | 3 |
| 13 | `vancouver:art:n47354269` | arterial | 0 | 3 |
| 14 | `vancouver:res:n47267506` | residential | 0 | 4 |
| 15 | `vancouver:res:n47231611` | residential | 2 | 3 |
| 16 | `vancouver:res:n47311982` | residential | 0 | 3 |
| 17 | `vancouver:res:n47298427` | residential | 1 | 3 |
| 18 | `vancouver:art:n47255244` | arterial | 1 | 4 |
| 19 | `vancouver:res:n47217968` | residential | 0 | 3 |
| 20 | `vancouver:res:n47308946` | residential | 1 | 3 |
| 21 | `vancouver:res:n47250462` | residential | 2 | 3 |
| 22 | `vancouver:art:n1327846257` | arterial | 1 | 3 |
| 23 | `vancouver:res:n47217157` | residential | 2 | 3 |
| 24 | `vancouver:res:n47380789` | residential | 2 | 3 |
| 25 | `vancouver:res:n47257976` | residential | 1 | 4 |
| 26 | `vancouver:res:n47311987` | residential | 3 | 4 |
| 27 | `vancouver:art:n47366286` | arterial | 1 | 4 |
| 28 | `vancouver:res:n47266847` | residential | 0 | 4 |
| 29 | `vancouver:art:n47217964` | arterial | 2 | 3 |
| 30 | `vancouver:res:n47221578` | residential | 1 | 3 |
| 31 | `vancouver:res:n47296021` | residential | 2 | 3 |
| 32 | `vancouver:res:n47259061` | residential | 2 | 3 |
| 33 | `vancouver:art:n3629347257` | arterial | 1 | 3 |
| 34 | `vancouver:res:n47221555` | residential | 2 | 3 |
| 35 | `vancouver:art:n1327794577` | arterial | 0 | 2 |
| 36 | `vancouver:art:n47231488` | arterial | 3 | 4 |
| 37 | `vancouver:res:n5905779135` | residential | 0 | 2 |
| 38 | `vancouver:art:n47270034` | arterial | 0 | 3 |
| 39 | `vancouver:res:n47245457` | residential | 1 | 3 |
| 40 | `vancouver:art:n47294771` | arterial | 0 | 3 |
| 41 | `vancouver:res:n47242079` | residential | 1 | 3 |
| 42 | `vancouver:res:n47292502` | residential | 2 | 4 |
| 43 | `vancouver:art:n47340275` | arterial | 2 | 3 |
| 44 | `vancouver:art:n47267635` | arterial | 1 | 4 |
| 45 | `vancouver:art:n47287544` | arterial | 1 | 3 |
| 46 | `vancouver:res:n47249341` | residential | 2 | 3 |
| 47 | `vancouver:res:n47241597` | residential | 3 | 4 |
| 48 | `vancouver:art:n47287812` | arterial | 1 | 3 |
| 49 | `vancouver:art:n11080050944` | arterial | 2 | 3 |
| 50 | `vancouver:res:n47344805` | residential | 1 | 3 |
| 51 | `vancouver:res:n47337412` | residential | 2 | 3 |
| 52 | `vancouver:art:n3801437411` | arterial | 1 | 2 |
| 53 | `vancouver:res:n47245823` | residential | 0 | 3 |
| 54 | `vancouver:art:n47266734` | arterial | 2 | 3 |
| 55 | `vancouver:res:n1637998606` | residential | 2 | 3 |
| 56 | `vancouver:res:n47236613` | residential | 0 | 3 |
| 57 | `vancouver:res:n47289064` | residential | 1 | 3 |
| 58 | `vancouver:res:n47311700` | residential | 0 | 3 |
| 59 | `vancouver:res:n7150972574` | residential | 2 | 3 |
| 60 | `vancouver:res:n47378222` | residential | 2 | 4 |
| 61 | `vancouver:res:n10213183520` | residential | 0 | 3 |
| 62 | `vancouver:art:n3901847987` | arterial | 2 | 3 |
| 63 | `vancouver:res:n47253671` | residential | 1 | 4 |
| 64 | `vancouver:res:n47203957` | residential | 2 | 3 |
| 65 | `vancouver:res:n47266845` | residential | 2 | 4 |
| 66 | `vancouver:res:n47236263` | residential | 2 | 3 |
| 67 | `vancouver:res:n4602349091` | residential | 1 | 3 |
| 68 | `vancouver:art:n47259759` | arterial | 0 | 3 |
| 69 | `vancouver:res:n47191612` | residential | 0 | 3 |
| 70 | `vancouver:art:n47227923` | arterial | 0 | 3 |
| 71 | `vancouver:res:n5845256816` | residential | 1 | 3 |
| 72 | `vancouver:art:n47289380` | arterial | 0 | 4 |
| 73 | `vancouver:res:n47377683` | residential | 0 | 3 |
| 74 | `vancouver:art:n7309694849` | arterial | 0 | 3 |
| 75 | `vancouver:res:n47261995` | residential | 0 | 3 |
| 76 | `vancouver:art:n1634520930` | arterial | 0 | 2 |
| 77 | `vancouver:art:n8231027652` | arterial | 2 | 3 |
| 78 | `vancouver:res:n47259718` | residential | 0 | 4 |
| 79 | `vancouver:res:n47245478` | residential | 0 | 3 |
| 80 | `vancouver:res:n47238420` | residential | 2 | 3 |
| 81 | `vancouver:res:n47250842` | residential | 2 | 4 |
| 82 | `vancouver:res:n47237302` | residential | 2 | 3 |
| 83 | `vancouver:res:n47344830` | residential | 2 | 3 |
| 84 | `vancouver:res:n977148364` | residential | 1 | 3 |
| 85 | `vancouver:art:n47222734` | arterial | 1 | 3 |
| 86 | `vancouver:art:n4458820934` | arterial | 1 | 2 |
| 87 | `vancouver:res:n47221560` | residential | 2 | 3 |
| 88 | `vancouver:res:n47225921` | residential | 1 | 3 |
| 89 | `vancouver:art:n47267877` | arterial | 2 | 3 |
| 90 | `vancouver:art:n1327865767` | arterial | 1 | 2 |
| 91 | `vancouver:res:n47250976` | residential | 2 | 3 |
| 92 | `vancouver:art:n47214972` | arterial | 0 | 4 |
| 93 | `vancouver:res:n47316268` | residential | 0 | 3 |
| 94 | `vancouver:res:n47224329` | residential | 2 | 3 |
| 95 | `vancouver:res:n47259568` | residential | 3 | 4 |
| 96 | `vancouver:res:n47254589` | residential | 0 | 3 |
| 97 | `vancouver:art:n47270713` | arterial | 1 | 3 |
| 98 | `vancouver:res:n47232867` | residential | 2 | 3 |
| 99 | `vancouver:res:n47364342` | residential | 0 | 3 |
