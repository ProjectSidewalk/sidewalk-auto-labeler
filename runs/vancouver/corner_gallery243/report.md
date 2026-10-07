# Corner present/absent gallery (RampNet#243): bundle report

Built 2026-10-06T16:01:47+00:00 by `scripts/corner_gallery.py` at `d7e974cbf29ef64d51dc5ae2da469d57dfd56223`. Seed **243**. Items sha256 `bbed3d6ddaf1835a66da3afb274c3f338cc80512d4af7c4567c96c1bf47a1459`.

| part | population | drawn | corners | views |
|---|---:|---:|---:|---:|
| false_absence | 35 | 35 | 101 | 297 |
| na_noramp | 944 | 40 | 128 | 374 |
| clean | 333 | 20 | 52 | 142 |

95 units, 281 corners. Corners with no pano within 40 m: 0.
Crops that could not be cut: 0.

## Sampling

- Populations (fusion arm, primary observability, intersections): `false_absence` = absent, >= 1 `Available` point, a #241 target unit; `na_noramp` = absent, every inventory point `NA` with no `RAMPTYPE`; `clean` = absent, no inventory point. Sizes match amendment A8 (35 / 944 / 333).
- Draw: every false absence; then one `random.Random(243)`, `rng.sample` of 40 from the sorted `na_noramp` keys, then 20 from the sorted `clean` keys.
- Review order: sha1 of the unit key, so a unit's slot says nothing about its part. The part is not shown to the rater.
- Views: up to 3 panos per corner from the candidates within 40 m of the corner point (taken from the panos within 25 m of the unit centre or the corner point). One slot always goes to the newest capture date among the candidates; the rest are nearest first among those >= 3 m away (nearer ones only to top up). Crops are 60 deg / 448 px squares of the equirect pano centred on the corner point, projected with a level camera at 2.5 m.

## Inputs

| input | sha256 |
|---|---|
| build_json | `b0d8fcdd4b37473ec4a439668d1c6d1ebf88e46d23e35e28ac6096ea9395afac` |
| corners_full | `ec20606098d0e8b34113b4b328e57d3737b7669d01a37f1d6d722faab4681f8d` |
| targets | `671565c26d984f968844878f52dd708286db4b98c204952faaab14418f45db2c` |
| false_absences | `623077677901638aab6f640615f402b9deb9306b6587eaa122095b9c80aa2cc7` |
| inventory | `c4d2497995f7b6261333c3859a668348cd62a36367593cef2dfdfa6b7e020d87` |
| results | `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28` |
| extra:vancouver_posdetect241:results | `69f46b26110df05d1ef52a4431c4b230ceb157941b212fd3d553fd64bb86bf0e` |
| extra:vancouver_posdetect241_gsv:results | `953716d1d7f32f2680c101fb9a4d7ab0933939459704fbc62375042768ca57b2` |

## Units

| order | unit | part | type | corners |
|---:|---|---|---|---:|
| 1 | `vancouver:res:n47384235` | false_absence | residential | 3 |
| 2 | `vancouver:res:n47219489` | na_noramp | residential | 3 |
| 3 | `vancouver:res:n1637998595` | na_noramp | residential | 3 |
| 4 | `vancouver:art:n47331236` | na_noramp | arterial | 3 |
| 5 | `vancouver:art:n3627781133` | clean | arterial | 3 |
| 6 | `vancouver:res:n47208364` | na_noramp | residential | 4 |
| 7 | `vancouver:res:n47288787` | na_noramp | residential | 3 |
| 8 | `vancouver:art:n47345153` | na_noramp | arterial | 3 |
| 9 | `vancouver:res:n47231852` | false_absence | residential | 3 |
| 10 | `vancouver:art:n47335720` | na_noramp | arterial | 3 |
| 11 | `vancouver:res:n47267448` | na_noramp | residential | 4 |
| 12 | `vancouver:res:n959123136` | clean | residential | 3 |
| 13 | `vancouver:res:n47281487` | na_noramp | residential | 3 |
| 14 | `vancouver:art:n47245550` | na_noramp | arterial | 3 |
| 15 | `vancouver:res:n47270098` | false_absence | residential | 3 |
| 16 | `vancouver:res:n2623620122` | clean | residential | 2 |
| 17 | `vancouver:res:n47397759` | na_noramp | residential | 3 |
| 18 | `vancouver:res:n47240587` | na_noramp | residential | 3 |
| 19 | `vancouver:res:n47305361` | false_absence | residential | 3 |
| 20 | `vancouver:art:n47190958` | clean | arterial | 3 |
| 21 | `vancouver:res:n47287936` | na_noramp | residential | 4 |
| 22 | `vancouver:sig:n1642453647` | clean | signalised | 2 |
| 23 | `vancouver:res:n3372672500` | false_absence | residential | 3 |
| 24 | `vancouver:res:n47232326` | false_absence | residential | 3 |
| 25 | `vancouver:res:n47318252` | na_noramp | residential | 4 |
| 26 | `vancouver:art:n1803133375` | na_noramp | arterial | 2 |
| 27 | `vancouver:art:n13059332216` | false_absence | arterial | 2 |
| 28 | `vancouver:art:n47270030` | false_absence | arterial | 3 |
| 29 | `vancouver:art:n47290563` | false_absence | arterial | 3 |
| 30 | `vancouver:art:n3629347257` | clean | arterial | 3 |
| 31 | `vancouver:res:n47234866` | na_noramp | residential | 3 |
| 32 | `vancouver:res:n47318468` | na_noramp | residential | 3 |
| 33 | `vancouver:art:n47245437` | na_noramp | arterial | 3 |
| 34 | `vancouver:res:n47236184` | na_noramp | residential | 3 |
| 35 | `vancouver:art:n3699326114` | false_absence | arterial | 2 |
| 36 | `vancouver:res:n47215596` | na_noramp | residential | 3 |
| 37 | `vancouver:art:n4612512437` | clean | arterial | 3 |
| 38 | `vancouver:res:n47236718` | false_absence | residential | 3 |
| 39 | `vancouver:res:n13090709554` | false_absence | residential | 3 |
| 40 | `vancouver:res:n47231855` | na_noramp | residential | 3 |
| 41 | `vancouver:res:n47311369` | false_absence | residential | 3 |
| 42 | `vancouver:res:n47258942` | clean | residential | 3 |
| 43 | `vancouver:res:n47362725` | clean | residential | 3 |
| 44 | `vancouver:res:n4598685562` | na_noramp | residential | 3 |
| 45 | `vancouver:res:n47287760` | false_absence | residential | 3 |
| 46 | `vancouver:res:n47289129` | na_noramp | residential | 3 |
| 47 | `vancouver:res:n47278131` | na_noramp | residential | 3 |
| 48 | `vancouver:res:n47286143` | false_absence | residential | 3 |
| 49 | `vancouver:res:n958722944` | clean | residential | 2 |
| 50 | `vancouver:res:n47355183` | na_noramp | residential | 3 |
| 51 | `vancouver:art:n47340212` | false_absence | arterial | 2 |
| 52 | `vancouver:res:n47380807` | false_absence | residential | 4 |
| 53 | `vancouver:art:n47244995` | na_noramp | arterial | 3 |
| 54 | `vancouver:res:n3699326241` | false_absence | residential | 3 |
| 55 | `vancouver:res:n47237304` | clean | residential | 3 |
| 56 | `vancouver:res:n47334787` | false_absence | residential | 3 |
| 57 | `vancouver:res:n47299669` | false_absence | residential | 3 |
| 58 | `vancouver:art:n1614603561` | clean | arterial | 2 |
| 59 | `vancouver:res:n47406308` | false_absence | residential | 3 |
| 60 | `vancouver:art:n1466256854` | clean | arterial | 2 |
| 61 | `vancouver:res:n7010814367` | clean | residential | 3 |
| 62 | `vancouver:res:n4602349091` | clean | residential | 3 |
| 63 | `vancouver:res:n12444613588` | false_absence | residential | 3 |
| 64 | `vancouver:res:n47303605` | na_noramp | residential | 3 |
| 65 | `vancouver:art:n1634536176` | clean | arterial | 2 |
| 66 | `vancouver:res:n47302452` | na_noramp | residential | 3 |
| 67 | `vancouver:res:n47263718` | na_noramp | residential | 4 |
| 68 | `vancouver:res:n47368067` | na_noramp | residential | 3 |
| 69 | `vancouver:art:n1375008444` | false_absence | arterial | 3 |
| 70 | `vancouver:res:n47259775` | na_noramp | residential | 4 |
| 71 | `vancouver:art:n3694670454` | false_absence | arterial | 2 |
| 72 | `vancouver:res:n47261571` | false_absence | residential | 3 |
| 73 | `vancouver:art:n12783988563` | false_absence | arterial | 3 |
| 74 | `vancouver:art:n13090709646` | false_absence | arterial | 3 |
| 75 | `vancouver:res:n47342337` | false_absence | residential | 3 |
| 76 | `vancouver:res:n47397103` | na_noramp | residential | 3 |
| 77 | `vancouver:res:n13538954649` | false_absence | residential | 3 |
| 78 | `vancouver:res:n47307691` | na_noramp | residential | 3 |
| 79 | `vancouver:res:n47249862` | na_noramp | residential | 3 |
| 80 | `vancouver:res:n47225933` | na_noramp | residential | 4 |
| 81 | `vancouver:art:n47231069` | na_noramp | arterial | 4 |
| 82 | `vancouver:art:n47325352` | false_absence | arterial | 3 |
| 83 | `vancouver:art:n1547961037` | clean | arterial | 2 |
| 84 | `vancouver:art:n47282622` | false_absence | arterial | 3 |
| 85 | `vancouver:res:n47332188` | na_noramp | residential | 4 |
| 86 | `vancouver:art:n1215066991` | false_absence | arterial | 2 |
| 87 | `vancouver:art:n3639899149` | clean | arterial | 3 |
| 88 | `vancouver:res:n47288915` | na_noramp | residential | 3 |
| 89 | `vancouver:res:n47387262` | na_noramp | residential | 3 |
| 90 | `vancouver:art:n47308718` | clean | arterial | 2 |
| 91 | `vancouver:art:n47337266` | false_absence | arterial | 3 |
| 92 | `vancouver:res:n47370116` | na_noramp | residential | 3 |
| 93 | `vancouver:res:n47240880` | false_absence | residential | 3 |
| 94 | `vancouver:res:n3642857970` | false_absence | residential | 3 |
| 95 | `vancouver:res:n47300698` | clean | residential | 3 |
