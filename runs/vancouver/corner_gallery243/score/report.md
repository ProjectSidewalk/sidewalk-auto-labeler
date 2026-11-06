# Corner present/absent gallery (RampNet#243): scores

Items sha256 `bbed3d6ddaf1835a66da3afb274c3f338cc80512d4af7c4567c96c1bf47a1459`, seed 243. Scored on the blind verdicts (given before the city inventory was shown), every unit that has them; the final verdicts, complete units only, are the sensitivity read. Definitions: the docstring of `scripts/corner_gallery_score.py`.

## Rater `jonf`

95 of 95 units complete; 95 with blind verdicts (0 reopened and not completed again); 1 with a verdict changed after the inventory was shown; 2 with absence reasons added after it (4 corners; reported on their own row, never in the blind kinds of absence or in present/absent).

### `NA` with no `RAMPTYPE`: does the city mean "no ramp"?

- Units rated absent at every corner, of units decided: **39/39 = 1.000 [0.910, 1.000]** (blind); final 39/39 = 1.000 [0.910, 1.000].
- Outcomes: {'present': 0, 'absent': 39, 'undetermined': 1} of 40 scored (population 944, sample 40). Conservative (undetermined counted against): 39/40 = 0.975 [0.871, 0.996].
- Corners holding the `NA` points: {'present': 0, 'absent': 99, 'cant_tell': 1}; absent share 99/99 = 1.000 [0.963, 1.000]; kinds of absence (blind reasons only) {'curb_no_ramp': 7, 'no_sidewalk': 59, 'unspecified': 33}.
- Reasons added after the reveal at those corners (absent before and after, no blind reason; can be biased by the inventory): {'curb_no_ramp': 0, 'no_sidewalk': 0}.

### The 35 false absences (35 of 35 scored)

- Classes: {'miss_at_inventory_corner': 4, 'miss_elsewhere': 0, 'artifact_built_after_imagery': 16, 'artifact_inventory_or_geometry': 15, 'undetermined': 0}.
- Real misses 4, artifacts 31; real-miss share of decided: **4/35 = 0.114 [0.045, 0.260]** (blind); final 4/35 = 0.114 [0.045, 0.260].
- Units where an inventory corner's newest crop shown is older than the newest pano available to the build: 0 (expected 0; per-corner dates are under `dating` in score.json).

### Control: clean absences

- Rated absent at every corner, of decided: **17/18 = 0.944 [0.742, 0.990]**; outcomes {'present': 1, 'absent': 17, 'undetermined': 2}; corner level 48/50 = 0.960 [0.865, 0.989].

### Units rated with live imagery

3 unit(s): `vancouver:res:n47287760`, `vancouver:res:n47311369`, `vancouver:res:n47362725`. Live imagery (a look-around link to Google Street View or Project Sidewalk Explore) is neither pinned by `images.sha256` nor dated like the crops, and the false-absence dating compares the install date with the crops only. The blind read without these units:

- `NA` units rated absent at every corner, of decided: 39/39 = 1.000 [0.910, 1.000] (with them 39/39 = 1.000 [0.910, 1.000]).
- False absences: classes {'miss_at_inventory_corner': 3, 'miss_elsewhere': 0, 'artifact_built_after_imagery': 16, 'artifact_inventory_or_geometry': 14, 'undetermined': 0}; real-miss share 3/33 = 0.091 [0.031, 0.236] (with them 4/35 = 0.114 [0.045, 0.260]).
- Control: 16/17 = 0.941 [0.730, 0.990] (with them 17/18 = 0.944 [0.742, 0.990]).

### Stratified estimate (descriptive, not the decision rule)

Absence precision against the rater over the sampled strata (1312 of 1323 absent units; the other 11 are the 9 non-target false absences and 2 'other' units): 0.983, 95% [0.926, 1.000] (conservative, ad hoc: Agresti-Coull per-stratum variance, normal interval, clipped to [0, 1]).

### Per unit

| unit | part | outcome (blind) | false-absence class | live imagery |
|---|---|---|---|---|
| `vancouver:res:n47384235` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n47231852` | false_absence | present | miss_at_inventory_corner |  |
| `vancouver:res:n47270098` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n47305361` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:res:n3372672500` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:res:n47232326` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:art:n13059332216` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:art:n47270030` | false_absence | present | miss_at_inventory_corner |  |
| `vancouver:art:n47290563` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:art:n3699326114` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n47236718` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:res:n13090709554` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n47311369` | false_absence | absent | artifact_inventory_or_geometry | yes |
| `vancouver:res:n47287760` | false_absence | present | miss_at_inventory_corner | yes |
| `vancouver:res:n47286143` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:art:n47340212` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:res:n47380807` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n3699326241` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n47334787` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n47299669` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:res:n47406308` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n12444613588` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:art:n1375008444` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:art:n3694670454` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:res:n47261571` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:art:n12783988563` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:art:n13090709646` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n47342337` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:res:n13538954649` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:art:n47325352` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:art:n47282622` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:art:n1215066991` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:art:n47337266` | false_absence | present | miss_at_inventory_corner |  |
| `vancouver:res:n47240880` | false_absence | absent | artifact_inventory_or_geometry |  |
| `vancouver:res:n3642857970` | false_absence | absent | artifact_built_after_imagery |  |
| `vancouver:res:n47219489` | na_noramp | absent |  |  |
| `vancouver:res:n1637998595` | na_noramp | absent |  |  |
| `vancouver:art:n47331236` | na_noramp | undetermined |  |  |
| `vancouver:res:n47208364` | na_noramp | absent |  |  |
| `vancouver:res:n47288787` | na_noramp | absent |  |  |
| `vancouver:art:n47345153` | na_noramp | absent |  |  |
| `vancouver:art:n47335720` | na_noramp | absent |  |  |
| `vancouver:res:n47267448` | na_noramp | absent |  |  |
| `vancouver:res:n47281487` | na_noramp | absent |  |  |
| `vancouver:art:n47245550` | na_noramp | absent |  |  |
| `vancouver:res:n47397759` | na_noramp | absent |  |  |
| `vancouver:res:n47240587` | na_noramp | absent |  |  |
| `vancouver:res:n47287936` | na_noramp | absent |  |  |
| `vancouver:res:n47318252` | na_noramp | absent |  |  |
| `vancouver:art:n1803133375` | na_noramp | absent |  |  |
| `vancouver:res:n47234866` | na_noramp | absent |  |  |
| `vancouver:res:n47318468` | na_noramp | absent |  |  |
| `vancouver:art:n47245437` | na_noramp | absent |  |  |
| `vancouver:res:n47236184` | na_noramp | absent |  |  |
| `vancouver:res:n47215596` | na_noramp | absent |  |  |
| `vancouver:res:n47231855` | na_noramp | absent |  |  |
| `vancouver:res:n4598685562` | na_noramp | absent |  |  |
| `vancouver:res:n47289129` | na_noramp | absent |  |  |
| `vancouver:res:n47278131` | na_noramp | absent |  |  |
| `vancouver:res:n47355183` | na_noramp | absent |  |  |
| `vancouver:art:n47244995` | na_noramp | absent |  |  |
| `vancouver:res:n47303605` | na_noramp | absent |  |  |
| `vancouver:res:n47302452` | na_noramp | absent |  |  |
| `vancouver:res:n47263718` | na_noramp | absent |  |  |
| `vancouver:res:n47368067` | na_noramp | absent |  |  |
| `vancouver:res:n47259775` | na_noramp | absent |  |  |
| `vancouver:res:n47397103` | na_noramp | absent |  |  |
| `vancouver:res:n47307691` | na_noramp | absent |  |  |
| `vancouver:res:n47249862` | na_noramp | absent |  |  |
| `vancouver:res:n47225933` | na_noramp | absent |  |  |
| `vancouver:art:n47231069` | na_noramp | absent |  |  |
| `vancouver:res:n47332188` | na_noramp | absent |  |  |
| `vancouver:res:n47288915` | na_noramp | absent |  |  |
| `vancouver:res:n47387262` | na_noramp | absent |  |  |
| `vancouver:res:n47370116` | na_noramp | absent |  |  |
| `vancouver:art:n3627781133` | clean | absent |  |  |
| `vancouver:res:n959123136` | clean | absent |  |  |
| `vancouver:res:n2623620122` | clean | absent |  |  |
| `vancouver:art:n47190958` | clean | undetermined |  |  |
| `vancouver:sig:n1642453647` | clean | absent |  |  |
| `vancouver:art:n3629347257` | clean | undetermined |  |  |
| `vancouver:art:n4612512437` | clean | absent |  |  |
| `vancouver:res:n47258942` | clean | absent |  |  |
| `vancouver:res:n47362725` | clean | absent |  | yes |
| `vancouver:res:n958722944` | clean | absent |  |  |
| `vancouver:res:n47237304` | clean | present |  |  |
| `vancouver:art:n1614603561` | clean | absent |  |  |
| `vancouver:art:n1466256854` | clean | absent |  |  |
| `vancouver:res:n7010814367` | clean | absent |  |  |
| `vancouver:res:n4602349091` | clean | absent |  |  |
| `vancouver:art:n1634536176` | clean | absent |  |  |
| `vancouver:art:n1547961037` | clean | absent |  |  |
| `vancouver:art:n3639899149` | clean | absent |  |  |
| `vancouver:art:n47308718` | clean | absent |  |  |
| `vancouver:res:n47300698` | clean | absent |  |  |

