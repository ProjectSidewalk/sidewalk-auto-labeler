# Onboarding Bend, OR to Project Sidewalk (manual runbook)

End-to-end steps to stand up a **Bend, OR** Project Sidewalk instance and then push
RampNet curb-ramp AI labels to it via this repo (`sidewalk-auto-labeler`).

Two repos are involved:
- **`ProjectSidewalk_Webpage`** (`D:\git\ProjectSidewalk_Webpage`) — build the city (steps 1–9).
- **`sidewalk-auto-labeler`** (this repo) — generate + submit labels (step 10).

Authoritative facts in this runbook were traced from the actual scripts
(`db/scripts/fill-new-schema.sh`, `create-new-schema.sh`, `check_streets_for_imagery.py`,
`hide-streets-without-imagery.sh`) and `conf/cityparams.conf`.

Chosen identifiers (change if you prefer):
- city-id: **`bend-or`**
- schema / db user: **`sidewalk_bend`**

---

## Phase A — Acquire the data

### A1. Neighborhood (region) boundaries
Download Bend's open-data Neighborhood Districts layer:
```
https://data.bendoregon.gov/datasets/bendoregon::neighborhood-districts.geojson
```
- 16 features, `NAME` field. **CRS is EPSG:2270** (Oregon State Plane) — must be reprojected.
- 13 named neighborhoods + **3 polygons named "Undesignated"** → you must resolve these
  (regions need real, unique names). Options: merge each into the adjacent neighborhood, or
  give them names (e.g. "North Undesignated"). Merging is simplest.

### A2. Street network (from OpenStreetMap, not the city portal)
Project Sidewalk builds streets from OSM. Extract OSM data covering Bend's bounding box
(`bend.geojson` extent: lon −121.382…−121.243, lat 43.999…44.124):
- Easiest: **BBBike.org** custom extract, or QGIS QuickOSM plugin.
- Keep only these `highway` classes: `trunk, primary, secondary, tertiary, residential,
  unclassified, pedestrian, living_street`.

---

## Phase B — Process in QGIS → two PostGIS tables

The whole point of this phase is to produce two tables that **exactly match the contract
`fill-new-schema.sh` reads**, both in **SRID 4326**:

### `qgis_region` (required columns)
| column | notes |
|---|---|
| `region_id` | **unique integer**. Make sure one region has `region_id = 1` (the tutorial region — see B-note). |
| `name` | lowercase column name; region display name (resolve the "Undesignated" 3 first) |
| `geom` | Polygon/MultiPolygon, **SRID 4326** |

### `qgis_road` (required columns)
| column | notes |
|---|---|
| `road_id` | **unique integer** (becomes `street_edge_id`) |
| `geom` | **LINESTRING**, SRID 4326, **split at every intersection** (and ideally at region boundaries) |
| `highway` | OSM road-class string (this is the `way_type` column; the script prompts for the name and defaults to `highway`) |
| `osm_id` | the OSM way id — **must cast to a single INT** (no lists, no commas) |
| `region_id` | the region each street segment falls in |

### Steps
1. **Reproject** the neighborhoods layer 2270 → **4326**.
2. **Resolve the 3 "Undesignated"** polygons (merge or rename), then assign a unique integer
   `region_id` to every region (use a field calculator; make one of them `1`).
3. **Filter OSM streets** to the road classes above; **clip** to the Bend boundary.
4. **Split streets at intersections** (QGIS: *Split with lines* / *v.clean*), and convert
   **multi-linestrings → single linestrings** (*Multipart to singleparts*). Optionally also
   split at region boundaries so each segment sits in one region.
5. **Spatially join** each street to its region to populate `road_id`'s `region_id` (assign by
   the segment's location; if a segment still spans two regions, pick by midpoint).
6. Assign a unique integer **`road_id`** to every street (field calculator, e.g. row number).
7. Ensure **`osm_id` is a single integer** per row (if your OSM import produced a list / text,
   reduce it to one id and cast to int).

> **B-note (tutorial region):** `fill-new-schema.sh` inserts a DC tutorial street and assigns
> it to "Tutorial region id" (default **1**). That region_id must exist in `qgis_region`, so
> keep a region with `region_id = 1`.

---

## Phase C — Create the schema and import the tables

All `make` targets run inside the `projectsidewalk-db` Docker container.

1. **Create the empty city schema** (clones the `sidewalk_init` template, makes the user, sets
   `search_path`):
   ```bash
   cd /d/git/ProjectSidewalk_Webpage
   make create-new-schema name=sidewalk_bend
   ```
2. **Import `qgis_region` and `qgis_road` into the `sidewalk_bend` schema** as PostGIS tables.
   Use QGIS *DB Manager* (export to PostGIS, schema `sidewalk_bend`) or `ogr2ogr`. They must
   land in the `sidewalk_bend` schema with the column names from Phase B and SRID 4326.

---

## Phase D — Fill the schema

```bash
make fill-new-schema
```
This is **interactive**. Answer the prompts:

| Prompt | Answer for Bend |
|---|---|
| Schema name | `sidewalk_bend` |
| OSM way_type column name | `highway` |
| Region data source (string) | e.g. `City of Bend Neighborhood Districts` |
| Region name column (lowercase) | `name` |
| Tutorial region id | `1` |
| Including all regions? | `y` (if you kept all neighborhoods public) |
| Proceed? | `y` |

The script then derives `street_edge`, `region`, `street_edge_region`,
`street_edge_priority`, `osm_way_street_edge`, sets `city_center_lat/lng` from the region
extent, and **drops** the `qgis_road` / `qgis_region` tables. (It also sets the SW/NE map
boundaries to center **±1°**, which is too wide — tighten in Phase F.)

---

## Phase E — Prune streets with no Street View imagery

```bash
# 1. From the sidewalk_bend schema, export a CSV named street_edge_endpoints.csv with columns:
#    street_edge_id, x1, y1, x2, y2, geom   (place it in the ProjectSidewalk_Webpage root)
# 2. Provide a Google Maps API key and run the checker (slow — it probes every street):
export GOOGLE_MAPS_API_KEY=...your key...
python check_streets_for_imagery.py        # writes db/streets_with_no_imagery.csv
# 3. Mark those streets deleted:
make hide-streets-without-imagery          # prompts: schema name = sidewalk_bend, CSV path
```

---

## Phase F — Configure the app for Bend

### F1. `conf/cityparams.conf` — add a `bend-or` entry to **every** map
Add `"bend-or"` to the `city-ids` list, then a `bend-or = …` line in each of:
`db-schema` (`"sidewalk_bend"`), `city-short-name`, `state-id` (`"oregon"`),
`country-id` (`"usa"`), `status` (`"public"` or `"private"`), `launch-date`, `skyline-img`,
`logo-img`, `landing-page-url` (prod/test/local URLs + Google Analytics `G-…` ids),
`ai-tag-suggestions-enabled`, `ai-validation-enabled`, `ai-validation-min-accuracy`.
(Model the values on an existing OR city like `newberg-or`.)

### F2. `docker-compose.yml` (ideally `docker-compose.override.yml`)
Set `SIDEWALK_CITY_ID=bend-or` and `DATABASE_USER=sidewalk_bend`.

### F3. City name translations
Add the Bend city name to the message/translation files (`conf/messages*`) for each UI
language you support.

### F4. Google Maps / Street View API key
Whitelist your local/test/prod Bend URLs on the API key in Google Cloud Console.

### F5. Refine the `config` table (in the `sidewalk_bend` schema)
- Tighten `southwest_boundary_*` / `northeast_boundary_*` (the script set them to center ±1°).
- Set `default_map_zoom` and `excluded_tags` (copy from a similar city).

---

## Phase G — Launch and verify locally
Bring up the dev environment, open the site with `SIDEWALK_CITY_ID=bend-or`, and confirm:
- the map centers on Bend,
- all neighborhoods render,
- you can audit a street and the tutorial works.

---

## Phase H — Generate + submit AI labels (this repo)
Once the Bend instance is live and reachable:
```bash
conda activate sidewalk-auto-labeler

# Scope first (pano count + runtime estimate; no model load, nothing processed):
python main.py example_geojson/bend.geojson --name bend --scan-only

# Run the labeler; all per-area state lands in runs/bend/ (results.jsonl, resume cache, etc.):
python main.py example_geojson/bend.geojson --name bend

# (Optional) spot-check and score the detections before submitting:
python scripts/spot_check_gallery.py runs/bend
python scripts/score_validation.py runs/bend
```
Then submit `runs/bend/results.jsonl` to your Bend server's `/ai/submitLabelsOnPano`
(preview with `--dry-run` first):
```bash
python send_to_ps.py runs/bend/results.jsonl --dry-run
python send_to_ps.py runs/bend/results.jsonl --endpoint https://<your-bend-server>/ai/submitLabelsOnPano
```

> Running on a lab GPU server (e.g. makelab2)? See
> [`makeability-quickstart.md`](makeability-quickstart.md) for the tmux / long-run workflow.

> Keep the `bend.geojson` search area and the PS region/street coverage describing the **same
> Bend**, so submitted labels attach to real street edges.

---

## Phase I — Deploy to a real server (later)
`pg_dump -Fc` the `sidewalk_bend` schema and provision server infra (DNS, prod DB) with IT,
per the [Creating database for a new city](https://github.com/ProjectSidewalk/SidewalkWebpage/wiki/Creating-database-for-a-new-city)
wiki. Not required for local testing.
