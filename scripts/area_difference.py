"""
Builds the run area for a city whose Project Sidewalk boundary grew: the new boundary
minus the area an earlier run already covered.

Why not just re-run the whole new boundary? A run dir is bound to its geometry, so it
would need a new run dir anyway, and that run would re-thin the old area from a fresh
coverage scan. Mapillary/Panoramax thinning keeps the newest pano per 5 m cell, so any
capture added since the first run would replace a pano whose labels are already live,
and PS is insert-only: those labels would be duplicated with no way to retire them.
Running only the difference avoids that. The thinning grid is global (lat/lon cells,
not anchored to the area), so the new run's cells line up with the old run's.

`new` is the city's current boundary: a GeoJSON geometry, Feature or FeatureCollection
(e.g. a pull of `https://<server>/v3/api/regions`); every polygon in it is unioned.
`old` is the earlier run's `runs/<city>/area.geojson`. Difference slivers left by the
two boundaries' shared edges (floating-point residue, ~0 m^2) are dropped below
`--min-area-m2`.

Usage (Juneau, 2026-11-07; reproduces the committed files byte for byte):

    curl -s https://sidewalk-juneau.cs.washington.edu/v3/api/regions \
        -o example_geojson/juneau_2026-11.ps_regions.geojson
    python scripts/area_difference.py example_geojson/juneau_2026-11.ps_regions.geojson \
        runs/juneau/area.geojson \
        --union-out example_geojson/juneau_2026-11.geojson \
        --out example_geojson/juneau_expansion_2026-11.geojson
    python main.py example_geojson/juneau_expansion_2026-11.geojson \
        --name juneau_expansion --source mapillary
"""
import argparse
import json
import math
import sys

from shapely.geometry import MultiPolygon, mapping, shape
from shapely.ops import unary_union

# Larger than any sliver, far smaller than any real block. Juneau: 19 slivers, all < 1 m^2.
DEFAULT_MIN_AREA_M2 = 1000.0


def polygons_of(doc):
    """Returns every Polygon/MultiPolygon in a GeoJSON geometry, Feature or FeatureCollection."""
    if doc.get("type") == "FeatureCollection":
        return [shape(f["geometry"]) for f in doc["features"]]
    if doc.get("type") == "Feature":
        return [shape(doc["geometry"])]
    return [shape(doc)]


def approx_area_m2(geom):
    """Planar area in m^2 at the geometry's latitude; good enough to tell a sliver from a block."""
    lat = geom.representative_point().y
    return geom.area * 111320.0 ** 2 * math.cos(math.radians(lat))


def area_difference(new_doc, old_doc, min_area_m2=DEFAULT_MIN_AREA_M2):
    """Returns (union of new, new minus old with slivers dropped, number of slivers dropped)."""
    new = unary_union(polygons_of(new_doc))
    old = unary_union(polygons_of(old_doc))
    diff = new.difference(old).buffer(0)
    parts = list(diff.geoms) if diff.geom_type == "MultiPolygon" else [diff]
    keep = [p for p in parts if not p.is_empty and approx_area_m2(p) >= min_area_m2]
    if not keep:
        return new, None, len(parts)
    out = keep[0] if len(keep) == 1 else MultiPolygon(keep)
    return new, out, len(parts) - len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("new", help="current boundary (geometry, Feature or FeatureCollection)")
    ap.add_argument("old", help="the earlier run's runs/<city>/area.geojson")
    ap.add_argument("--out", required=True, help="where to write new-minus-old")
    ap.add_argument("--union-out", help="also write the unioned new boundary here")
    ap.add_argument("--min-area-m2", type=float, default=DEFAULT_MIN_AREA_M2)
    args = ap.parse_args()

    with open(args.new) as f:
        new_doc = json.load(f)
    with open(args.old) as f:
        old_doc = json.load(f)
    new, diff, dropped = area_difference(new_doc, old_doc, args.min_area_m2)
    old = unary_union(polygons_of(old_doc))
    uncovered = old.difference(new)
    print(f"new boundary: {approx_area_m2(new) / 1e6:.3f} km2; old run area: "
          f"{approx_area_m2(old) / 1e6:.3f} km2")
    if not uncovered.is_empty and approx_area_m2(uncovered) >= args.min_area_m2:
        # Not fatal (a boundary can also shrink), but the old run's labels there stay live.
        print(f"NOTE: {approx_area_m2(uncovered) / 1e6:.3f} km2 of the old run area is "
              f"outside the new boundary")
    if args.union_out:
        with open(args.union_out, "w") as f:
            json.dump(mapping(new), f)
    if diff is None:
        print("nothing new: the new boundary adds no area beyond the old run", file=sys.stderr)
        return 1
    with open(args.out, "w") as f:
        json.dump(mapping(diff), f)
    print(f"new-only area: {diff.geom_type}, {approx_area_m2(diff) / 1e6:.3f} km2, "
          f"bounds {tuple(round(b, 6) for b in diff.bounds)}; {dropped} sliver(s) dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
