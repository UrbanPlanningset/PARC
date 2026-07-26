from __future__ import annotations

from pathlib import Path
import argparse
import json
import math


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value).lower().replace("m", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def estimate_height(props: dict, default_height: float, level_height: float) -> tuple[float, str]:
    for key in ["height", "building:height", "height_m", "HEIGHT", "building_height"]:
        value = parse_float(props.get(key))
        if value and value > 0:
            return value, f"attribute:{key}"
    for key in ["building:levels", "levels", "LEVELS", "floors"]:
        levels = parse_float(props.get(key))
        if levels and levels > 0:
            return levels * level_height, f"levels:{key}"
    return default_height, "default_prior"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--default-height", type=float, default=15.0)
    parser.add_argument("--level-height", type=float, default=3.2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    in_path = Path(args.input) if args.input else root / "data/sites" / args.site_id / "raw/osm_buildings.geojson"
    out_path = Path(args.out) if args.out else root / "data/sites" / args.site_id / "processed/buildings_with_height.geojson"
    data = json.loads(in_path.read_text(encoding="utf-8"))
    features = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        height, method = estimate_height(props, args.default_height, args.level_height)
        feature["properties"] = {
            **props,
            "height_m": height,
            "height_estimation_method": method,
        }
        features.append(feature)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")
    method_counts: dict[str, int] = {}
    for feature in features:
        method = feature["properties"]["height_estimation_method"]
        method_counts[method] = method_counts.get(method, 0) + 1
    print(f"Wrote {len(features)} buildings to {out_path}")
    print(method_counts)


if __name__ == "__main__":
    main()
