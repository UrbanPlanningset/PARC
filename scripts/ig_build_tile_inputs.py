"""Build SOLWEIG input rasters for any candidate tile (geopandas-free: json + shapely +
pyproj + rasterio). Same offline pilot recipe as build_pilot_inputs.py (flat DEM, treeless
baseline, landcover = ground/building) so ig_generate_and_run_solweig.py works unchanged.

.venv/bin/python scripts/ig_build_tile_inputs.py --tiles data/candidate_tiles/scale8_tiles.csv
"""
from __future__ import annotations

from pathlib import Path
import argparse, json, math, os, shutil, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RASTERIO_PROJ = Path(sys.executable).resolve().parents[1] / "lib/python3.12/site-packages/rasterio/proj_data"
if RASTERIO_PROJ.exists():
    os.environ["PROJ_DATA"] = str(RASTERIO_PROJ); os.environ["PROJ_LIB"] = str(RASTERIO_PROJ)
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

sys.path.insert(0, str(ROOT))
from scripts.estimate_site_building_heights import estimate_height

CELL = 5.0
DEFAULT_HEIGHT, LEVEL_HEIGHT = 15.0, 3.2
OSM_DIR = ROOT / "data/candidate_tiles/unified4_osm_buildings"
# point at the per-city EPW that ships inside the representative pilot tiles (on the server),
# so full-scale tile building needs no extra weather upload. (Each city shares one EPW.)
WEATHER_BY_CITY = {
    "New York": "data/sites/nyc_core_nyc_core_midtown_gx0001_gy0001/solweig_inputs_pilot/weather.epw",
    "Seoul": "data/sites/seoul_core_seoul_core_gangnam_gx0005_gy0006/solweig_inputs_pilot/weather.epw",
    "Shanghai": "data/sites/shanghai_core_shanghai_core_puxi_gx0001_gy0005/solweig_inputs_pilot/weather.epw",
    "Beijing": "data/sites/beijing_core_beijing_core_second_ring_gx0003_gy0011/solweig_inputs_pilot/weather.epw",
}


def write_tif(path, arr, transform, crs, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata,
                       compress="lzw") as d:
        d.write(arr, 1)


def build_tile(row) -> dict:
    sid, city, crs = row["site_id"], row["city"], row["crs"]
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    (minx, miny) = tf.transform(row["min_lon"], row["min_lat"])
    (maxx, maxy) = tf.transform(row["max_lon"], row["max_lat"])
    minx, maxx = min(minx, maxx), max(minx, maxx)
    miny, maxy = min(miny, maxy), max(miny, maxy)
    width = int(math.ceil((maxx - minx) / CELL)); height = int(math.ceil((maxy - miny) / CELL))
    transform = from_origin(minx, maxy, CELL, CELL)

    gj = json.loads((OSM_DIR / f"{sid}.geojson").read_text())
    shapes = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        g = shp_transform(lambda x, y, z=None: tf.transform(x, y), shape(geom))
        h, _ = estimate_height(feat.get("properties", {}), DEFAULT_HEIGHT, LEVEL_HEIGHT)
        shapes.append((g, float(h)))

    bh = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0,
                   all_touched=True, dtype="float32") if shapes else np.zeros((height, width), np.float32)
    dem = np.zeros((height, width), np.float32)
    dsm = (dem + bh).astype(np.float32)
    cdsm = np.zeros((height, width), np.float32)
    landcover = np.zeros((height, width), np.uint8)   # 0=ground
    landcover[bh > 0] = 2                              # 2=building

    out = ROOT / "data/sites" / sid / "solweig_inputs_pilot"
    write_tif(out / "dem.tif", dem, transform, crs)
    write_tif(out / "dsm.tif", dsm, transform, crs)
    write_tif(out / "cdsm_tree_canopy.tif", cdsm, transform, crs)
    write_tif(out / "landcover.tif", landcover, transform, crs, nodata=255)
    epw_src = ROOT / WEATHER_BY_CITY[city]
    shutil.copyfile(epw_src, out / "weather.epw")
    return {"site_id": sid, "city": city, "shape": f"{height}x{width}",
            "n_buildings": len(shapes), "max_h": float(bh.max()), "bld_cells": int((bh > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/candidate_tiles/scale8_tiles.csv")
    ap.add_argument("--site-id", default=None)
    args = ap.parse_args()
    tiles = pd.read_csv(ROOT / args.tiles)
    if args.site_id:
        tiles = tiles[tiles.site_id == args.site_id]
    rows, failed = [], 0
    for i, (_, r) in enumerate(tiles.iterrows()):
        try:
            info = build_tile(r)
            rows.append(info)
        except Exception as e:                 # skip a bad tile, don't kill the whole 1359-tile build
            failed += 1
            print(f"  !! skip {r['site_id'][:42]}: {e}", flush=True)
            continue
        if (i + 1) % 50 == 0:
            print(f"  built {len(rows)} tiles ({i+1}/{len(tiles)}) ...", flush=True)
    print(f"\nBuilt {len(rows)} tiles -> data/sites/<id>/solweig_inputs_pilot/  ({failed} skipped)")


if __name__ == "__main__":
    main()
