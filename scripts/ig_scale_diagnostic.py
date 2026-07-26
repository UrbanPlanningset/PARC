"""Diagnostic: at what canopy footprint + coverage does a micro-update produce a
measurable cooling signal? Measures whole-tile vs hotspot-focused cooling so we
can pick the right intervention scale and primary metric before generating data.

.venv/bin/python scripts/ig_scale_diagnostic.py
"""
from __future__ import annotations

from pathlib import Path
import os, sys, shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ig_generate_and_run_solweig import (
    hottest_3day_window, read_tif, center_pad, write_tif, parse_epw_location,
    DAY_HOURS, NIGHT_HOUR, HEAT_THRESHOLDS_DAY, TARGET,
)
from src.microupdate.action_space import build_ig_baseline_landcover, LC_BUILDING, LC_COBBLE, LC_GRASS
from rasterio.transform import from_origin

SID = "nyc_core_nyc_core_midtown_gx0001_gy0001"


def disk_centers(cand_rc, n, rng):
    idx = rng.choice(len(cand_rc), size=min(n, len(cand_rc)), replace=False)
    return cand_rc[idx]


def stamp_disk(arr, r, c, radius, value, mode="max"):
    r0, r1 = max(0, r - radius), min(arr.shape[0], r + radius + 1)
    c0, c1 = max(0, c - radius), min(arr.shape[1], c + radius + 1)
    if mode == "max":
        arr[r0:r1, c0:c1] = np.maximum(arr[r0:r1, c0:c1], value)
    else:
        arr[r0:r1, c0:c1] = value


def main():
    import solweig
    solweig.disable_gpu()
    import logging
    logging.disable(logging.WARNING)

    pin = ROOT / "data/sites" / SID / "solweig_inputs_pilot"
    epw = pin / "weather.epw"
    dem, _, _, crs = read_tif(pin / "dem.tif")
    dsm, _, transform, _ = read_tif(pin / "dsm.tif")
    cdsm0, _, _, _ = read_tif(pin / "cdsm_tree_canopy.tif")
    lc_pilot, _, _, _ = read_tif(pin / "landcover.tif")
    lc_base = build_ig_baseline_landcover(lc_pilot)
    H, W = dsm.shape

    start, end, _ = hottest_3day_window(epw)
    loc = parse_epw_location(epw)
    location = solweig.Location(latitude=40.747, longitude=-73.99,
                                altitude=loc.get("altitude", 10.0), utc_offset=loc.get("utc_offset", -5))
    weather = solweig.Weather.from_epw(epw, start=start, end=end, hours=DAY_HOURS + [NIGHT_HOUR])
    ground = lc_base != LC_BUILDING
    rng = np.random.default_rng(7)
    cand_rc = np.stack(np.where(ground), axis=1)

    def run(cdsm, lc, tag):
        dsm_p, pt, pl = center_pad(dsm.astype(np.float32), 0.0)
        dem_p, _, _ = center_pad(dem.astype(np.float32), 0.0)
        cdsm_p, _, _ = center_pad(cdsm.astype(np.float32), 0.0)
        lc_p, _, _ = center_pad(lc.astype(np.uint8), 0)
        ptr = from_origin(transform.c - pl * transform.a, transform.f + pt * transform.a, transform.a, transform.a)
        tmp = ROOT / "data/ig/_diag" / tag
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        for nm, a in [("dsm", dsm_p), ("dem", dem_p), ("cdsm", cdsm_p)]:
            write_tif(tmp / f"{nm}.tif", a, ptr, crs)
        write_tif(tmp / "landcover.tif", lc_p, ptr, crs)
        surface = solweig.SurfaceData.prepare(dsm=tmp/"dsm.tif", dem=tmp/"dem.tif", cdsm=tmp/"cdsm.tif",
                                              land_cover=tmp/"landcover.tif", working_dir=tmp/"cache",
                                              tile_size=64, force_recompute=True)
        solweig.calculate(surface=surface, weather=weather, location=location, output_dir=tmp/"out",
                          outputs=[], tile_size=64, max_shadow_distance_m=250.0,
                          heat_thresholds_day=HEAT_THRESHOLDS_DAY)
        def g(name):
            arr, _, _, _ = read_tif(tmp/"out"/"summary"/name)
            return arr[pt:pt+H, pl:pl+W]
        out = {n: g(n) for n in ["utci_day_mean.tif", "tmrt_day_mean.tif", "utci_hours_above_38_day.tif"]}
        shutil.rmtree(tmp)
        return out

    base = run(cdsm0, lc_base, "base")
    bu = base["utci_day_mean.tif"]; bt = base["tmrt_day_mean.tif"]; b38 = base["utci_hours_above_38_day.tif"]
    hot = ground & (bu >= np.quantile(bu[ground], 0.75))   # hottest-quartile ground = pedestrian hotspots
    base_n38 = int(((b38 > 0) & ground).sum())
    print(f"baseline: ground UTCI_day mean={bu[ground].mean():.2f}  hotspot mean={bu[hot].mean():.2f}  "
          f">38 day-exceed cells={base_n38}")
    print(f"{'scenario':30} {'cover%':>6} {'ΔUTCI_all':>9} {'ΔUTCI_hot':>9} {'ΔTmrt_hot':>9} {'Δhot38cells':>11}")

    def report(scn, lc_or_cdsm, tag, cover):
        u = scn["utci_day_mean.tif"]; t = scn["tmrt_day_mean.tif"]; e = scn["utci_hours_above_38_day.tif"]
        d_all = (bu[ground] - u[ground]).mean()
        d_hot = (bu[hot] - u[hot]).mean()
        dt_hot = (bt[hot] - t[hot]).mean()
        d38 = base_n38 - int(((e > 0) & ground).sum())
        print(f"{tag:30} {cover*100:5.1f}% {d_all:+9.3f} {d_hot:+9.3f} {dt_hot:+9.3f} {d38:>+11}")

    # trees with real canopy footprint (radius 1 = ~15m crown) at increasing coverage
    for n in (150, 400, 800):
        cdsm = cdsm0.copy().astype(np.float32)
        for r, c in disk_centers(cand_rc, n, rng):
            stamp_disk(cdsm, r, c, radius=1, value=12.0, mode="max")
        cover = (cdsm > 0).sum() / ground.sum()
        report(run(cdsm, lc_base, f"tree{n}"), cdsm, f"tree_large_r1_n{n}", cover)

    # trees single-cell (old footprint) for contrast, big N
    cdsm = cdsm0.copy().astype(np.float32)
    for r, c in disk_centers(cand_rc, 800, rng):
        cdsm[r, c] = 12.0
    report(run(cdsm, lc_base, "tree_sc"), cdsm, "tree_large_singlecell_n800", (cdsm > 0).sum()/ground.sum())

    # materials at scale (cobble + grass) radius 1
    for lc_id, nm in ((LC_COBBLE, "cool_pavement"), (LC_GRASS, "greening")):
        lc = lc_base.copy().astype(np.uint8)
        cnt = 0
        for r, c in disk_centers(cand_rc, 800, rng):
            r0, r1 = max(0, r-1), min(H, r+2); c0, c1 = max(0, c-1), min(W, c+2)
            sub = lc[r0:r1, c0:c1]; sub[sub != LC_BUILDING] = lc_id; cnt += sub.size
        report(run(cdsm0, lc, f"mat_{nm}"), lc, f"{nm}_r1_n800", cnt/ground.sum())


if __name__ == "__main__":
    main()
