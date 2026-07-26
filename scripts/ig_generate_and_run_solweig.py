"""Investment-grade Stage 2: generate micro-update scenarios and run real SOLWEIG.

For each pilot tile:
  - build the IG baseline (paved ground -> Dark_asphalt, the hottest UMEP class)
  - generate scenarios: baseline + single-type saturations + mixed presets + random mixes
  - run SOLWEIG over the city's hottest 3-day window, hourly 08-18 + 22:00 night (36 timesteps)
  - save per-scenario modified input rasters (cdsm/landcover) + cropped summary grids

SOLWEIG (solweig 0.1.0b84) runs ~2-3 s/scenario, so the whole pilot is minutes.
Run with the project venv:  .venv/bin/python scripts/ig_generate_and_run_solweig.py
"""
from __future__ import annotations

from pathlib import Path
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RASTERIO_PROJ = Path(sys.executable).resolve().parents[1] / "lib/python3.12/site-packages/rasterio/proj_data"
if RASTERIO_PROJ.exists():
    os.environ["PROJ_DATA"] = str(RASTERIO_PROJ)
    os.environ["PROJ_LIB"] = str(RASTERIO_PROJ)
import rasterio
from rasterio.transform import from_origin

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.microupdate.action_space import (
    ACTIONS, LC_BUILDING, build_ig_baseline_landcover, candidate_mask, apply_action,
)

PILOT_CSV = ROOT / "data/candidate_tiles/pilot4_tiles.csv"
IG_ROOT = ROOT / "data/ig"
TARGET = 256
DAY_HOURS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
NIGHT_HOUR = 22
HEAT_THRESHOLDS_DAY = [38.0]   # UTCI very-strong heat-stress onset


def parse_epw_location(path: Path) -> dict:
    first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].split(",")
    if len(first) < 10 or first[0] != "LOCATION":
        return {}
    return {"latitude": float(first[6]), "longitude": float(first[7]),
            "utc_offset": float(first[8]), "altitude": float(first[9])}


def hottest_3day_window(epw_path: Path) -> tuple[str, str, int]:
    """Return (start_mmdd, end_mmdd, start_doy) of the hottest 3 consecutive days."""
    lines = epw_path.read_text(errors="ignore").splitlines()[8:]
    T = np.array([float(l.split(",")[6]) for l in lines if l.strip()])
    k = 72
    csum = np.cumsum(np.insert(T, 0, 0))
    roll = (csum[k:] - csum[:-k]) / k
    i = int(np.argmax(roll))
    start_doy = i // 24 + 1
    d0 = dt.date(2017, 1, 1) + dt.timedelta(days=start_doy - 1)
    d2 = d0 + dt.timedelta(days=2)
    return d0.strftime("%m-%d"), d2.strftime("%m-%d"), start_doy


def read_tif(path: Path):
    with rasterio.open(path) as s:
        return s.read(1), s.profile, s.transform, s.crs


def center_pad(arr: np.ndarray, fill) -> tuple[np.ndarray, int, int]:
    r, c = arr.shape
    pt = max((TARGET - r) // 2, 0)
    pl = max((TARGET - c) // 2, 0)
    pb = max(TARGET - r - pt, 0)
    pr = max(TARGET - c - pl, 0)
    padded = np.pad(arr, ((pt, pb), (pl, pr)), mode="constant", constant_values=fill)
    return padded, pt, pl


def write_tif(path: Path, arr: np.ndarray, transform, crs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=str(arr.dtype), crs=crs, transform=transform,
                       compress="lzw") as dst:
        dst.write(arr, 1)


def generate_scenarios(cand_rc: np.ndarray, rng: np.random.Generator, n_random: int) -> list[dict]:
    """Each scenario: {id, applications:[(r,c,aid),...]} where each app is a disk centre.
    Centre counts scale with ground size M so coverage spans ~5-55% across scenarios
    (the diagnostic-validated regime where cooling is a real signal)."""
    M = len(cand_rc)
    name_to_aid = {a.name: a.aid for a in ACTIONS}

    def pick(n):
        idx = rng.choice(M, size=min(int(n), M), replace=False)
        return [(int(cand_rc[i, 0]), int(cand_rc[i, 1])) for i in idx]

    scen: list[dict] = [{"id": "baseline", "applications": []}]

    # single-type saturations at 3 levels — RL-RELEVANT counts (tens, not thousands): RL
    # plans place ~50 disks under budget, so the surrogate only needs accuracy in that range;
    # ultra-dense scenarios (M//16 ≈ 1000+ trees) are useless for RL and choke CPU-SVF.
    levels = {"lo": 15, "md": 50, "hi": 120}
    for a in ACTIONS:
        for lvl, n in levels.items():
            scen.append({"id": f"single_{a.name}_{lvl}",
                         "applications": [(r, c, a.aid) for r, c in pick(n)]})

    # mixed presets (named, interpretable) — RL-relevant counts
    u = 25
    presets = [
        ("mix_trees", [("tree_small", 1.5 * u), ("tree_medium", 1.5 * u), ("tree_large", u)]),
        ("mix_tree_cool", [("tree_medium", 2 * u), ("cool_pavement", 2.5 * u)]),
        ("mix_green_water", [("greening", 3 * u), ("water", u), ("tree_small", u)]),
        ("mix_all", [("tree_large", u), ("tree_medium", 1.2 * u), ("cool_pavement", 1.5 * u),
                     ("greening", 1.2 * u), ("water", 0.5 * u)]),
    ]
    for pid, spec in presets:
        pts = pick(sum(int(c) for _, c in spec))
        apps, p = [], 0
        for nm, c in spec:
            for (r, cc) in pts[p:p + int(c)]:
                apps.append((r, cc, name_to_aid[nm]))
            p += int(c)
        scen.append({"id": pid, "applications": apps})

    # random mixes (varied count + composition) -> dense surrogate coverage
    for j in range(n_random):
        n = int(rng.integers(12, 150))
        pts = pick(n)
        aids = rng.integers(0, len(ACTIONS), size=len(pts))
        apps = [(r, c, int(aid)) for (r, c), aid in zip(pts, aids)]
        scen.append({"id": f"rand_{j:03d}", "applications": apps})
    return scen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=str(PILOT_CSV.relative_to(ROOT)), help="tiles CSV")
    ap.add_argument("--site-id", default=None, help="run a single site (default: all in --tiles)")
    ap.add_argument("--baseline-only", action="store_true",
                    help="full-scale: generate ONLY the baseline scenario (no singles/mixed/random) + run SOLWEIG")
    ap.add_argument("--n-random", type=int, default=35)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--tile-size", type=int, default=64)
    ap.add_argument("--max-shadow-distance-m", type=float, default=250.0)
    args = ap.parse_args()

    import solweig
    if os.environ.get("MICROUPDATE_SOLWEIG_CPU"):   # GPU (CUDA/Metal) is ~1000x faster for SVF;
        solweig.disable_gpu()                       # force CPU only if explicitly requested

    pilots = pd.read_csv(ROOT / args.tiles)
    if args.site_id:
        pilots = pilots[pilots.site_id == args.site_id]

    all_rows = []
    for _, row in pilots.iterrows():
        sid = row["site_id"]
        city = row["city"]
        pin = ROOT / "data/sites" / sid / "solweig_inputs_pilot"
        epw = pin / "weather.epw"
        dem, _, _, crs = read_tif(pin / "dem.tif")
        dsm, _, transform, _ = read_tif(pin / "dsm.tif")
        cdsm0, _, _, _ = read_tif(pin / "cdsm_tree_canopy.tif")
        lc_pilot, _, _, _ = read_tif(pin / "landcover.tif")
        lc_base = build_ig_baseline_landcover(lc_pilot)

        start, end, sdoy = hottest_3day_window(epw)
        loc = parse_epw_location(epw)
        location = solweig.Location(latitude=float(row["center_lat"]), longitude=float(row["center_lon"]),
                                    altitude=loc.get("altitude", 10.0), utc_offset=loc.get("utc_offset", 0.0))
        weather = solweig.Weather.from_epw(epw, start=start, end=end, hours=DAY_HOURS + [NIGHT_HOUR])
        n_steps = len(weather) if hasattr(weather, "__len__") else 1
        print(f"\n=== {sid} ({city}) hottest 3-day {start}..{end} -> {n_steps} timesteps ===", flush=True)

        cand = candidate_mask(lc_base, cdsm0)
        rr, cc = np.where(cand)
        cand_rc = np.stack([rr, cc], axis=1)
        rng = np.random.default_rng(args.seed + sdoy)
        scenarios = ([{"id": "baseline", "applications": []}] if args.baseline_only
                     else generate_scenarios(cand_rc, rng, args.n_random))

        site_root = IG_ROOT / sid
        (site_root).mkdir(parents=True, exist_ok=True)
        # tile-constant rasters saved once (original, unpadded extent)
        write_tif(site_root / "dsm.tif", dsm.astype(np.float32), transform, crs)
        write_tif(site_root / "dem.tif", dem.astype(np.float32), transform, crs)
        write_tif(site_root / "landcover_baseline.tif", lc_base, transform, crs)

        for spec in scenarios:
            cdsm = cdsm0.copy().astype(np.float32)
            lc = lc_base.copy().astype(np.uint8)
            modified = np.zeros_like(lc, dtype=np.uint8)
            cost = 0.0
            per_type = {a.name: 0 for a in ACTIONS}
            for (r, c, aid) in spec["applications"]:
                a = ACTIONS[aid]
                apply_action(cdsm, lc, modified, r, c, a)
                cost += a.cost
                per_type[a.name] += 1

            sdir = site_root / "scenarios" / spec["id"]
            sdir.mkdir(parents=True, exist_ok=True)
            write_tif(sdir / "cdsm_tree_canopy.tif", cdsm, transform, crs)
            write_tif(sdir / "landcover.tif", lc, transform, crs)
            write_tif(sdir / "modified_mask.tif", modified, transform, crs)

            # pad to 256 for the solweig beta; remember crop window
            dsm_p, pt, pl = center_pad(dsm.astype(np.float32), 0.0)
            dem_p, _, _ = center_pad(dem.astype(np.float32), 0.0)
            cdsm_p, _, _ = center_pad(cdsm, 0.0)
            lc_p, _, _ = center_pad(lc, LC_BUILDING * 0)  # pad with 0 (cobble), border is inert ground
            H, W = dsm.shape
            ptr = from_origin(transform.c - pl * transform.a, transform.f + pt * transform.a,
                              transform.a, transform.a)

            cache = site_root / "_cache" / spec["id"]
            out = site_root / "_out" / spec["id"]
            for d in (cache, out):
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
            # write padded inputs to a temp dir for solweig
            tmp = site_root / "_tmp" / spec["id"]
            tmp.mkdir(parents=True, exist_ok=True)
            write_tif(tmp / "dsm.tif", dsm_p, ptr, crs)
            write_tif(tmp / "dem.tif", dem_p, ptr, crs)
            write_tif(tmp / "cdsm.tif", cdsm_p, ptr, crs)
            write_tif(tmp / "landcover.tif", lc_p.astype(np.uint8), ptr, crs)

            t0 = time.time()
            surface = solweig.SurfaceData.prepare(
                dsm=tmp / "dsm.tif", dem=tmp / "dem.tif", cdsm=tmp / "cdsm.tif",
                land_cover=tmp / "landcover.tif", working_dir=cache,
                tile_size=args.tile_size, force_recompute=True,
            )
            solweig.calculate(
                surface=surface, weather=weather, location=location, output_dir=out,
                outputs=["tmrt"], tile_size=args.tile_size,
                max_shadow_distance_m=args.max_shadow_distance_m,
                heat_thresholds_day=HEAT_THRESHOLDS_DAY,
            )
            elapsed = time.time() - t0

            # crop summary grids back to original extent, save to scenario dir
            summ_src = out / "summary"
            summ_dst = sdir / "summary"
            summ_dst.mkdir(exist_ok=True)
            for tif in sorted(summ_src.glob("*.tif")):
                arr, _, _, _ = read_tif(tif)
                cropped = arr[pt:pt + H, pl:pl + W]
                write_tif(summ_dst / tif.name, cropped.astype(np.float32), transform, crs)

            (sdir / "metadata.json").write_text(json.dumps({
                "scenario_id": spec["id"], "site_id": sid, "city": city,
                "n_actions": len(spec["applications"]), "cost": cost, "per_type": per_type,
                "window": [start, end], "n_timesteps": n_steps,
            }, indent=2), encoding="utf-8")
            shutil.rmtree(tmp); shutil.rmtree(cache); shutil.rmtree(out)

            all_rows.append({"site_id": sid, "city": city, "scenario_id": spec["id"],
                             "n_actions": len(spec["applications"]), "cost": cost,
                             "elapsed_s": round(elapsed, 2), **{f"n_{k}": v for k, v in per_type.items()}})
            print(f"  {spec['id']:24} actions={len(spec['applications']):3} cost={cost:6.0f} {elapsed:4.1f}s", flush=True)

    man = pd.DataFrame(all_rows)
    IG_ROOT.mkdir(parents=True, exist_ok=True)
    if args.site_id:                         # parallel-safe: one manifest per site (concat later)
        (IG_ROOT / "_manifests").mkdir(exist_ok=True)
        man.to_csv(IG_ROOT / "_manifests" / f"{args.site_id}.csv", index=False)
    else:
        mpath = IG_ROOT / "scenario_manifest.csv"
        if mpath.exists():                   # merge: keep other tiles, replace re-run ones
            old = pd.read_csv(mpath)
            old = old[~old.site_id.isin(man.site_id.unique())]
            man = pd.concat([old, man], ignore_index=True)
        man.to_csv(mpath, index=False)
    print(f"\nWrote {IG_ROOT/'scenario_manifest.csv'}  ({len(man)} scenarios, "
          f"total {man.elapsed_s.sum():.0f}s)")


if __name__ == "__main__":
    main()
