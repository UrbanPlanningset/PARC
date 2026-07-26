"""Stage 5: backtest each method's best plan through real SOLWEIG (the two-stage paradigm).

For every (tile, method) plan saved by ig_run_rl_and_baselines.py, run SOLWEIG on the
plan layout over the same hottest-3-day window, then compute TRUE cooling metrics vs the
tile baseline. Reports: (a) true-value method comparison (gate: RL >= greedy on truth);
(b) surrogate-vs-truth fidelity (backtest shrinkage) per the plan's validation口径.

.venv/bin/python scripts/ig_backtest_solweig.py
"""
from __future__ import annotations

from pathlib import Path
import os, sys, json, shutil, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ig_generate_and_run_solweig import (
    hottest_3day_window, read_tif, center_pad, write_tif, parse_epw_location,
    DAY_HOURS, NIGHT_HOUR, HEAT_THRESHOLDS_DAY,
)
from src.microupdate.action_space import LC_BUILDING
from rasterio.transform import from_origin

IG = ROOT / "data/ig"
RL = ROOT / "results/ig/rl"
OUT = ROOT / "results/ig/backtest"
EXTREME = 38.0


def run_layout(solweig, site_id, cdsm, lc, dsm, dem, transform, crs, weather, location, tag):
    dsm_p, pt, pl = center_pad(dsm.astype(np.float32), 0.0)
    dem_p, _, _ = center_pad(dem.astype(np.float32), 0.0)
    cdsm_p, _, _ = center_pad(cdsm.astype(np.float32), 0.0)
    lc_p, _, _ = center_pad(lc.astype(np.uint8), 0)
    H, W = dsm.shape
    ptr = from_origin(transform.c - pl * transform.a, transform.f + pt * transform.a, transform.a, transform.a)
    tmp = IG / "_backtest" / f"{site_id}_{tag}"
    if tmp.exists():
        shutil.rmtree(tmp)
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
    res = {n: g(n + ".tif") for n in ["utci_day_mean", "tmrt_day_mean", "utci_hours_above_38_day",
                                      "utci_night_mean"]}
    shutil.rmtree(tmp)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/candidate_tiles/pilot4_tiles.csv")
    args = ap.parse_args()
    import solweig
    if os.environ.get("MICROUPDATE_SOLWEIG_CPU"):   # GPU is ~1000x faster for SVF
        solweig.disable_gpu()
    pilots = pd.read_csv(ROOT / args.tiles)
    surro = pd.read_csv(RL / "surrogate_comparison.csv")
    rows = []
    for _, prow in pilots.iterrows():
        sid = prow["site_id"]; city = prow["city"]
        if not (RL / sid).exists():
            continue
        dem, _, _, crs = read_tif(IG / sid / "dem.tif")
        dsm, _, transform, _ = read_tif(IG / sid / "dsm.tif")
        lc_base = read_tif(IG / sid / "landcover_baseline.tif")[0]
        # read baseline summary already computed in Stage 2
        bu = read_tif(IG / sid / "scenarios/baseline/summary/utci_day_mean.tif")[0]
        bt = read_tif(IG / sid / "scenarios/baseline/summary/tmrt_day_mean.tif")[0]
        ground = lc_base != LC_BUILDING
        hot = ground & (bu >= np.quantile(bu[ground], 0.75))
        base_hours38 = read_tif(IG / sid / "scenarios/baseline/summary/utci_hours_above_38_day.tif")[0]
        base_ph38 = float(base_hours38[ground].sum())

        epw = ROOT / "data/sites" / sid / "solweig_inputs_pilot/weather.epw"
        start, end, _ = hottest_3day_window(epw)
        loc = parse_epw_location(epw)
        location = solweig.Location(latitude=float(prow["center_lat"]), longitude=float(prow["center_lon"]),
                                    altitude=loc.get("altitude", 10.0), utc_offset=loc.get("utc_offset", 0.0))
        weather = solweig.Weather.from_epw(epw, start=start, end=end, hours=DAY_HOURS + [NIGHT_HOUR])

        methods = sorted([p.name for p in (RL / sid).iterdir() if p.is_dir()])
        print(f"\n=== {city} ({sid[:30]}) backtest {len(methods)} methods ===", flush=True)
        for method in methods:
            mdir = RL / sid / method
            cdsm = read_tif(mdir / "cdsm_tree_canopy.tif")[0]
            lc = read_tif(mdir / "landcover.tif")[0]
            t0 = time.time()
            r = run_layout(solweig, sid, cdsm, lc, dsm, dem, transform, crs, weather, location, method)
            d_utci_hot = float((bu[hot] - r["utci_day_mean"][hot]).mean())
            d_utci_grd = float((bu[ground] - r["utci_day_mean"][ground]).mean())
            d_tmrt_hot = float((bt[hot] - r["tmrt_day_mean"][hot]).mean())
            ph38 = float(r["utci_hours_above_38_day"][ground].sum())
            d_ph38 = base_ph38 - ph38
            # surrogate-predicted hotspot cooling for the same plan
            srow = surro[(surro.site_id == sid) & (surro.method == method)]
            pred_hot = float(srow["mean_cool_hot"].iloc[0]) if len(srow) else np.nan
            rows.append({"city": city, "site_id": sid, "method": method,
                         "true_d_utci_hot": round(d_utci_hot, 4), "pred_d_utci_hot": round(pred_hot, 4),
                         "true_d_utci_ground": round(d_utci_grd, 4), "true_d_tmrt_hot": round(d_tmrt_hot, 4),
                         "true_d_ph38": round(d_ph38, 1)})
            print(f"  {method:14} true_cool_hot={d_utci_hot:.3f}  pred={pred_hot:.3f}  "
                  f"Δperson-hrs>38={d_ph38:.0f}  [{time.time()-t0:.1f}s]", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "truth_comparison.csv", index=False)
    print("\n=== TRUE-VALUE (SOLWEIG) hotspot ΔUTCI by method ===")
    piv = df.pivot_table(index="method", columns="city", values="true_d_utci_hot")
    piv["mean"] = piv.mean(axis=1)
    print(piv.sort_values("mean", ascending=False).round(3).to_string())
    print("\n=== GATE: RL >= greedy on TRUTH ===")
    for city, g in df.groupby("city"):
        gg = g.set_index("method")["true_d_utci_hot"]
        if "RL_DQN" in gg and "greedy" in gg:
            print(f"  {city:9} RL={gg['RL_DQN']:.3f}  greedy={gg['greedy']:.3f}  "
                  f"-> {'PASS' if gg['RL_DQN'] >= gg['greedy'] - 1e-6 else 'FAIL'}")
    # surrogate fidelity (backtest shrinkage)
    sub = df[df.pred_d_utci_hot.notna() & (df.method != "no_action")]
    if len(sub):
        corr = np.corrcoef(sub.true_d_utci_hot, sub.pred_d_utci_hot)[0, 1]
        print(f"\nSurrogate fidelity (pred vs true hotspot ΔUTCI): r={corr:.3f}  "
              f"mean shrinkage true/pred={ (sub.true_d_utci_hot/sub.pred_d_utci_hot.clip(0.01)).mean():.2f}")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
