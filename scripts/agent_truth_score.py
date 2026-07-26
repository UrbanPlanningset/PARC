"""exp3 (scoring): run REAL SOLWEIG on every saved agentic layout (from agent_truth_gen.py) and
compute the truth multi-criteria objective with the SAME formula as the main-paper backtest.

MUST run with the locked env: MICROUPDATE_NIGHT_NORM=10 MICROUPDATE_W_STORM=5 MICROUPDATE_W_SURF=5
  CUDA_VISIBLE_DEVICES=0 python scripts/agent_truth_score.py --city-key beijing
"""
from __future__ import annotations
from pathlib import Path
import os, sys, time, argparse
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ig_generate_and_run_solweig import (hottest_3day_window, read_tif, parse_epw_location,
                                                 DAY_HOURS, NIGHT_HOUR)
from scripts.ig_backtest_solweig import run_layout
from src.microupdate.action_space import LC_BUILDING, LC_WATER, LC_GRASS, LC_BARE
from src.microupdate.surrogate import lc_to_props
from src.microupdate.env import W_HOT, W_GROUND, W_PEN, W_NIGHT, NIGHT_NORM, W_STORM, W_SURF

IG = ROOT / "data/ig"
PLANS = ROOT / "results/agentic/truth_plans"
EX = 38.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-key", required=True)
    a = ap.parse_args()
    import solweig
    full = pd.read_csv(ROOT / "data/candidate_tiles/fullscale_tiles.csv").set_index("site_id")
    ck = a.city_key
    rows = []
    for tdir in sorted((PLANS / ck).iterdir()):
        sid = tdir.name
        try:
            dem, _, _, crs = read_tif(IG / sid / "dem.tif")
            dsm, _, transform, _ = read_tif(IG / sid / "dsm.tif")
            lc_base = read_tif(IG / sid / "landcover_baseline.tif")[0]
            bu = read_tif(IG / sid / "scenarios/baseline/summary/utci_day_mean.tif")[0]
            bun = read_tif(IG / sid / "scenarios/baseline/summary/utci_night_mean.tif")[0]
        except Exception as e:
            print(f"skip {sid[:30]}: {e}", flush=True); continue
        ground = (lc_base != LC_BUILDING) & (lc_base != LC_WATER)
        hot = ground & (bu >= np.quantile(bu[ground], 0.75))
        base_frac38 = float((bu[ground] > EX).mean())
        ts_base = lc_to_props(lc_base)[1]
        epw = ROOT / "data/sites" / sid / "solweig_inputs_pilot/weather.epw"
        start, end, _ = hottest_3day_window(epw)
        li = parse_epw_location(epw)
        r0 = full.loc[sid]
        location = solweig.Location(latitude=float(r0.center_lat), longitude=float(r0.center_lon),
                                    altitude=li.get("altitude", 10.0), utc_offset=li.get("utc_offset", 0.0))
        weather = solweig.Weather.from_epw(epw, start=start, end=end, hours=DAY_HOURS + [NIGHT_HOUR])
        for vdir in sorted(x for x in tdir.iterdir() if x.is_dir()):
            try:
                cdsm = read_tif(vdir / "cdsm_tree_canopy.tif")[0]
                lc = read_tif(vdir / "landcover.tif")[0]
                t0 = time.time()
                res = run_layout(solweig, sid, cdsm, lc, dsm, dem, transform, crs, weather, location, vdir.name)
            except Exception as e:
                print(f"!! {sid[-16:]} {vdir.name}: {e}", flush=True); continue
            d_hot = float((bu[hot] - res["utci_day_mean"][hot]).mean())
            d_grd = float((bu[ground] - res["utci_day_mean"][ground]).mean())
            d_night = float((bun[ground] - res["utci_night_mean"][ground]).mean())
            tf = float((res["utci_day_mean"][ground] > EX).mean())
            perv = (lc == LC_GRASS) | (lc == LC_BARE) | (lc == LC_WATER)
            storm = float(perv[ground].mean())
            surf = float((ts_base[ground] - lc_to_props(lc)[1][ground]).mean())
            tobj = (W_HOT * d_hot + W_GROUND * d_grd + W_PEN * (base_frac38 - tf)
                    + W_NIGHT * NIGHT_NORM * d_night + W_STORM * storm + W_SURF * surf)
            rows.append(dict(ck=ck, tile=sid, variant=vdir.name, true_obj=round(tobj, 4),
                             true_hot=round(d_hot, 4), true_night=round(d_night, 4),
                             storm=round(storm, 4), surface=round(surf, 4)))
            print(f"{ck} {sid[-14:]} {vdir.name:22} true={tobj:.4f} [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / f"results/agentic/truth_scores_{ck}.csv", index=False)
    print(f"saved truth_scores_{ck}.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
