"""④ SOLWEIG truth backtest for the SPLIT experiment.

Re-runs each method's saved plan through REAL SOLWEIG on a sample of the held-out TEST tiles,
computes TRUE cooling vs the tile baseline, and reports:
  (a) true-value method ranking on physics — does RL stay #1 when scored by real SOLWEIG?
  (b) surrogate fidelity on UNSEEN test tiles — surrogate-predicted vs true hotspot cooling
      (this is also the surrogate-generalisation check the reviewers will ask for).

Plans come from run_split_experiment.py --save-plans:
  results/ig/split_experiment/plans/<ck>/<sid>/<method>/{cdsm_tree_canopy,landcover}.tif

  MICROUPDATE_SOLWEIG_CPU=1 python scripts/ig_backtest_split.py --sample-per-city 15
  (optionally shard:  ... --city Beijing )
"""
from __future__ import annotations

from pathlib import Path
import os, sys, time, argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ig_generate_and_run_solweig import (
    hottest_3day_window, read_tif, parse_epw_location, DAY_HOURS, NIGHT_HOUR,
)
from scripts.ig_backtest_solweig import run_layout
from src.microupdate.action_space import LC_BUILDING, LC_WATER, LC_GRASS, LC_BARE
from src.microupdate.surrogate import lc_to_props
from src.microupdate.env import W_HOT, W_GROUND, W_PEN, W_NIGHT, NIGHT_NORM, W_STORM, W_SURF

IG = ROOT / "data/ig"
PLANS = ROOT / "results/ig/split_experiment/plans"
SPLITEXP = ROOT / "results/ig/split_experiment"
OUT = ROOT / "results/ig/split_backtest"
EXTREME = 38.0


def city_key(c):
    return c.lower().replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/candidate_tiles/splits.csv")
    ap.add_argument("--full-tiles", default="data/candidate_tiles/fullscale_tiles.csv")
    ap.add_argument("--sample-per-city", type=int, default=15)
    ap.add_argument("--city", default="", help="restrict to one city (for sharding); empty = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-tag", default="",
                    help="backtest a tagged ablation's plans: plans/<ck>_<tag>/ (e.g. nocov); output truth_<ck>_<tag>.csv")
    ap.add_argument("--methods", default="",
                    help="comma-separated methods to backtest (default: all in the plan dir; e.g. RL_DQN to only do RL)")
    args = ap.parse_args()

    import solweig
    if os.environ.get("MICROUPDATE_SOLWEIG_CPU"):
        solweig.disable_gpu()

    sp = pd.read_csv(ROOT / args.splits)
    full = pd.read_csv(ROOT / args.full_tiles)
    meta = full.set_index("site_id")[["center_lat", "center_lon"]]
    rng = np.random.default_rng(args.seed)

    test = sp[sp.split == "test"]
    if args.city:
        test = test[test.city == args.city]

    keep = {m.strip() for m in args.methods.split(",") if m.strip()} if args.methods else None
    rows = []
    for city, g in test.groupby("city"):
        ck = city_key(city)
        ckt = f"{ck}_{args.plan_tag}" if args.plan_tag else ck     # tagged ablation plans/csv
        scsv = SPLITEXP / f"{ckt}.csv"
        surro = pd.read_csv(scsv) if scsv.exists() else None
        sids = [s for s in g.site_id if (PLANS / ckt / s).exists()]
        rng.shuffle(sids)
        sids = sids[:args.sample_per_city]
        print(f"\n##### {city}: SOLWEIG backtest on {len(sids)} test tiles #####", flush=True)
        for sid in sids:
            if sid not in meta.index:
                print(f"  skip {sid} (no lat/lon)", flush=True); continue
            try:
                dem, _, _, crs = read_tif(IG / sid / "dem.tif")
                dsm, _, transform, _ = read_tif(IG / sid / "dsm.tif")
                lc_base = read_tif(IG / sid / "landcover_baseline.tif")[0]
                bu = read_tif(IG / sid / "scenarios/baseline/summary/utci_day_mean.tif")[0]
                bun = read_tif(IG / sid / "scenarios/baseline/summary/utci_night_mean.tif")[0]
                ground = (lc_base != LC_BUILDING) & (lc_base != LC_WATER)
                hot = ground & (bu >= np.quantile(bu[ground], 0.75))
                base_h38 = read_tif(IG / sid / "scenarios/baseline/summary/utci_hours_above_38_day.tif")[0]
                base_ph38 = float(base_h38[ground].sum())
                base_frac38 = float((bu[ground] > EXTREME).mean())     # for the multi-criteria objective
                ts_base = lc_to_props(lc_base)[1]                       # baseline surface temperature

                epw = ROOT / "data/sites" / sid / "solweig_inputs_pilot/weather.epw"
                start, end, _ = hottest_3day_window(epw)
                li = parse_epw_location(epw)
                r0 = meta.loc[sid]
                location = solweig.Location(latitude=float(r0.center_lat), longitude=float(r0.center_lon),
                                            altitude=li.get("altitude", 10.0), utc_offset=li.get("utc_offset", 0.0))
                weather = solweig.Weather.from_epw(epw, start=start, end=end, hours=DAY_HOURS + [NIGHT_HOUR])

                methods = sorted(p.name for p in (PLANS / ckt / sid).iterdir() if p.is_dir())
                if keep:
                    methods = [m for m in methods if m in keep]
                for method in methods:
                    mdir = PLANS / ckt / sid / method
                    cdsm = read_tif(mdir / "cdsm_tree_canopy.tif")[0]
                    lc = read_tif(mdir / "landcover.tif")[0]
                    t0 = time.time()
                    res = run_layout(solweig, sid, cdsm, lc, dsm, dem, transform, crs, weather, location, method)
                    d_hot = float((bu[hot] - res["utci_day_mean"][hot]).mean())
                    d_grd = float((bu[ground] - res["utci_day_mean"][ground]).mean())
                    d_night = float((bun[ground] - res["utci_night_mean"][ground]).mean())
                    ph38 = float(res["utci_hours_above_38_day"][ground].sum())
                    # multi-criteria objective on REAL physics: true day/night cooling + true extreme-heat
                    # reduction (surrogate terms, which over-optimisers shrink on truth) + GEOMETRIC
                    # stormwater & surface co-benefits (do NOT shrink — computed from the plan landcover)
                    true_frac38 = float((res["utci_day_mean"][ground] > EXTREME).mean())
                    pervious = (lc == LC_GRASS) | (lc == LC_BARE) | (lc == LC_WATER)
                    storm = float(pervious[ground].mean())
                    surface = float((ts_base[ground] - lc_to_props(lc)[1][ground]).mean())
                    true_obj = (W_HOT * d_hot + W_GROUND * d_grd + W_PEN * (base_frac38 - true_frac38)
                                + W_NIGHT * NIGHT_NORM * d_night + W_STORM * storm + W_SURF * surface)
                    pred = np.nan
                    if surro is not None:
                        sr = surro[(surro.site_id == sid) & (surro.method == method)]
                        if len(sr):
                            pred = float(sr.hot_cool.iloc[0])
                    rows.append({"city": city, "site_id": sid, "method": method,
                                 "true_hot": round(d_hot, 4), "pred_hot": round(pred, 4),
                                 "true_ground": round(d_grd, 4), "true_night": round(d_night, 4),
                                 "storm": round(storm, 4), "surface": round(surface, 4),
                                 "true_obj": round(true_obj, 4), "true_d_ph38": round(base_ph38 - ph38, 1)})
                    print(f"  {sid[:28]:28} {method:14} true={d_hot:+.3f} pred={pred:+.3f} [{time.time()-t0:.1f}s]", flush=True)
            except Exception as e:
                print(f"  !! {sid} failed: {e}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [city_key(args.city)] if args.city else ["all"]
    if args.plan_tag:
        parts.append(args.plan_tag)
    out_tag = "_".join(parts)
    df.to_csv(OUT / f"truth_{out_tag}.csv", index=False)

    if df.empty:
        print("\n!! no results — check plans dir / paths"); return
    print("\n=== TRUE (SOLWEIG) MULTI-CRITERIA OBJECTIVE by method ===")
    pv = df.pivot_table(index="method", values=["true_obj", "true_hot", "true_night", "storm", "surface"],
                        aggfunc="mean").sort_values("true_obj", ascending=False)
    print(pv.round(4).to_string())
    EXCL = ["RL_DQN", "tree100", "greedy_full", "RL_raw"]   # self, tree-fill ceiling, unbounded-eval ref, ablation
    if "RL_DQN" in pv.index:
        rl = pv.loc["RL_DQN", "true_obj"]; o = pv.drop(index=EXCL, errors="ignore")
        if len(o):                                   # multi-method backtest: full ranking
            print(f"  -> [TRUE OBJ] RL={rl:.4f}  best fair other={o.index[0]}({o.iloc[0]['true_obj']:.4f})  "
                  f"RL rank #{1 + int((o['true_obj'] > rl).sum())}")
            w = df.pivot_table(index=["city", "site_id"], columns="method", values="true_obj")
            fair = [m for m in w.columns if m not in ("tree100", "greedy_full", "RL_raw")]
            best = w[[m for m in fair if m != "RL_DQN"]].max(axis=1)
            print(f"  逐块真值配对 N={len(w)}: RL 全场第一 {(w[fair].idxmax(axis=1)=='RL_DQN').mean():.0%};  "
                  f"RL>=最佳其他 {(w['RL_DQN']>=best).mean():.0%}")
            for opp in ["BO", "greedy", "NSGA2", "equity"]:
                if opp in w:
                    print(f"    RL > {opp:14s} {(w['RL_DQN'] > w[opp]).mean():.0%}")
        else:                                        # single-method (ablation) backtest: just RL's true obj
            print(f"  -> [TRUE OBJ] RL_DQN={rl:.4f}  ({len(df)} rows, single-method ablation backtest)")
    sub = df[df.pred_hot.notna() & (df.method != "no_action")]
    if len(sub) > 2:
        r = np.corrcoef(sub.true_hot, sub.pred_hot)[0, 1]
        print(f"\n代理保真度(测试块,pred vs true 热点ΔUTCI):r={r:.3f}  缩水 true/pred={(sub.true_hot/sub.pred_hot.clip(0.01)).mean():.2f}")
    print(f"\nsaved {OUT / ('truth_' + out_tag + '.csv')}")


if __name__ == "__main__":
    main()
