"""Build the scene-level label table from the IG SOLWEIG outputs.

Per scenario, relative to its tile baseline, compute the metrics the reward/analysis
need: coverage, ground-mean ΔUTCI, hotspot ΔUTCI (hottest-quartile ground), and
>38°C day-exceedance cell reduction. Writes data/ig/scene_dataset.csv.

.venv/bin/python scripts/ig_build_scene_dataset.py
"""
from __future__ import annotations

from pathlib import Path
import os, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RASTERIO_PROJ = Path(sys.executable).resolve().parents[1] / "lib/python3.12/site-packages/rasterio/proj_data"
if RASTERIO_PROJ.exists():
    os.environ["PROJ_DATA"] = str(RASTERIO_PROJ); os.environ["PROJ_LIB"] = str(RASTERIO_PROJ)
import rasterio
sys.path.insert(0, str(ROOT))
from src.microupdate.action_space import LC_BUILDING

IG = ROOT / "data/ig"


def read(p):
    with rasterio.open(p) as s:
        return s.read(1)


def main():
    man = pd.read_csv(IG / "scenario_manifest.csv")
    rows = []
    for sid, grp in man.groupby("site_id"):
        sdir = IG / sid / "scenarios"
        lc_base = read(IG / sid / "landcover_baseline.tif")
        ground = lc_base != LC_BUILDING
        bu = read(sdir / "baseline/summary/utci_day_mean.tif")
        bt = read(sdir / "baseline/summary/tmrt_day_mean.tif")
        b38 = read(sdir / "baseline/summary/utci_hours_above_38_day.tif")
        hot = ground & (bu >= np.quantile(bu[ground], 0.75))
        base_n38 = int(((b38 > 0) & ground).sum())
        n_ground = int(ground.sum())
        for _, r in grp.iterrows():
            scn = r.scenario_id
            sc = sdir / scn
            u = read(sc / "summary/utci_day_mean.tif")
            t = read(sc / "summary/tmrt_day_mean.tif")
            e = read(sc / "summary/utci_hours_above_38_day.tif")
            mod = read(sc / "modified_mask.tif") > 0
            rows.append({
                "site_id": sid, "city": r.city, "scenario_id": scn,
                "n_actions": int(r.n_actions), "cost": float(r.cost),
                "coverage": round(float((mod & ground).sum()) / n_ground, 4),
                "d_utci_ground": round(float((bu[ground] - u[ground]).mean()), 4),
                "d_utci_hot": round(float((bu[hot] - u[hot]).mean()), 4),
                "d_tmrt_hot": round(float((bt[hot] - t[hot]).mean()), 4),
                "d_n38": int(base_n38 - int(((e > 0) & ground).sum())),
                "base_n38": base_n38, "n_ground": n_ground,
                **{k: int(r[k]) for k in r.index if k.startswith("n_") and k != "n_actions"},
            })
    df = pd.DataFrame(rows)
    df.to_csv(IG / "scene_dataset.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"Wrote {IG/'scene_dataset.csv'}  ({len(df)} rows)\n")
    print("=== signal range per city (non-baseline scenarios) ===")
    nb = df[df.scenario_id != "baseline"]
    print(nb.groupby("city")[["coverage", "d_utci_ground", "d_utci_hot", "d_tmrt_hot", "d_n38"]]
          .agg(["min", "median", "max"]).round(2).to_string())
    print("\n=== single-action saturation (hi) — which levers cool the hotspots? ===")
    hi = df[df.scenario_id.str.contains("_hi$", regex=True)]
    print(hi[["city", "scenario_id", "coverage", "d_utci_hot", "d_tmrt_hot", "d_n38"]].to_string(index=False))


if __name__ == "__main__":
    main()
