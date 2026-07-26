"""Add PSO + ACO baselines to the existing pilot tiles on the shared surrogate (no RL
retrain). Saves their plans next to the others so ig_backtest_solweig.py picks them up.

.venv/bin/python scripts/ig_add_metaheuristics.py
"""
from __future__ import annotations

from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RASTERIO_PROJ = Path(sys.executable).resolve().parents[1] / "lib/python3.12/site-packages/rasterio/proj_data"
if RASTERIO_PROJ.exists():
    os.environ["PROJ_DATA"] = str(RASTERIO_PROJ); os.environ["PROJ_LIB"] = str(RASTERIO_PROJ)
import rasterio
import torch

sys.path.insert(0, str(ROOT))
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem
from src.microupdate import baselines as B

IG = ROOT / "data/ig"
RL = ROOT / "results/ig/rl"
PILOTS = pd.read_csv(ROOT / "data/candidate_tiles/pilot4_tiles.csv")


def save_layout(problem, placements, path: Path):
    L = B.layout_from_plan(problem, placements)
    path.mkdir(parents=True, exist_ok=True)
    src = rasterio.open(IG / problem.site_id / "dsm.tif")
    for name, arr in [("cdsm_tree_canopy.tif", L.cdsm.astype(np.float32)),
                      ("landcover.tif", L.lc.astype(np.uint8))]:
        with rasterio.open(path / name, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype=str(arr.dtype), crs=src.crs, transform=src.transform,
                           compress="lzw") as d:
            d.write(arr, 1)
    json.dump([[int(a), int(b)] for a, b in placements], open(path / "placements.json", "w"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/candidate_tiles/pilot4_tiles.csv")
    args = ap.parse_args()
    global PILOTS
    PILOTS = pd.read_csv(ROOT / args.tiles)
    from src.microupdate import pick_device
    device = pick_device()
    model = SurrogateCNN()
    model.load_state_dict(torch.load(ROOT / "results/ig/surrogate/surrogate_cnn.pt", map_location=device))
    model.to(device).eval()
    comp = pd.read_csv(RL / "surrogate_comparison.csv")
    comp = comp[~comp.method.isin(["PSO", "ACO"])]  # idempotent
    rows = []
    for _, prow in PILOTS.iterrows():
        sid = prow["site_id"]
        problem = TileProblem(sid, model, device=device)
        for name, fn in [("PSO", B.pso), ("ACO", B.aco)]:
            plan = fn(problem, n_eval=2000, rng=np.random.default_rng(7))
            m = B.plan_metrics(problem, plan)
            save_layout(problem, plan, RL / sid / name)
            rows.append({"city": prow["city"], "site_id": sid, "method": name, **m,
                         "K": problem.K, "budget": problem.budget})
            print(f"  {prow['city']:9} {name}: obj={m['objective']:.3f} cool_hot={m['mean_cool_hot']:.3f} "
                  f"n={m['n_actions']}", flush=True)
    out = pd.concat([comp, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(RL / "surrogate_comparison.csv", index=False)
    print("\nUpdated surrogate_comparison.csv with PSO+ACO")


if __name__ == "__main__":
    main()
