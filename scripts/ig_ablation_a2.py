"""Ablation A2 (执行计划 §2.2): is the Dueling Double design necessary?

On one tile, train 4 variants under identical config — full (Dueling+Double),
−Dueling, −Double, naive (neither) — plus greedy as reference. Saves each plan as a
method dir so ig_backtest_solweig.py scores it on real SOLWEIG.

.venv/bin/python scripts/ig_ablation_a2.py [--site-id ...] [--episodes 250]
"""
from __future__ import annotations

from pathlib import Path
import os, sys, json, time, argparse
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
from src.microupdate.env import TileProblem, MicroUpdateEnv
from src.microupdate.dqn import DQNAgent, train_dqn, rollout_plan
from src.microupdate import baselines as B
from src.microupdate.action_space import N_PLACEMENT_ACTIONS

IG = ROOT / "data/ig"
RL = ROOT / "results/ig/rl"
OUT = ROOT / "results/ig/ablation"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-id", default="nyc_core_nyc_core_midtown_gx0001_gy0001")
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from src.microupdate import pick_device
    device = pick_device()
    model = SurrogateCNN()
    model.load_state_dict(torch.load(ROOT / "results/ig/surrogate/surrogate_cnn.pt", map_location=device))
    model.to(device).eval()

    sid = args.site_id
    problem = TileProblem(sid, model, device=device)
    sh = 96
    cand96 = np.stack([np.clip(np.round(problem.cand[:, 0] * sh / problem.H), 0, sh - 1),
                       np.clip(np.round(problem.cand[:, 1] * sh / problem.W), 0, sh - 1)], axis=1).astype(int)

    variants = [("full_DD", True, True), ("noDueling", False, True),
                ("noDouble", True, False), ("naive", False, False)]
    rows = []
    # greedy reference
    gplan = B.greedy(problem); gm = B.plan_metrics(problem, gplan)
    rows.append({"variant": "greedy_ref", "objective": gm["objective"], "cool_hot": gm["mean_cool_hot"],
                 "n_actions": gm["n_actions"]})
    save_layout(problem, gplan, RL / sid / "ablA2_greedy_ref")
    print(f"=== Ablation A2 on {sid}  ({args.episodes} episodes/variant) ===", flush=True)
    print(f"  greedy_ref      obj={gm['objective']:.3f} cool_hot={gm['mean_cool_hot']:.3f}", flush=True)

    for name, dueling, double in variants:
        t0 = time.time()
        env = MicroUpdateEnv(problem, max_steps=problem.K)
        agent = DQNAgent(in_ch=6, n_scalar=8, n_actions=env.n_actions, device=device,
                         cand_coords=cand96, n_types=N_PLACEMENT_ACTIONS, dueling=dueling, double=double)
        train_dqn(env, agent, episodes=args.episodes, log=None)
        plan, _, m = rollout_plan(env, agent)
        save_layout(problem, plan, RL / sid / f"ablA2_{name}")
        rows.append({"variant": name, "objective": m["objective"], "cool_hot": m["mean_cool_hot"],
                     "n_actions": m["n_actions"]})
        print(f"  {name:15} obj={m['objective']:.3f} cool_hot={m['mean_cool_hot']:.3f} "
              f"n={m['n_actions']} [{time.time()-t0:.0f}s]", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"a2_{sid}.csv", index=False)
    print("\n=== A2 surrogate comparison (higher = better) ===")
    print(df.to_string(index=False))
    print(f"\nSaved plans as ablA2_* under {RL/sid} — run ig_backtest_solweig.py for truth.")


if __name__ == "__main__":
    main()
