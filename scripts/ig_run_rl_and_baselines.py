"""Stage 4: train Dueling Double DQN per tile, run the baseline suite on the shared
surrogate, and compare. Gate: RL objective >= greedy and beats domain rule + metaheuristics.
Saves each method's plan (placements + layout rasters) for the SOLWEIG backtest (Stage 5).

.venv/bin/python scripts/ig_run_rl_and_baselines.py [--episodes 300] [--n-eval 3000]
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

IG = ROOT / "data/ig"
OUT = ROOT / "results/ig/rl"
PILOTS = pd.read_csv(ROOT / "data/candidate_tiles/pilot4_tiles.csv")


def save_layout(problem, placements, path: Path):
    L = B.layout_from_plan(problem, placements)
    path.mkdir(parents=True, exist_ok=True)
    tr = rasterio.open(IG / problem.site_id / "dsm.tif").transform
    crs = rasterio.open(IG / problem.site_id / "dsm.tif").crs
    for name, arr in [("cdsm_tree_canopy.tif", L.cdsm.astype(np.float32)),
                      ("landcover.tif", L.lc.astype(np.uint8))]:
        with rasterio.open(path / name, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype=str(arr.dtype), crs=crs, transform=tr, compress="lzw") as d:
            d.write(arr, 1)
    json.dump([[int(a), int(b)] for a, b in placements], open(path / "placements.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--tiles", default="data/candidate_tiles/pilot4_tiles.csv")
    ap.add_argument("--site-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from src.microupdate import pick_device
    device = pick_device()

    model = SurrogateCNN()
    model.load_state_dict(torch.load(ROOT / "results/ig/surrogate/surrogate_cnn.pt", map_location=device))
    model.to(device).eval()

    pilots = pd.read_csv(ROOT / args.tiles)
    if args.site_id:
        pilots = pilots[pilots.site_id == args.site_id]
    rows = []
    for _, prow in pilots.iterrows():
        sid = prow["site_id"]; city = prow["city"]
        t0 = time.time()
        problem = TileProblem(sid, model, device=device)
        print(f"\n=== {city} ({sid[:34]})  K={problem.K} budget={problem.budget:.0f} ===", flush=True)

        methods = {}
        rng = np.random.default_rng(args.seed)
        methods["no_action"] = B.no_action(problem)
        methods["random"] = B.random_search(problem, n_eval=args.n_eval, rng=np.random.default_rng(args.seed))
        methods["tree100"] = B.tree100(problem)
        methods["heat_priority"] = B.heat_priority(problem)
        methods["greedy"] = B.greedy(problem)
        methods["GA"] = B.ga(problem, n_eval=args.n_eval, rng=np.random.default_rng(args.seed + 1))
        methods["SA"] = B.sa(problem, n_eval=args.n_eval, rng=np.random.default_rng(args.seed + 2))
        for name, plan in methods.items():
            m = B.plan_metrics(problem, plan)
            print(f"  {name:14} obj={m['objective']:+.3f} cool_hot={m['mean_cool_hot']:.3f} "
                  f"cool_grd={m['mean_cool_ground']:.3f} cost={m['cost']:.0f} n={m['n_actions']}", flush=True)

        # RL — max_steps = K so the budget (not a step cap) is the binding constraint
        env = MicroUpdateEnv(problem, max_steps=problem.K)
        from src.microupdate.action_space import N_PLACEMENT_ACTIONS
        sh = 96  # state grid (matches env._state downsample)
        cand96 = np.stack([np.clip(np.round(problem.cand[:, 0] * sh / problem.H), 0, sh - 1),
                           np.clip(np.round(problem.cand[:, 1] * sh / problem.W), 0, sh - 1)], axis=1).astype(int)
        agent = DQNAgent(in_ch=6, n_scalar=8, n_actions=env.n_actions, device=device,
                         cand_coords=cand96, n_types=N_PLACEMENT_ACTIONS)
        train_dqn(env, agent, episodes=args.episodes, log=max(50, args.episodes // 6))
        rl_plan, _, rl_m = rollout_plan(env, agent)
        methods["RL_DQN"] = rl_plan
        print(f"  {'RL_DQN':14} obj={rl_m['objective']:+.3f} cool_hot={rl_m['mean_cool_hot']:.3f} "
              f"cool_grd={rl_m['mean_cool_ground']:.3f} cost={rl_m['cost']:.0f} n={rl_m['n_actions']}", flush=True)

        for name, plan in methods.items():
            m = B.plan_metrics(problem, plan)
            rows.append({"city": city, "site_id": sid, "method": name, **m, "K": problem.K,
                         "budget": problem.budget})
            save_layout(problem, plan, OUT / sid / name)
        print(f"  [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    cpath = OUT / "surrogate_comparison.csv"
    if cpath.exists():                       # merge: keep other tiles' rows, replace re-run ones
        old = pd.read_csv(cpath)
        old = old[~old.site_id.isin(df.site_id.unique())]
        pd.concat([old, df], ignore_index=True).to_csv(cpath, index=False)
    else:
        df.to_csv(cpath, index=False)
    print("\n=== SURROGATE-OBJECTIVE COMPARISON (per tile) ===")
    piv = df.pivot_table(index="method", columns="city", values="objective").round(3)
    # order methods by mean objective
    piv["mean"] = piv.mean(axis=1)
    print(piv.sort_values("mean", ascending=False).to_string())
    # gate check
    print("\n=== GATE: RL vs greedy/domain/metaheuristic (objective) ===")
    for city, g in df.groupby("city"):
        gg = g.set_index("method")["objective"]
        rl = gg["RL_DQN"]
        comps = {k: gg[k] for k in ["greedy", "heat_priority", "GA", "SA"] if k in gg}
        verdict = "PASS" if all(rl >= v - 1e-6 for v in comps.values()) else "FAIL"
        print(f"  {city:9} RL={rl:.3f}  greedy={comps.get('greedy',float('nan')):.3f}  "
              f"GA={comps.get('GA',float('nan')):.3f}  SA={comps.get('SA',float('nan')):.3f}  -> {verdict}")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
