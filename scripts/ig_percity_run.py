"""Per-city RL + full 15-baseline comparison (执行计划: 每城独立).

For each city: load THAT city's surrogate, build TileProblems for the city's tiles, train ONE
generalist RL policy on them, then run the full 15 baselines + RL on every city tile (all on
the city surrogate, same budget/N_eval). Saves plans under results/ig/rl/<sid>/<method>/ so
ig_backtest_solweig.py scores them on real SOLWEIG. Memory-slim + sequential (laptop-friendly).

Prereq: per-city surrogates exist (run scripts/ig_train_surrogate.py first).
.venv/bin/python scripts/ig_percity_run.py --tiles data/candidate_tiles/tiles12.csv --episodes 400
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
from src.microupdate import pick_device
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem, MicroUpdateEnv
from src.microupdate.dqn_generalist import GeneralistAgent, train_city, apply_policy
from src.microupdate import baselines as B

IG = ROOT / "data/ig"
OUT = ROOT / "results/ig/rl"
SURR = ROOT / "results/ig/surrogate"


def city_key(c): return c.lower().replace(" ", "")


# the full 15 baselines + RL; N = surrogate-evaluation budget (fair-compare; same for all search methods)
def build_baselines(N):
    return [
        ("no_action", lambda p, rng: B.no_action(p)),
        ("random", lambda p, rng: B.random_search(p, n_eval=N, rng=rng)),
        ("tree100", lambda p, rng: B.tree100(p)),
        ("best_found", lambda p, rng: B.best_found(p, n_eval=2 * N, rng=rng)),
        ("heat_priority", lambda p, rng: B.heat_priority(p)),
        ("plantable", lambda p, rng: B.plantable_priority(p)),
        ("equity", lambda p, rng: B.equity_priority(p)),
        ("canopy_cluster", lambda p, rng: B.canopy_clustering(p)),
        ("greedy", lambda p, rng: B.greedy(p)),
        ("GA", lambda p, rng: B.ga(p, n_eval=N, rng=rng)),
        ("SA", lambda p, rng: B.sa(p, n_eval=N, rng=rng)),
        ("NSGA2", lambda p, rng: B.nsga2(p, n_eval=N, rng=rng)),
        ("PSO", lambda p, rng: B.pso(p, n_eval=N, rng=rng)),
        ("ACO", lambda p, rng: B.aco(p, n_eval=N, rng=rng)),
        ("BO", lambda p, rng: B.bo(p, n_eval=min(N, 400), rng=rng)),
    ]


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
    ap.add_argument("--tiles", default="data/candidate_tiles/tiles12.csv")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = pick_device()
    print(f"device={device}", flush=True)
    tiles = pd.read_csv(ROOT / args.tiles)
    baseline_specs = build_baselines(args.n_eval)

    rows = []
    for city, grp in tiles.groupby("city"):
        ck = city_key(city)
        spath = SURR / f"surrogate_{ck}.pt"
        if not spath.exists():
            print(f"!! missing {spath} — run ig_train_surrogate.py first; skipping {city}")
            continue
        model = SurrogateCNN(); model.load_state_dict(torch.load(spath, map_location=device))
        model.to(device).eval()
        sids = list(grp.site_id)
        problems = [TileProblem(s, model, device=device) for s in sids]
        envs = [MicroUpdateEnv(p, max_steps=p.K) for p in problems]
        print(f"\n=== {city}: {len(sids)} tiles, surrogate_{ck} | train generalist RL ({args.episodes} ep) ===", flush=True)
        t0 = time.time()
        agent = GeneralistAgent(device=device)
        train_city(envs, agent, episodes=args.episodes, rng=np.random.default_rng(args.seed), log=print)
        print(f"  RL trained in {time.time()-t0:.0f}s; running 15 baselines + RL on each tile", flush=True)

        for sid, problem, env in zip(sids, problems, envs):
            rng = np.random.default_rng(args.seed + 1)
            results = {}
            for name, fn in baseline_specs:
                plan = fn(problem, rng)
                results[name] = (plan, B.plan_metrics(problem, plan))
            rl_plan, rl_m = apply_policy(env, agent)        # generalist policy applied to this tile
            results["RL_DQN"] = (rl_plan, rl_m)
            for name, (plan, m) in results.items():
                save_layout(problem, plan, OUT / sid / name)
                rows.append({"city": city, "site_id": sid, "method": name, **m,
                             "K": problem.K, "budget": problem.budget})
            best = max(results, key=lambda n: results[n][1]["mean_cool_hot"])
            print(f"    {sid[:42]:42} best_hot={best}({results[best][1]['mean_cool_hot']:.3f}) "
                  f"RL={rl_m['mean_cool_hot']:.3f} greedy={results['greedy'][1]['mean_cool_hot']:.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "surrogate_comparison.csv", index=False)
    # per-city RL-vs-best-baseline on the primary (hotspot) metric
    print("\n=== per-city hotspot cooling: RL vs best baseline (surrogate) ===")
    for city, g in df.groupby("city"):
        piv = g.pivot_table(index="site_id", columns="method", values="mean_cool_hot")
        rl = piv["RL_DQN"]; basemax = piv.drop(columns=["RL_DQN", "tree100"], errors="ignore").max(axis=1)
        wins = int((rl >= basemax - 1e-6).sum())
        print(f"  {city:9}: RL >= best-budget-baseline on {wins}/{len(piv)} tiles "
              f"(RL mean {rl.mean():.3f} vs best-baseline mean {basemax.mean():.3f})")
    print(f"\nSaved {OUT/'surrogate_comparison.csv'}")


if __name__ == "__main__":
    main()
