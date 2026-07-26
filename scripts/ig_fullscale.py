"""FULL-SCALE experiment (执行计划: 方案搜索在完整 benchmark 上做).

The small representative tiles were ONLY for fitting the per-city surrogate. The EXPERIMENT —
RL + all 15 baselines — runs on the city's FULL usable tile set (≥50 buildings) via the
surrogate (no SOLWEIG in the search loop). One city per call (shard across GPUs).

Per city:
  1. load that city's surrogate (trained on the small set)
  2. train the generalist RL policy on the city's representative tiles (have full scenario data)
  3. apply RL + 15 baselines to EVERY full-set tile (needs each tile's baseline SOLWEIG summary)
  4. write per-tile metrics for all methods -> results/ig/fullscale/<city>.csv

Prereq: per-city surrogate (ig_train_surrogate), representative scenario data (the 12 tiles),
and baseline SOLWEIG for the full-set tiles (ig_generate_and_run_solweig --baseline-only).

  CUDA_VISIBLE_DEVICES=0 python scripts/ig_fullscale.py --city Beijing
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
import torch

sys.path.insert(0, str(ROOT))
from src.microupdate import pick_device
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem, MicroUpdateEnv
from src.microupdate.dqn_generalist import GeneralistAgent, train_city, optimize_tile
from src.microupdate import baselines as B
from scripts.ig_percity_run import build_baselines, city_key

IG = ROOT / "data/ig"
SURR = ROOT / "results/ig/surrogate"
OUT = ROOT / "results/ig/fullscale"


def has_baseline(sid: str) -> bool:
    return (IG / sid / "scenarios/baseline/summary/utci_day_mean.tif").exists() and (IG / sid / "dsm.tif").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--rep-tiles", default="data/candidate_tiles/tiles12.csv", help="surrogate/RL-training tiles")
    ap.add_argument("--full-tiles", default="data/candidate_tiles/fullscale_tiles.csv", help="the FULL experiment set")
    ap.add_argument("--episodes", type=int, default=400, help="generalist warm-start episodes (on rep tiles)")
    ap.add_argument("--rl-episodes", type=int, default=50, help="PER-TILE RL optimisation episodes (warm-started)")
    ap.add_argument("--n-eval", type=int, default=1000)   # fair budget for all search methods; lower=faster at scale
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = pick_device()
    ck = city_key(args.city)
    print(f"[{args.city}] device={device}  surrogate=surrogate_{ck}", flush=True)

    model = SurrogateCNN()
    model.load_state_dict(torch.load(SURR / f"surrogate_{ck}.pt", map_location=device))
    model.to(device).eval()

    # 1) train the generalist policy on this city's representative tiles
    rep = pd.read_csv(ROOT / args.rep_tiles)
    rep = rep[rep.city == args.city]
    rep_probs = [TileProblem(s, model, device=device) for s in rep.site_id]
    rep_envs = [MicroUpdateEnv(p, max_steps=p.K) for p in rep_probs]
    t0 = time.time()
    agent = GeneralistAgent(device=device)
    train_city(rep_envs, agent, episodes=args.episodes, rng=np.random.default_rng(args.seed))
    warm_state = {k: v.clone() for k, v in agent.online.state_dict().items()}   # warm-start for per-tile RL
    print(f"[{args.city}] generalist warm-start trained on {len(rep_envs)} rep tiles in {time.time()-t0:.0f}s", flush=True)

    # 2) the FULL experiment set for this city
    full = pd.read_csv(ROOT / args.full_tiles)
    full = full[full.city == args.city]
    sids = [s for s in full.site_id if has_baseline(s)]
    missing = len(full) - len(sids)
    print(f"[{args.city}] FULL set: {len(full)} usable tiles, {len(sids)} with baseline SOLWEIG ready"
          f"{f' ({missing} missing baseline -> skipped)' if missing else ''}", flush=True)

    specs = build_baselines(args.n_eval)
    rows = []
    t0 = time.time()
    for k, sid in enumerate(sids):
        try:
            problem = TileProblem(sid, model, device=device)
            env = MicroUpdateEnv(problem, max_steps=problem.K)
            rng = np.random.default_rng(args.seed + 1)
            res = {name: B.plan_metrics(problem, fn(problem, rng)) for name, fn in specs}
            # PER-TILE RL: optimise THIS tile (warm-started), surrogate = the temperature reward
            _, rl_m = optimize_tile(env, device=device, episodes=args.rl_episodes,
                                    warm_state=warm_state, rng=np.random.default_rng(args.seed + k))
            res["RL_DQN"] = rl_m
            for name, m in res.items():
                rows.append({"city": args.city, "site_id": sid, "method": name,
                             "hot_cool": m["mean_cool_hot"], "ground_cool": m["mean_cool_ground"],
                             "objective": m["objective"], "cost": m["cost"], "n_actions": m["n_actions"]})
        except Exception as e:
            print(f"[{args.city}] !! tile {sid} failed: {e}", flush=True)
        if (k + 1) % 20 == 0:
            print(f"[{args.city}] {k+1}/{len(sids)} tiles  ({(time.time()-t0)/(k+1):.1f}s/tile)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{ck}.csv", index=False)
    # headline: RL vs each method, mean hotspot cooling over the FULL set
    piv = df.pivot_table(index="method", values="hot_cool", aggfunc="mean").sort_values("hot_cool", ascending=False)
    nt = df.site_id.nunique()
    print(f"\n[{args.city}] === FULL-SET ({nt} tiles) mean hotspot ΔUTCI (surrogate) ===")
    print(piv.round(3).to_string())
    rl = piv.loc["RL_DQN", "hot_cool"]
    budget = piv.drop(index=["RL_DQN", "tree100"], errors="ignore")
    print(f"[{args.city}] RL={rl:.3f}  best baseline={budget.iloc[0].name}({budget.iloc[0,0]:.3f})  "
          f"rank of RL among budget methods: #{1 + (budget['hot_cool'] > rl).sum()}")
    print(f"[{args.city}] saved {OUT/(ck+'.csv')}  ({nt} tiles x {df.method.nunique()} methods)")


if __name__ == "__main__":
    main()
