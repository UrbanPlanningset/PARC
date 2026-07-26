"""Extended baselines for the KDD constraint battery (NO LLM calls; pure local compute).

Part A  classical-optimizer family: score the SAVED plans of GA/SA/NSGA2/PSO/ACO/BO/greedy/random/
        heat_priority/canopy_cluster against all 12 constraints -> shows the whole fixed-optimizer
        family, not just RL, cannot adapt.
Part B  rule-based repair: a deterministic, hand-coded repair operator per constraint applied to the
        RL plan -> the strongest NON-LLM adaptive baseline (the "why do you even need an LLM?" answer).

Run with locked env:
  MICROUPDATE_NIGHT_NORM=10 MICROUPDATE_W_STORM=5 MICROUPDATE_W_SURF=5 \
  python scripts/agent_baselines2.py --cities beijing --tiles-per-city 8
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, glob, argparse
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from agent_prototype import score
from agent_eval import CONS, _nt
import torch
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem
from src.microupdate.action_space import ACTIONS

SURR = ROOT / "results/ig/surrogate"
OUT = ROOT / "results/agentic"; OUT.mkdir(parents=True, exist_ok=True)
CITIES = [("Beijing", "beijing"), ("Shanghai", "shanghai"), ("Seoul", "seoul"), ("New York", "newyork")]
FAMILY = ["BO", "GA", "SA", "NSGA2", "PSO", "ACO", "greedy", "random", "heat_priority", "canopy_cluster"]
TREES = (0, 1, 2)


# ---------------- rule-based repair operators ----------------
def _tau(p):
    return p.baseline_utci[p.cand[:, 0], p.cand[:, 1]]


def _cost(P): return sum(ACTIONS[t].cost for _, t in P)


def _fill(p, P, budget, ttype, nmax=10**6):
    """hotspot-first fill with type ttype until budget exhausted (respects action_ok)."""
    tau = _tau(p); used = {l for l, _ in P}
    order = np.argsort(-tau)
    c = _cost(P); a = ACTIONS[ttype]; placed = 0
    for i in order:
        i = int(i)
        if placed >= nmax: break
        if i in used or not p.action_ok[i, a.aid] or c + a.cost > budget: continue
        P.append((i, a.aid)); used.add(i); c += a.cost; placed += 1
    return P


def _drop_coolest(p, P, keep_pred, budget):
    """drop placements at coolest cells (matching keep_pred=False first) until under budget."""
    tau = _tau(p)
    P_sorted = sorted(P, key=lambda x: tau[x[0]])          # coolest first
    out = list(P)
    for l, t in P_sorted:
        if _cost(out) <= budget: break
        if not keep_pred(t):
            out.remove((l, t))
    for l, t in sorted(out, key=lambda x: tau[x[0]]):      # still over: drop anything coolest-first
        if _cost(out) <= budget: break
        out.remove((l, t))
    return out


def repair(p, rl_plan, rl, cid, eff):
    """deterministic repair of the RL plan for constraint cid; returns a plan."""
    tau = _tau(p)
    P = list(rl_plan)
    if cid in ("no_water", "no_cool", "no_large"):
        ban = {"no_water": 5, "no_cool": 3, "no_large": 2}[cid]
        P = [(l, t) for l, t in P if t != ban]
        return _fill(p, P, eff, 0)                          # refill freed budget with small trees
    if cid == "trees_only":
        P = [(l, t) for l, t in P if t in TREES]
        return _fill(p, P, eff, 0)
    if cid in ("budget120", "budget80"):
        return _drop_coolest(p, P, lambda t: t in TREES, eff)
    if cid == "max30":
        P_sorted = sorted(P, key=lambda x: -tau[x[0]])
        return P_sorted[:30]
    if cid in ("storm_up", "night_up", "surface_up", "min_nt30", "min_perm"):
        add_type = {"storm_up": 7, "night_up": 4, "surface_up": 3, "min_nt30": 7, "min_perm": 7}[cid]
        # iteratively swap coolest trees -> co-benefit type until the checker passes (max 12 rounds)
        chk = next(c for c in CONS if c[0] == cid)[3]
        for k in range(1, 13):
            trees_sorted = [x for x in sorted(P, key=lambda y: tau[y[0]]) if x[1] in TREES]
            Q = [x for x in P if x not in trees_sorted[:3 * k]]
            Q = _fill(p, Q, eff, add_type)
            s = score(p, Q)
            if chk(Q, s, rl):
                return Q
        return Q
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="all")
    ap.add_argument("--tiles-per-city", type=int, default=8)
    a = ap.parse_args()
    cities = CITIES if a.cities == "all" else [c for c in CITIES if c[1] in a.cities.split(",")]
    rows = []
    for disp, ck in cities:
        model = SurrogateCNN(); model.load_state_dict(torch.load(SURR / f"surrogate_{ck}.pt", map_location="cpu")); model.eval()
        pdir = ROOT / f"results/ig/split_experiment/plans/{ck}"
        sids = [Path(x).name for x in sorted(glob.glob(str(pdir / "*")))
                if (Path(x) / "RL_DQN/placements.json").exists()][:a.tiles_per_city]
        for sid in sids:
            try:
                p = TileProblem(sid, model, device="cpu")
            except Exception as e:
                print("skip", sid[:30], e, flush=True); continue
            rl_plan = [(int(x), int(y)) for x, y in json.load(open(pdir / sid / "RL_DQN/placements.json"))]
            rl = score(p, rl_plan)
            plans = {}
            for m in FAMILY:                                # part A: fixed family
                f = pdir / sid / m / "placements.json"
                if f.exists():
                    plans[m] = [(int(x), int(y)) for x, y in json.load(open(f))]
            for cid, nl, bud, check, cat in CONS:
                eff = bud if bud is not None else p.budget
                for m, P in plans.items():
                    s = score(p, P)
                    rows.append(dict(city=disp, tile=sid, cid=cid, cat=cat, method=m,
                                     ok=check(P, s, rl), obj=s["objective"], nt=_nt(P)))
                Q = repair(p, rl_plan, rl, cid, eff)        # part B: rule repair
                sq = score(p, Q)
                rows.append(dict(city=disp, tile=sid, cid=cid, cat=cat, method="rule_repair",
                                 ok=check(Q, sq, rl), obj=sq["objective"], nt=_nt(Q)))
            print(f"{ck} {sid[-18:]} done", flush=True)
    df = pd.DataFrame(rows)
    tag = "_".join(c[1] for c in cities)
    df.to_csv(OUT / f"baselines2_{tag}.csv", index=False)
    print("\n===== 汇总(遵守率 / 目标)=====")
    g = df.groupby("method").agg(sat=("ok", "mean"), obj=("obj", "mean"), nt=("nt", "mean"))
    print(g.round(3).sort_values("sat", ascending=False).to_string())


if __name__ == "__main__":
    main()
