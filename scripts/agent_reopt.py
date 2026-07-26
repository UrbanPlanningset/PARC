"""Constrained re-optimization baseline (RQ6): hand-encode each of the 12 constraints
into the optimizer and RE-RUN the search per (tile, constraint) case.

Search space = composition vector (count per action type), placed hotspot-first via the
same deterministic fill as the executor -- the same decision space as PARC's planner,
so the comparison isolates WHO chooses the composition, not how it is placed.

Methods: greedy_c, random_c, sa_c, ga_c (constraints enforced by construction/rejection).

  python scripts/agent_reopt.py --cities beijing --tiles-per-city 8 --methods greedy_c,random_c,sa_c,ga_c
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, glob, argparse, time
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
NTYPES = len(ACTIONS)
TREES = (0, 1, 2)

# ---- hand-encoded constraint translation (this is the human engineering PARC replaces) ----
def encode(cid, p):
    allowed = set(range(NTYPES))
    maxn = 10**6
    forced = {}                       # type -> minimum count
    if cid == "no_water":  allowed -= {5}
    if cid == "no_cool":   allowed -= {3}
    if cid == "no_large":  allowed -= {2}
    if cid == "trees_only": allowed &= set(TREES)
    if cid == "max30": maxn = 30
    if cid == "min_perm": forced[7] = 10
    if cid == "storm_up":  forced[7] = max(forced.get(7, 0), 6)
    if cid == "night_up":  forced[4] = 6
    if cid == "surface_up": forced[3] = 6
    return allowed, maxn, forced

def place(p, comp, eff, maxn):
    """composition dict {type: n} -> hotspot-first plan within budget/count."""
    tau = p.baseline_utci[p.cand[:, 0], p.cand[:, 1]]
    order = np.argsort(-tau)
    P, used, c = [], set(), 0.0
    for t, n in sorted(comp.items(), key=lambda kv: -ACTIONS[kv[0]].cost):
        a = ACTIONS[t]; placed = 0
        for i in order:
            i = int(i)
            if placed >= n or len(P) >= maxn: break
            if i in used or not p.action_ok[i, a.aid] or c + a.cost > eff: continue
            P.append((i, a.aid)); used.add(i); c += a.cost; placed += 1
    return P

def rand_comp(rng, allowed, eff, maxn, forced):
    comp = dict(forced)
    budget_left = eff - sum(ACTIONS[t].cost * n for t, n in comp.items())
    types = sorted(allowed)
    for _ in range(200):
        if budget_left <= 0 or sum(comp.values()) >= maxn: break
        t = int(rng.choice(types))
        if ACTIONS[t].cost <= budget_left:
            comp[t] = comp.get(t, 0) + 1
            budget_left -= ACTIONS[t].cost
        elif all(ACTIONS[x].cost > budget_left for x in types):
            break
    return comp

def fitness(p, comp, eff, maxn, check, rl):
    P = place(p, comp, eff, maxn)
    if not P: return -1e9, None, None
    s = score(p, P)
    ok = check(P, s, rl)
    return (s["objective"] if ok else -1e9), s, P     # feasible-only search: violators rejected

def neighbors(rng, comp, allowed, forced):
    q = dict(comp); types = sorted(allowed)
    op = rng.integers(3)
    t = int(rng.choice(types))
    if op == 0: q[t] = q.get(t, 0) + 1
    elif op == 1 and q.get(t, 0) > forced.get(t, 0): q[t] = q[t] - 1
    else:
        u = int(rng.choice(types))
        if q.get(t, 0) > forced.get(t, 0): q[t] = q.get(t, 0) - 1; q[u] = q.get(u, 0) + 1
    return {k: v for k, v in q.items() if v > 0}

def run_method(method, p, cid, eff, rl, check, seed=0):
    allowed, maxn, forced = encode(cid, p)
    rng = np.random.default_rng(seed)
    if method == "greedy_c":
        comp = dict(forced)
        best = fitness(p, comp, eff, maxn, check, rl)
        for _ in range(60):                                   # steepest-ascent add-one
            cands = [{**comp, t: comp.get(t, 0) + 1} for t in sorted(allowed)]
            scored = [(fitness(p, c, eff, maxn, check, rl), c) for c in cands]
            (f, s, P), c = max(scored, key=lambda x: x[0][0])
            if f <= best[0]: break
            best, comp = (f, s, P), c
        return best
    if method == "random_c":
        best = (-1e9, None, None)
        for _ in range(300):
            best = max(best, fitness(p, rand_comp(rng, allowed, eff, maxn, forced), eff, maxn, check, rl), key=lambda x: x[0])
        return best
    if method == "sa_c":
        comp = rand_comp(rng, allowed, eff, maxn, forced)
        cur = fitness(p, comp, eff, maxn, check, rl); best = cur; T = 0.05
        for k in range(500):
            q = neighbors(rng, comp, allowed, forced)
            f = fitness(p, q, eff, maxn, check, rl)
            if f[0] > cur[0] or rng.random() < np.exp(min(0, (f[0] - cur[0]) / max(T, 1e-9))):
                comp, cur = q, f
                if f[0] > best[0]: best = f
            T *= 0.99
        return best
    if method == "ga_c":
        pop = [rand_comp(rng, allowed, eff, maxn, forced) for _ in range(20)]
        evals = [fitness(p, c, eff, maxn, check, rl) for c in pop]
        best = max(evals, key=lambda x: x[0])
        for g in range(25):
            idx = np.argsort([-e[0] for e in evals])[:10]
            elite = [pop[i] for i in idx]
            children = []
            while len(children) < 20:
                a, b = rng.choice(len(elite), 2, replace=False)
                child = {t: (elite[a].get(t, 0) if rng.random() < .5 else elite[b].get(t, 0)) for t in set(elite[a]) | set(elite[b])}
                child.update(forced)
                children.append(neighbors(rng, child, allowed, forced))
            pop = children
            evals = [fitness(p, c, eff, maxn, check, rl) for c in pop]
            cand = max(evals, key=lambda x: x[0])
            if cand[0] > best[0]: best = cand
        return best
    raise ValueError(method)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="all")
    ap.add_argument("--tiles-per-city", type=int, default=8)
    ap.add_argument("--methods", default="greedy_c,random_c,sa_c,ga_c")
    ap.add_argument("--constraints", default="all")
    ap.add_argument("--tile-filter", default="")
    a = ap.parse_args()
    cities = CITIES if a.cities == "all" else [c for c in CITIES if c[1] in a.cities.split(",")]
    methods = a.methods.split(",")
    cons = CONS if a.constraints == "all" else [c for c in CONS if c[0] in a.constraints.split(",")]
    rows = []
    t0 = time.time()
    for disp, ck in cities:
        model = SurrogateCNN(); model.load_state_dict(torch.load(SURR / f"surrogate_{ck}.pt", map_location="cpu")); model.eval()
        pdir = ROOT / f"results/ig/split_experiment/plans/{ck}"
        sids = [Path(x).name for x in sorted(glob.glob(str(pdir / "*")))
                if (Path(x) / "RL_DQN/placements.json").exists()]
        if a.tile_filter: sids = [s2 for s2 in sids if a.tile_filter in s2]
        sids = sids[:a.tiles_per_city]
        for sid in sids:
            try:
                p = TileProblem(sid, model, device="cpu")
            except Exception as e:
                print("skip", sid[:30], e, flush=True); continue
            rl_plan = [(int(x), int(y)) for x, y in json.load(open(pdir / sid / "RL_DQN/placements.json"))]
            rl = score(p, rl_plan)
            for cid, nl, bud, check, cat in cons:
                eff = bud if bud is not None else p.budget
                for m in methods:
                    f, s, P = run_method(m, p, cid, eff, rl, check)
                    ok = bool(P) and f > -1e8
                    rows.append(dict(city=disp, tile=sid, cid=cid, cat=cat, method=m,
                                     ok=ok and check(P, s, rl), obj=(s["objective"] if s else 0.0),
                                     nt=_nt(P) if P else 0.0))
                print(f"{ck} {sid[-14:]} {cid} done {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    tag = "_".join(c[1] for c in cities)
    df.to_csv(OUT / f"reopt_{tag}.csv", index=False)
    print("\n===== constrained re-optimization summary =====")
    print(df.groupby("method").agg(sat=("ok", "mean"), obj=("obj", "mean")).round(3).to_string())

if __name__ == "__main__":
    main()
