"""exp3 (generation): build RL / Agent / Naive (+ random-placement executor ablation) plans for a
tile x constraint subset, save SOLWEIG-ready layouts + a manifest with surrogate scores.

Constraints chosen so truth adds real information:
  no_water  -> near-unconstrained: measures plan QUALITY gap on truth;
  budget120 -> quality under a budget cut, on truth;
  storm_up  -> priority constraint whose satisfaction is itself truth-verifiable.

MUST run with the paper's locked objective env:
  MICROUPDATE_NIGHT_NORM=10 MICROUPDATE_W_STORM=5 MICROUPDATE_W_SURF=5
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, glob, argparse
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from agent_prototype import apply_strategy, score
from agent_eval import agent_adapt, naive, refine, CONS
import torch
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem
from src.microupdate.action_space import ACTIONS, ACTION_BY_NAME
from scripts.ig_percity_run import save_layout

SURR = ROOT / "results/ig/surrogate"
OUTP = ROOT / "results/agentic/truth_plans"
CITIES = [("Beijing", "beijing"), ("Shanghai", "shanghai"), ("Seoul", "seoul"), ("New York", "newyork")]
SUB = ["no_water", "budget120", "storm_up"]


def place_random(p, comp, budget, rng):
    """Executor ablation: SAME composition, random feasible placement (no hotspot targeting)."""
    B = float(budget)
    order = rng.permutation(p.K)
    used, plan, cost = set(), [], 0.0
    for name, n in comp.items():
        a = ACTION_BY_NAME.get(name)
        if a is None:
            continue
        placed = 0
        for i in order:
            i = int(i)
            if placed >= int(n):
                break
            if i in used or not p.action_ok[i, a.aid] or cost + a.cost > B:
                continue
            plan.append((i, a.aid)); used.add(i); cost += a.cost; placed += 1
    return plan


def comp_of(plan):
    from collections import Counter
    c = Counter(t for _, t in plan)
    return {ACTIONS[t].name: int(c[t]) for t in sorted(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="all")
    ap.add_argument("--tiles-per-city", type=int, default=3)
    a = ap.parse_args()
    cities = CITIES if a.cities == "all" else [c for c in CITIES if c[1] in a.cities.split(",")]
    cons = [c for c in CONS if c[0] in SUB]
    rng = np.random.default_rng(0)
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
            variants = {"rl__base": rl_plan}
            rl = score(p, rl_plan)
            for cid, nl, bud, check, cat in cons:
                eff = bud if bud is not None else p.budget
                agp = agent_adapt(p, rl, nl, eff)
                agp = refine(p, nl, eff, agp, score(p, agp))       # grounded agent (main variant)
                variants[f"agent__{cid}"] = agp
                variants[f"naive__{cid}"] = naive(p, nl, eff)
                if cid == "storm_up" and agp:                       # executor ablation on the hardest case
                    variants[f"agentrand__{cid}"] = place_random(p, comp_of(agp), eff, rng)
            for vid, plan in variants.items():
                if not plan:
                    print(f"!! empty plan {ck} {sid[-16:]} {vid}", flush=True); continue
                s = score(p, plan)
                save_layout(p, plan, OUTP / ck / sid / vid)
                rows.append(dict(city=disp, ck=ck, tile=sid, variant=vid,
                                 sur_obj=s["objective"], sur_day=s["day_cool"], sur_night=s["night"],
                                 sur_storm=s["storm"], sur_surface=s["surface"], cost=s["cost"], n=s["n_actions"]))
                print(f"{ck} {sid[-16:]} {vid:22} sur_obj={s['objective']}", flush=True)
    tag = "_".join(c[1] for c in cities)
    pd.DataFrame(rows).to_csv(ROOT / f"results/agentic/truth_manifest_{tag}.csv", index=False)
    print("saved manifest", flush=True)


if __name__ == "__main__":
    main()
