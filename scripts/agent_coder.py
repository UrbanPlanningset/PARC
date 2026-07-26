"""LLM-Coder baseline (code-as-policies): the LLM writes a Python program that
translates the constraint into composition logic over shared placement primitives.
No surrogate/simulator access -- isolates code-generation from physical grounding.

  MICROUPDATE_NIGHT_NORM=10 MICROUPDATE_W_STORM=5 MICROUPDATE_W_SURF=5 \
  LLM_API_KEY=... python scripts/agent_coder.py --cities all --tiles-per-city 8 --runs 3
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, glob, argparse, time, traceback
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from agent_prototype import deepseek, score, _json_from
from agent_eval import CONS, _nt
import torch
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem
from src.microupdate.action_space import ACTIONS

SURR = ROOT / "results/ig/surrogate"
OUT = ROOT / "results/agentic"; OUT.mkdir(parents=True, exist_ok=True)
CITIES = [("Beijing", "beijing"), ("Shanghai", "shanghai"), ("Seoul", "seoul"), ("New York", "newyork")]

CATALOG = "\n".join(f"  aid={a.aid} name={a.name} cost={a.cost}" for a in ACTIONS)

SYS = (
"You write Python to plan urban micro-renewal interventions under a natural-language constraint.\n"
"You are given these primitives (already defined, do not redefine):\n"
"  ACTIONS: list of interventions, each with .aid, .name, .cost;\n" + CATALOG + "\n"
"  BUDGET: float, the total cost cap (hard);\n"
"  BASELINE: dict {aid: count}, a strong unconstrained reference composition;\n"
"  place(composition: dict[int,int]) -> plan: deterministic hotspot-first placement of a\n"
"    composition {aid: count} respecting siting rules and BUDGET; returns the placed plan.\n"
"Write a COMPLETE Python function:  def solve():\n"
"  It must translate the constraint into composition logic (filter banned types, respect\n"
"  the budget and any counts or shares the text demands), build a composition dict,\n"
"  and return place(composition).\n"
"You have NO physics feedback: rely on the constraint text and the baseline only.\n"
"Return ONLY a Python code block, no prose.")

def extract_code(txt):
    if "```" in txt:
        seg = txt.split("```")[1]
        return seg[6:] if seg.startswith("python") else seg
    return txt

def run_case(p, rl_plan, nl, eff, seed_note=""):
    baseline = {}
    for _, t in rl_plan: baseline[t] = baseline.get(t, 0) + 1
    def place(comp):
        tau = p.baseline_utci[p.cand[:, 0], p.cand[:, 1]]
        order = np.argsort(-tau); P, used, c = [], set(), 0.0
        for t, n in sorted(comp.items(), key=lambda kv: -ACTIONS[kv[0]].cost):
            a = ACTIONS[int(t)]; placed = 0
            for i in order:
                i = int(i)
                if placed >= int(n): break
                if i in used or not p.action_ok[i, a.aid] or c + a.cost > eff: continue
                P.append((i, a.aid)); used.add(i); c += a.cost; placed += 1
        return P
    user = f"Constraint: {nl}\nBUDGET = {eff}\nBASELINE = {baseline}\nWrite solve() now.{seed_note}"
    txt = deepseek([{"role": "system", "content": SYS}, {"role": "user", "content": user}])
    code = extract_code(txt)
    env = {"ACTIONS": ACTIONS, "BUDGET": eff, "BASELINE": baseline, "place": place, "np": np}
    try:
        exec(compile(code, "<llm>", "exec"), env)
        plan = env["solve"]()
        assert isinstance(plan, list) and all(len(x) == 2 for x in plan)
        return plan, True
    except Exception:
        return [], False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="all"); ap.add_argument("--tiles-per-city", type=int, default=8)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    cities = CITIES if a.cities == "all" else [c for c in CITIES if c[1] in a.cities.split(",")]
    rows = []
    for run in range(a.runs):
        for disp, ck in cities:
            model = SurrogateCNN(); model.load_state_dict(torch.load(SURR / f"surrogate_{ck}.pt", map_location="cpu")); model.eval()
            pdir = ROOT / f"results/ig/split_experiment/plans/{ck}"
            sids = [Path(x).name for x in sorted(glob.glob(str(pdir / "*")))
                    if (Path(x) / "RL_DQN/placements.json").exists()][:a.tiles_per_city]
            for sid in sids:
                try: p = TileProblem(sid, model, device="cpu")
                except Exception as e: print("skip", sid[:24], e, flush=True); continue
                rl_plan = [(int(x), int(y)) for x, y in json.load(open(pdir / sid / "RL_DQN/placements.json"))]
                rl = score(p, rl_plan)
                for cid, nl, bud, check, cat in CONS:
                    eff = bud if bud is not None else p.budget
                    plan, ok_exec = run_case(p, rl_plan, nl, eff, seed_note=f" (attempt {run+1})")
                    s = score(p, plan) if plan else {"objective": 0.0, "storm": 0, "night": 0, "surface": 0}
                    rows.append(dict(run=run, city=disp, tile=sid, cid=cid, cat=cat,
                                     ok=bool(plan) and check(plan, s, rl), obj=s["objective"],
                                     nt=_nt(plan) if plan else 0.0, exec_ok=ok_exec))
                print(f"run{run} {ck} {sid[-14:]} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "coder_eval.csv", index=False)
    print(df.groupby("city").agg(sat=("ok", "mean"), obj=("obj", "mean"), exec_ok=("exec_ok", "mean")).round(3))

if __name__ == "__main__":
    main()
