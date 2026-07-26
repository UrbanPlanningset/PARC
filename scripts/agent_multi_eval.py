"""exp2 (full): multi-agent stakeholder negotiation across tiles/cities, quantified.

For each tile: 4 stakeholder agents (day / night / storm / surface) each propose a wish-list;
a coordinator negotiates one plan under budget. We compare the negotiated plan vs the RL
single-objective baseline on per-component scores and report:
  - comp_wins: on how many CO-BENEFIT components (night/storm/surface) negotiation beats RL;
  - day_ret:   how much day cooling is retained (neg/RL);
  - obj_ratio: negotiated objective / RL objective (honesty: usually < 1 — balance costs objective).
Negotiation texts are saved for the interpretability analysis.

Run with locked env: MICROUPDATE_NIGHT_NORM=10 MICROUPDATE_W_STORM=5 MICROUPDATE_W_SURF=5
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, glob, argparse
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from agent_prototype import apply_strategy, score
from agent_multi import wish, coordinate, STAKEHOLDERS
import torch
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem

SURR = ROOT / "results/ig/surrogate"
OUT = ROOT / "results/agentic"
CITIES = [("Beijing", "beijing"), ("Shanghai", "shanghai"), ("Seoul", "seoul"), ("New York", "newyork")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-per-city", type=int, default=2)
    ap.add_argument("--cities", default="all")
    a = ap.parse_args()
    cities = CITIES if a.cities == "all" else [c for c in CITIES if c[1] in a.cities.split(",")]
    rows, texts = [], []
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
            wishes = [wish(p, n, d) for n, d in STAKEHOLDERS]
            res = coordinate(p, rl, wishes)
            plan = apply_strategy(p, res.get("plan", {}))
            s = score(p, plan)
            wins = sum(int(s[k] > rl[k]) for k in ("night", "storm", "surface"))
            rows.append(dict(city=disp, tile=sid,
                             rl_obj=rl["objective"], neg_obj=s["objective"],
                             rl_day=rl["day_cool"], neg_day=s["day_cool"],
                             rl_night=rl["night"], neg_night=s["night"],
                             rl_storm=rl["storm"], neg_storm=s["storm"],
                             rl_surface=rl["surface"], neg_surface=s["surface"],
                             comp_wins=wins,
                             day_ret=round(s["day_cool"] / rl["day_cool"], 3) if rl["day_cool"] else None,
                             obj_ratio=round(s["objective"] / rl["objective"], 3) if rl["objective"] else None))
            texts.append(dict(city=disp, tile=sid, wishes=wishes, negotiation=res.get("negotiation", "")))
            print(f"{ck} {sid[-16:]} neg_obj={s['objective']} (RL {rl['objective']}) "
                  f"共益胜{wins}/3 day_ret={rows[-1]['day_ret']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "multi_eval.csv", index=False)
    json.dump(texts, open(OUT / "multi_texts.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n===== exp2 汇总 (n={len(df)}) =====")
    print(f"共益分项胜出(夜/雨洪/地表, 0-3): 平均 {df.comp_wins.mean():.2f}/3")
    print(f"白天降温保留率: {df.day_ret.mean():.0%}   目标比(neg/RL): {df.obj_ratio.mean():.2f}")
    print("saved multi_eval.csv + multi_texts.json")


if __name__ == "__main__":
    main()
