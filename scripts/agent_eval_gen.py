"""RQ5: OPEN-VOCABULARY generalization — 12 constraints OUTSIDE the training battery
(4 paraphrased + 4 compositional + 4 novel), where the rule-based repair must rely on a
keyword ROUTER (a fair, strong one: parses numbers, chains matched operators), while PARC
just reads the language. Also logs tokens/latency per case for the cost analysis (RQ6).

  env-locked run:  MICROUPDATE_NIGHT_NORM=10 MICROUPDATE_W_STORM=5 MICROUPDATE_W_SURF=5 \
    python scripts/agent_eval_gen.py --cities beijing --tiles-per-city 4
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, glob, re, time, argparse
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import agent_prototype as AP
from agent_prototype import score
from agent_eval import agent_adapt, refine, naive, _nt
from agent_baselines2 import _fill, _drop_coolest, _tau, _cost, TREES
from agent_coder import run_case as coder_case
import torch
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem

SURR = ROOT / "results/ig/surrogate"
OUT = ROOT / "results/agentic"; OUT.mkdir(parents=True, exist_ok=True)
CITIES = [("Beijing", "beijing"), ("Shanghai", "shanghai"), ("Seoul", "seoul"), ("New York", "newyork")]


def _cnt(P, t): return sum(1 for _, tt in P if tt == t)
def _share(P, t): return _cnt(P, t) / len(P) if P else 0.0


# ---- generalization cases: (id, group, NL, budget(None=default), checker) ----
GEN = [
    # A. paraphrases of battery constraints (same semantics, different wording)
    ("p_water",  "paraphrase", "居民担心蚊虫和维护费用,请避免任何新增的水面景观。", None,
     lambda P, s, rl: _cnt(P, 5) == 0),
    ("p_budget", "paraphrase", "经费吃紧,这一期的改造投入请压在 120 以内。", 120,
     lambda P, s, rl: _cost(P) <= 120),
    ("p_storm",  "paraphrase", "这片一下大雨就积水,新方案要比现在的更能吸纳雨水。", None,
     lambda P, s, rl: s["storm"] > rl["storm"]),
    ("p_trees",  "paraphrase", "手续只批了乔木类绿化,其余改造这一期都先别上。", None,
     lambda P, s, rl: len(P) > 0 and all(t in TREES for _, t in P)),
    # B. compositional (two requirements at once)
    ("c_water_budget", "compositional", "禁止使用水体,并且预算削减到 120。", 120,
     lambda P, s, rl: _cnt(P, 5) == 0 and _cost(P) <= 120),
    ("c_trees_max30",  "compositional", "只批乔木类改造,并且最多 30 处。", None,
     lambda P, s, rl: 0 < len(P) <= 30 and all(t in TREES for _, t in P)),
    ("c_nolarge_storm", "compositional", "禁用大乔木,同时雨洪滞蓄要高于基线方案。", None,
     lambda P, s, rl: _cnt(P, 2) == 0 and s["storm"] > rl["storm"]),
    ("c_nocool_nt30",  "compositional", "禁用冷铺装,且非乔木改造至少要占三成。", None,
     lambda P, s, rl: _cnt(P, 3) == 0 and _nt(P) >= 0.30),
    # C. novel constraint types (no operator exists in the battery)
    ("n_max5large", "novel", "地下管线复杂,大乔木最多只能种 5 棵。", None,
     lambda P, s, rl: _cnt(P, 2) <= 5),
    ("n_green20",   "novel", "低矮绿化的占比至少要有两成。", None,
     lambda P, s, rl: _share(P, 4) >= 0.20),
    ("n_notrees",   "novel", "本期乔木指标已经用完了,请全部采用非乔木措施。", None,
     lambda P, s, rl: len(P) > 0 and all(t not in TREES for _, t in P)),
    ("n_budget100", "novel", "这一片区只剩 100 的额度可用。", 100,
     lambda P, s, rl: _cost(P) <= 100),
]


# ---- fair keyword router for the rule system: parses numbers, CHAINS all matched ops ----
def rule_router(p, rl_plan, rl, nl, default_budget):
    P = list(rl_plan); tau = _tau(p)
    eff = default_budget
    m = re.search(r"(预算|投入|经费|额度)[^0-9]{0,6}(\d+)", nl)
    ops = []
    if m: eff = float(m.group(2)); ops.append("budget")
    if re.search(r"水体|水面|水景", nl): ops.append("no_water")
    if re.search(r"冷铺装|反照率", nl): ops.append("no_cool")
    if re.search(r"大乔木|大树", nl) and re.search(r"禁|不|别|避免", nl): ops.append("no_large")
    if re.search(r"只批|仅批|只.{0,4}乔木|乔木类", nl): ops.append("trees_only")
    if re.search(r"雨水|雨洪|积水|内涝", nl): ops.append("storm_up")
    mc = re.search(r"最多[^0-9]{0,4}(\d+)[^0-9]{0,3}(处|个)", nl)
    if mc: ops.append(("cap", int(mc.group(1))))
    if re.search(r"非乔木", nl) and re.search(r"三成|30", nl): ops.append("min_nt30")
    for op in ops:                                     # chain all matched operators
        if op == "budget":
            P = _drop_coolest(p, P, lambda t: t in TREES, eff)
        elif op == "no_water":
            P = [(l, t) for l, t in P if t != 5]; P = _fill(p, P, eff, 0)
        elif op == "no_cool":
            P = [(l, t) for l, t in P if t != 3]; P = _fill(p, P, eff, 0)
        elif op == "no_large":
            P = [(l, t) for l, t in P if t != 2]; P = _fill(p, P, eff, 0)
        elif op == "trees_only":
            P = [(l, t) for l, t in P if t in TREES]; P = _fill(p, P, eff, 0)
        elif op == "storm_up":
            for k in range(1, 13):
                trees_sorted = [x for x in sorted(P, key=lambda y: tau[y[0]]) if x[1] in TREES]
                Q = [x for x in P if x not in trees_sorted[:3 * k]]
                Q = _fill(p, Q, eff, 7)
                if score(p, Q)["storm"] > rl["storm"]:
                    P = Q; break
            else:
                P = Q
        elif op == "min_nt30":
            for k in range(1, 13):
                trees_sorted = [x for x in sorted(P, key=lambda y: tau[y[0]]) if x[1] in TREES]
                Q = [x for x in P if x not in trees_sorted[:3 * k]]
                Q = _fill(p, Q, eff, 7)
                if _nt(Q) >= 0.30:
                    P = Q; break
            else:
                P = Q
        elif isinstance(op, tuple) and op[0] == "cap":
            P = sorted(P, key=lambda x: -tau[x[0]])[:op[1]]
    return P, bool(ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="all")
    ap.add_argument("--tiles-per-city", type=int, default=4)
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
            for cid, grp, nl, bud, check in GEN:
                eff = bud if bud is not None else p.budget
                R, matched = rule_router(p, rl_plan, rl, nl, eff)
                sr = score(p, R); r_ok = check(R, sr, rl)
                t0 = time.time()
                nvp = naive(p, nl, eff); nvs = score(p, nvp); nv_ok = check(nvp, nvs, rl)
                t1 = time.time()
                cdp, cd_exec = coder_case(p, rl_plan, nl, eff)
                cds = score(p, cdp) if cdp else {"objective": 0.0, "storm": 0, "night": 0, "surface": 0}
                cd_ok = bool(cdp) and check(cdp, cds, rl)
                t1c = time.time()
                agp = agent_adapt(p, rl, nl, eff)
                agp = refine(p, nl, eff, agp, score(p, agp))
                ags = score(p, agp); ag_ok = check(agp, ags, rl)
                t2 = time.time()
                rows.append(dict(city=disp, tile=sid, cid=cid, grp=grp,
                                 rule_ok=r_ok, rule_obj=sr["objective"], rule_matched=matched,
                                 nv_ok=nv_ok, nv_obj=nvs["objective"], nv_sec=round(t1 - t0, 1),
                                 cd_ok=cd_ok, cd_obj=cds["objective"], cd_exec=cd_exec, cd_sec=round(t1c - t1, 1),
                                 ag_ok=ag_ok, ag_obj=ags["objective"], ag_sec=round(t2 - t1c, 1)))
                print(f"{ck} {sid[-14:]} {cid:16} rule={'Y' if r_ok else '.'} nv={'Y' if nv_ok else '.'} "
                      f"cd={'Y' if cd_ok else '.'} ag={'Y' if ag_ok else '.'}", flush=True)
    df = pd.DataFrame(rows)
    tag = "_".join(c[1] for c in cities)
    df.to_csv(OUT / f"gen_{tag}.csv", index=False)
    print("\n===== RQ5 汇总 =====")
    print("总遵守率: rule={:.0%} naive={:.0%} coder={:.0%} PARC={:.0%}".format(
        df.rule_ok.mean(), df.nv_ok.mean(), df.cd_ok.mean(), df.ag_ok.mean()))
    print(df.groupby("grp")[["rule_ok", "nv_ok", "cd_ok", "ag_ok"]].mean().round(2).to_string())
    print("目标: rule={:.3f} naive={:.3f} coder={:.3f} PARC={:.3f}".format(
        df.rule_obj.mean(), df.nv_obj.mean(), df.cd_obj.mean(), df.ag_obj.mean()))
    print("coder 代码可执行率: {:.0%}".format(df.cd_exec.mean()))
    print("时延(s/方案): naive={:.1f} coder={:.1f} PARC={:.1f}".format(df.nv_sec.mean(), df.cd_sec.mean(), df.ag_sec.mean()))


if __name__ == "__main__":
    main()
