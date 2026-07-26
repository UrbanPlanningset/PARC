"""Quantitative agentic evaluation (KDD experiment 1).

For a sample of real tiles and a battery of natural-language planning constraints, compare:
  - RL           : the trained policy's FIXED plan (strong optimiser, cannot adapt);
  - Naive-LLM    : an LLM proposing a plan from scratch (reasons, but no learned optimiser);
  - Agent(RL-tool): an LLM that uses the RL plan as a strong starting point, then adapts.
Every plan is feasibly placed + physics-scored by the CNN surrogate. We report constraint
SATISFACTION RATE and the retained multi-criteria OBJECTIVE — the hard data for "what agentic adds".

  python scripts/agent_eval.py --tiles-per-city 3
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, glob, argparse
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from agent_prototype import deepseek, describe_tile, apply_strategy, score, _json_from
import torch
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem
from src.microupdate.action_space import ACTIONS

SURR = ROOT / "results/ig/surrogate"
OUT = ROOT / "results/agentic"; OUT.mkdir(parents=True, exist_ok=True)
CITIES = [("Beijing", "beijing"), ("Shanghai", "shanghai"), ("Seoul", "seoul"), ("New York", "newyork")]

# action ids: 0 tree_s 1 tree_m 2 tree_l 3 cool 4 greening 5 water 6 awning 7 permeable
def _c(P, t): return sum(1 for _, tt in P if tt == t)
def _cost(P): return sum(ACTIONS[t].cost for _, t in P)
def _nt(P): return (sum(1 for _, t in P if t not in (0, 1, 2)) / len(P)) if P else 0.0

# (id, 自然语言约束, 该情景预算(None=默认), 满足检查, 类别)
CONS = [
    ("no_water",  "禁止使用水体(维护成本高)。",          None, lambda P, s, rl: _c(P, 5) == 0, "type-ban"),
    ("no_cool",   "禁用高反照率冷铺装(反光刺眼投诉)。",   None, lambda P, s, rl: _c(P, 3) == 0, "type-ban"),
    ("no_large",  "地下管线密集,禁止大乔木。",            None, lambda P, s, rl: _c(P, 2) == 0, "type-ban"),
    ("trees_only", "本期仅批乔木类改造(其余审批未过)。",  None, lambda P, s, rl: len(P) > 0 and all(t in (0, 1, 2) for _, t in P), "type-limit"),
    ("budget120", "改造预算削减到 120。",                 120,  lambda P, s, rl: _cost(P) <= 120, "budget"),
    ("budget80",  "改造预算削减到 80。",                  80,   lambda P, s, rl: _cost(P) <= 80, "budget"),
    ("max30",     "施工窗口短,最多 30 处改造。",          None, lambda P, s, rl: 0 < len(P) <= 30, "count"),
    ("storm_up",  "内涝严重,雨洪指标须高于基线。",        None, lambda P, s, rl: s["storm"] > rl["storm"], "priority"),
    ("night_up",  "夜间闷热,夜间指标须高于基线。",        None, lambda P, s, rl: s["night"] > rl["night"], "priority"),
    ("surface_up", "地表过热,地表降温须高于基线。",       None, lambda P, s, rl: s["surface"] > rl["surface"], "priority"),
    ("min_nt30",  "多样性要求:非乔木改造至少占 30%。",    None, lambda P, s, rl: _nt(P) >= 0.30, "min-req"),
    ("min_perm",  "泄洪区:透水铺装至少 10 处。",          None, lambda P, s, rl: _c(P, 7) >= 10, "min-req"),
]

SYS_A = ("你是城市微更新规划智能体。系统会给你一个训练好的优化器(RL)的强基线方案作为参考:它在无约束时"
         "质量很高,但你【不必拘泥于它】——满足给定的自然语言约束是【硬性要求,必须做到】,必要时可大幅"
         "偏离甚至完全重构基线(例如约束要求少改造、要材料为主、或某准则优先时)。在【确保满足约束】的前提"
         "下,再尽量借鉴基线以保持综合多准则表现(白天降温+夜间+雨洪+地表)。预算上限系统自动执行,专注"
         "改造的类型、数量与优先级。只输出 JSON:{\"plan\":{\"改造英文名\":数量}}。")
SYS_N = ("你是城市微更新规划助手。请在满足给定自然语言约束的前提下,直接给出一组改造落点,兼顾降温与"
         "雨洪/地表/夜间共益。预算上限由系统自动执行。只输出 JSON:{\"plan\":{\"改造英文名\":数量}}。")


def _ask(system, user, p, budget, retries=2):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(retries + 1):
        out = deepseek(msgs, max_tokens=1200)
        pl = apply_strategy(p, _json_from(out).get("plan", {}), budget=budget)
        if pl:
            return pl
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": '只输出 JSON:{"plan":{"英文名":数量,...}},至少一项。'}]
    return []


def agent_adapt(p, rl, nl, budget):
    u = (describe_tile(p) + (f"\n\n【本情景预算上限】{budget:.0f}。" if budget != p.budget else "") +
         f"\n\n优化器(RL)强基线:组成={rl['composition']},综合目标={rl['objective']}。"
         f"\n【约束】{nl}\n请在该约束下调整方案。")
    return _ask(SYS_A, u, p, budget)


def naive(p, nl, budget):
    u = (describe_tile(p) + (f"\n\n【本情景预算上限】{budget:.0f}。" if budget != p.budget else "") +
         f"\n【约束】{nl}\n请给出满足约束的方案。")
    return _ask(SYS_N, u, p, budget)


def refine(p, nl, budget, plan1, s1):
    """物理接地消融:把上一版方案的 CNN 代理评分反馈给智能体,让它据此精修一版。取综合目标更高者。"""
    u = (describe_tile(p) + (f"\n\n【本情景预算上限】{budget:.0f}。" if budget != p.budget else "") +
         f"\n【约束】{nl}\n你上一版方案={s1['composition']},其物理评估:综合目标={s1['objective']}"
         f"(白天={s1['day_cool']} 夜间={s1['night']} 雨洪={s1['storm']} 地表={s1['surface']})。"
         f"请在【仍满足约束】的前提下,利用此物理反馈改进方案以提升综合表现。")
    plan2 = _ask(SYS_A, u, p, budget)
    return plan2 if (plan2 and score(p, plan2)["objective"] >= s1["objective"]) else plan1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-per-city", type=int, default=3)
    ap.add_argument("--cities", default="all")
    ap.add_argument("--cons", default="", help="逗号分隔的约束 id 子集(默认全部;用于小样验证)")
    ap.add_argument("--out-tag", default="", help="输出后缀 eval_<tag>.csv(并行时避免覆盖)")
    a = ap.parse_args()
    cities = CITIES if a.cities == "all" else [c for c in CITIES if c[1] in a.cities.split(",")]
    cons = CONS if not a.cons else [c for c in CONS if c[0] in a.cons.split(",")]

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
                print(f"skip {sid[:30]}: {e}", flush=True); continue
            rl_plan = [(int(x), int(y)) for x, y in json.load(open(pdir / sid / "RL_DQN/placements.json"))]
            rl = score(p, rl_plan)
            for cid, nl, bud, check, cat in cons:
                eff = bud if bud is not None else p.budget
                rl_ok = check(rl_plan, rl, rl)
                nvp = naive(p, nl, eff); nvs = score(p, nvp); nv_ok = check(nvp, nvs, rl)
                agp = agent_adapt(p, rl, nl, eff); ags = score(p, agp); ag_ok = check(agp, ags, rl)
                grp = refine(p, nl, eff, agp, ags); grs = score(p, grp); gr_ok = check(grp, grs, rl)  # +物理接地
                rows.append(dict(city=disp, tile=sid, cid=cid, cat=cat,
                                 rl_ok=rl_ok, nv_ok=nv_ok, ag_ok=ag_ok, gr_ok=gr_ok,
                                 rl_obj=rl["objective"], nv_obj=nvs["objective"], ag_obj=ags["objective"], gr_obj=grs["objective"],
                                 rl_nt=_nt(rl_plan), nv_nt=_nt(nvp), ag_nt=_nt(agp), gr_nt=_nt(grp)))
                print(f"{disp[:3]} {sid[-18:]:18} {cid:10} RL={'Y' if rl_ok else '.'} "
                      f"NV={'Y' if nv_ok else '.'} AG={'Y' if ag_ok else '.'} GR={'Y' if gr_ok else '.'}", flush=True)

    fn = f"eval_{a.out_tag}.csv" if a.out_tag else "eval.csv"
    df = pd.DataFrame(rows); df.to_csv(OUT / fn, index=False)
    print(f"\n===== 汇总(n={len(df)} 个 地块×约束)=====")
    print("约束遵守率:  RL={:.0%}  Naive-LLM={:.0%}  Agent={:.0%}  Agent+接地={:.0%}".format(
        df.rl_ok.mean(), df.nv_ok.mean(), df.ag_ok.mean(), df.gr_ok.mean()))
    print("平均综合目标: RL={:.3f}  Naive-LLM={:.3f}  Agent={:.3f}  Agent+接地={:.3f}".format(
        df.rl_obj.mean(), df.nv_obj.mean(), df.ag_obj.mean(), df.gr_obj.mean()))
    print("非乔木占比(多样性): RL={:.0%}  Naive-LLM={:.0%}  Agent={:.0%}  Agent+接地={:.0%}".format(
        df.rl_nt.mean(), df.nv_nt.mean(), df.ag_nt.mean(), df.gr_nt.mean()))
    print("\n按类别·遵守率(RL / Naive / Agent / Agent+接地):")
    g = df.groupby("cat")[["rl_ok", "nv_ok", "ag_ok", "gr_ok"]].mean().round(2)
    print(g.to_string())
    print(f"\nsaved {OUT/'eval.csv'}")


if __name__ == "__main__":
    main()
