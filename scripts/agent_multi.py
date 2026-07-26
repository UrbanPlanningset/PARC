"""Multi-agent stakeholder negotiation for micro-update (KDD experiment 2).

Four stakeholder agents each advocate for ONE objective (day cooling / nighttime UHI / stormwater /
surface), proposing an intervention wish-list. A coordinator agent negotiates a single plan under the
budget, grounded by the CNN surrogate, and produces a transparent negotiation summary. This maps the
'multiple spaces' of micro-update onto explicit, interpretable multi-objective negotiation — vs the
opaque scalarised objective of RL.

  python scripts/agent_multi.py
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, glob
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from agent_prototype import deepseek, describe_tile, apply_strategy, score, _json_from
import torch
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem

SURR = ROOT / "results/ig/surrogate"

STAKEHOLDERS = [
    ("热舒适", "白天行人降温(遮阴,乔木最有效)"),
    ("夜间",   "夜间热岛释放(乔木夜间反而保温,开敞/材料更利散热)"),
    ("雨洪",   "暴雨滞蓄(透水铺装、低矮绿化、水体最有效)"),
    ("地表",   "地表过热(高反照率冷铺装、绿化最有效)"),
]


def wish(p, name, desc):
    sys_p = (f"你代表城市微更新中的『{name}』诉求方,只关心:{desc}。给定地块,提出你希望采用的改造"
             f"(类型英文名+大致数量)来最大化你的诉求,并用一句话说明理由。只输出 JSON:"
             '{"wish":{"改造英文名":数量},"reason":"一句话"}。')
    out = deepseek([{"role": "system", "content": sys_p}, {"role": "user", "content": describe_tile(p)}], max_tokens=500)
    o = _json_from(out)
    return {"name": name, "wish": o.get("wish", {}), "reason": o.get("reason", "")}


def coordinate(p, rl, wishes):
    board = "\n".join(f"  · {w['name']}方:想要 {w['wish']} —— {w['reason']}" for w in wishes)
    sys_p = ("你是城市微更新的协调规划者。四个诉求方各自提出了愿望,但预算有限、且诉求相互冲突(如乔木"
             "利白天却损夜间;材料利雨洪/地表却弱降温)。请综合各方,在预算内给出一个平衡的最终方案,"
             "并简述你如何权衡与取舍。预算由系统自动执行,专注组成与优先级。只输出 JSON:"
             '{"plan":{"改造英文名":数量},"negotiation":"你如何权衡各方的简述"}。')
    user = (describe_tile(p) + f"\n\n四方诉求:\n{board}\n\n参考:RL 优化器基线方案={rl['composition']},"
            f"综合目标={rl['objective']}。请协商出最终方案。")
    out = deepseek([{"role": "system", "content": sys_p}, {"role": "user", "content": user}], max_tokens=900)
    return _json_from(out)


def main():
    model = SurrogateCNN(); model.load_state_dict(torch.load(SURR / "surrogate_beijing.pt", map_location="cpu")); model.eval()
    pdir = ROOT / "results/ig/split_experiment/plans/beijing"
    sid = next(Path(x).name for x in sorted(glob.glob(str(pdir / "*"))) if (Path(x) / "RL_DQN/placements.json").exists())
    p = TileProblem(sid, model, device="cpu")
    rl_plan = [(int(x), int(y)) for x, y in json.load(open(pdir / sid / "RL_DQN/placements.json"))]
    rl = score(p, rl_plan)
    print(f"\n=== 地块 {sid[-30:]}  预算={p.budget:.0f} ===")
    print(f"[RL 单目标基线] 综合目标={rl['objective']} 组成={rl['composition']}\n")

    print("--- 四方诉求(各自提愿望)---")
    wishes = [wish(p, n, d) for n, d in STAKEHOLDERS]
    for w in wishes:
        print(f"  {w['name']}方: {w['wish']}  ——  {w['reason']}")

    print("\n--- 协调者博弈 ---")
    res = coordinate(p, rl, wishes)
    plan = apply_strategy(p, res.get("plan", {}))
    s = score(p, plan)
    print(f"  最终方案: {s['composition']}")
    print(f"  分项: 白天={s['day_cool']} 夜间={s['night']} 雨洪={s['storm']} 地表={s['surface']} | 综合目标={s['objective']}")
    print(f"  协商说明: {res.get('negotiation','').strip()}\n")

    print("=== 对比(分项;体现多方平衡)===")
    print(f"  {'':10}{'白天':>8}{'夜间':>9}{'雨洪':>9}{'地表':>9}{'综合':>9}")
    print(f"  {'RL 单目标':10}{rl['day_cool']:>8}{rl['night']:>9}{rl['storm']:>9}{rl['surface']:>9}{rl['objective']:>9}")
    print(f"  {'多智能体':10}{s['day_cool']:>8}{s['night']:>9}{s['storm']:>9}{s['surface']:>9}{s['objective']:>9}")
    print("\n要点:多智能体把'多空间'做成显式、可解释的多方协商;RL 是不透明的标量化单目标。")


if __name__ == "__main__":
    main()
