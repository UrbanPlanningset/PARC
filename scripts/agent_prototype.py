"""Agentic micro-update — single-agent PROOF OF CONCEPT.

An LLM agent (DeepSeek) proposes a retrofit plan for a real tile; the CNN surrogate GROUNDS it in
physics (returns the multi-criteria objective + sub-scores); the agent SELF-CRITIQUES and refines;
finally it EXPLAINS its plan in natural language. We compare the agent's plan to the trained RL
policy on the same tile. This validates the loop:  propose -> physics-ground -> critique -> explain.

  1-solweig-mlp-2-3-rl/.venv/bin/python scripts/agent_prototype.py            # auto-pick a local Beijing tile
  ... scripts/agent_prototype.py --tile <site_id>
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, re, glob, time, argparse, urllib.request
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RP = Path(sys.executable).resolve().parents[1] / "lib/python3.12/site-packages/rasterio/proj_data"
if RP.exists():
    os.environ["PROJ_DATA"] = str(RP); os.environ["PROJ_LIB"] = str(RP)
import torch
sys.path.insert(0, str(ROOT))
from src.microupdate.surrogate import SurrogateCNN
from src.microupdate.env import TileProblem
from src.microupdate.action_space import ACTIONS, ACTION_BY_NAME
from src.microupdate import baselines as B

SURR = ROOT / "results/ig/surrogate"
PLANS = ROOT / "results/ig/split_experiment/plans/beijing"


# ---------------- LLM backend ----------------
def _env(p=ROOT / ".env"):
    e = {}
    if p.exists():
        for line in open(p):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1); e[k] = v
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "AGENT_LLM_MODEL",
                "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
        if os.environ.get(key):
            e[key] = os.environ[key]
    return e
ENV = _env()


def deepseek(messages, temperature=0.4, max_tokens=1400):
    model = ENV.get("AGENT_LLM_MODEL", "deepseek-chat")
    api_key = ENV.get("LLM_API_KEY") or ENV.get("DEEPSEEK_API_KEY")
    base_url = ENV.get("LLM_BASE_URL") or ENV.get("DEEPSEEK_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "Set LLM_API_KEY and LLM_BASE_URL (or the backward-compatible "
            "DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL) before running agent scripts."
        )
    data = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    hdr = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    for attempt in range(6):                       # 网络中断(IncompleteRead/超时)重试,避免整轮崩溃
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=data, headers=hdr)
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise last


def _json_from(text):
    """Robust JSON extraction: try a ```json block, else scan for the first balanced {...}."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    s = text.find("{")
    while s != -1:
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:i + 1])
                    except Exception:
                        break
        s = text.find("{", s + 1)
    return {}


def parse_plan(text):
    obj = _json_from(text)
    return obj.get("plan", {}) if isinstance(obj, dict) else {}


# ---------------- tile <-> strategy ----------------
def describe_tile(p):
    menu = "\n".join(f"  - {a.name}(类别 {a.kind},单位成本 {a.cost},本块可落点 {int(p.action_ok[:, a.aid].sum())} 处)"
                     for a in ACTIONS)
    return (f"地块共有 {p.K} 个候选落点,改造总预算 = {p.budget:.0f}(成本累加不得超)。\n"
            f"可用改造(会自动只落在允许的位置上,按最热优先):\n{menu}\n"
            f"物理背景:乔木白天降温最强(遮阴)但夜间略保温;低矮绿化/冷铺装/透水白天降温弱,"
            f"但提供雨洪滞蓄、地表降温、夜间等共益。预算有限,需权衡。")


def apply_strategy(p, comp, budget=None):
    """把 {改造名:数量} 落成具体方案:按候选点基线 UTCI 从热到冷,放在允许且预算内的位置。
    budget=None 用地块默认预算;传入值(如约束里的更低预算)则以其为上限。"""
    B = p.budget if budget is None else float(budget)
    hot = p.baseline_utci[p.cand[:, 0], p.cand[:, 1]]
    order = np.argsort(-hot)
    used, plan, cost = set(), [], 0.0
    for name, n in comp.items():
        a = ACTION_BY_NAME.get(name)
        if a is None:
            continue
        placed = 0
        for i in order:
            if placed >= int(n):
                break
            i = int(i)
            if i in used or not p.action_ok[i, a.aid] or cost + a.cost > B:
                continue
            plan.append((i, a.aid)); used.add(i); cost += a.cost; placed += 1
    return plan


def score(p, plan):
    m = B.plan_metrics(p, plan)
    c = Counter(t for _, t in plan)
    comp = {ACTIONS[t].name: c[t] for t in sorted(c)}
    return {"objective": round(m["objective"], 4), "day_cool": round(m["mean_cool_hot"], 3),
            "night": round(m["mean_cool_night"], 4), "storm": round(m.get("storm", 0), 4),
            "surface": round(m.get("surface", 0), 4), "cost": round(m["cost"]),
            "n_actions": m["n_actions"], "composition": comp}


def fb(tag, s):
    return (f"{tag} 的物理评估:综合目标={s['objective']} | 白天降温={s['day_cool']} 夜间={s['night']} "
            f"雨洪={s['storm']} 地表={s['surface']} | 成本={s['cost']} | 实际落成={s['composition']}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="")
    a = ap.parse_args()
    model = SurrogateCNN(); model.load_state_dict(torch.load(SURR / "surrogate_beijing.pt", map_location="cpu")); model.eval()

    sid = a.tile
    if not sid:
        for d in glob.glob("data/ig/*"):
            s = Path(d).name
            if Path(d, "dsm.tif").exists() and (PLANS / s / "RL_DQN/placements.json").exists():
                sid = s; break
    p = TileProblem(sid, model, device="cpu")
    print(f"\n=== 地块 {sid[-34:]}  (K={p.K}, 预算={p.budget:.0f}) ===\n")

    rl_plan = [(int(x), int(y)) for x, y in json.load(open(PLANS / sid / "RL_DQN/placements.json"))]
    rl = score(p, rl_plan)

    SYS = ("你是城市微更新规划智能体。任务:在给定地块上选一组改造落点,最大化综合多准则目标"
           "(白天降温+夜间+雨洪+地表,已加权),受预算与街道分区约束。你会收到基于物理模拟的反馈,"
           "必须据此自我批判、改进。严格输出 JSON:"
           '{"reasoning":"一句话思路","plan":{"改造英文名":数量}}。改造名用系统给定的英文名。')
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": describe_tile(p) + "\n\n请给出方案 v1(JSON)。"}]

    hist = []
    for i, turn in enumerate(["v1", "v2"], 1):
        out = deepseek(msgs)
        comp = parse_plan(out)
        plan = apply_strategy(p, comp)
        s = score(p, plan); hist.append((turn, s))
        print(f"--- 智能体 {turn} ---\n提议组成: {comp}\n{fb(turn, s)}\n")
        msgs.append({"role": "assistant", "content": out})
        if i == 1:
            msgs.append({"role": "user", "content": fb("你的方案v1", s) +
                         "\n请指出v1的不足(哪个准则可以更好、预算是否用满、乔木/共益是否失衡),给出改进的 v2(JSON)。"})
        else:
            msgs.append({"role": "user", "content": fb("你的方案v2", s) +
                         "\n这是最终方案。请用一段自然语言解释你的最终方案:为什么这样组合改造、如何权衡各准则。"})
    explanation = deepseek(msgs, max_tokens=600)
    print("--- 智能体的自然语言解释 ---\n" + explanation.strip() + "\n")

    best = max(hist, key=lambda x: x[1]["objective"])
    print("=== 对比(综合目标越高越好)===")
    print(f"  RL(训练策略)     : {rl['objective']}   组成={rl['composition']}")
    for turn, s in hist:
        print(f"  智能体 {turn}          : {s['objective']}   组成={s['composition']}")
    print(f"\n  智能体最优({best[0]}) vs RL: {best[1]['objective']} vs {rl['objective']}  "
          f"({'智能体更高' if best[1]['objective'] > rl['objective'] else 'RL 更高'})")
    print("\n注:这是概念验证——重点是'提议→物理接地→自我批判→解释'的闭环跑通,而非智能体一定超过 RL。")


if __name__ == "__main__":
    main()
