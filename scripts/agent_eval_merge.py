"""Merge the per-city agent_eval CSVs (tag prefix given by --prefix) and print the aggregate table
(satisfaction / objective / diversity) across RL / Naive / Agent / Agent+grounding."""
import glob, sys
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results/agentic"
prefix = sys.argv[1] if len(sys.argv) > 1 else "v3"
fs = sorted(glob.glob(str(OUT / f"eval_{prefix}_*.csv")))
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
df.to_csv(OUT / f"eval_{prefix}.csv", index=False)

has_gr = "gr_ok" in df.columns
print(f"===== {prefix} 合并 (n={len(df)} 个 地块×约束, {df.city.nunique()} 城) =====")
cols = [("RL(固定)", "rl"), ("Naive-LLM", "nv"), ("Agent", "ag")] + ([("Agent+接地", "gr")] if has_gr else [])
print("约束遵守率:  " + "  ".join(f"{name}={df[c+'_ok'].mean():.0%}" for name, c in cols))
print("平均综合目标: " + "  ".join(f"{name}={df[c+'_obj'].mean():.3f}" for name, c in cols))
print("非乔木占比:  " + "  ".join(f"{name}={df[c+'_nt'].mean():.0%}" for name, c in cols))
okcols = [c + "_ok" for _, c in cols]
print("\n按类别·遵守率(" + " / ".join(n for n, _ in cols) + "):")
print(df.groupby("cat")[okcols].mean().round(2).to_string())
