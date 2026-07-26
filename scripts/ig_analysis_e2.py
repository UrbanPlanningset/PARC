"""E2 problem-oriented analyses (微更新研究方案说明 §7.2) from the pilot results.

Reads the surrogate comparison, the SOLWEIG truth backtest, the scene dataset and the
saved optimal plans; emits the E2 tables + figures the plan calls for:
  1 action-type marginal value (which lever cools, by city form)
  2 optimal-plan anatomy (what mix RL chooses, by city form)
  3 hotspot targeting & equity (>38 person-hour reduction, truth)
  4 spatial pattern (cluster score of the RL plan)
  5 method comparison on TRUTH (the E1 headline)

.venv/bin/python scripts/ig_analysis_e2.py
"""
from __future__ import annotations

from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RASTERIO_PROJ = Path(sys.executable).resolve().parents[1] / "lib/python3.12/site-packages/rasterio/proj_data"
if RASTERIO_PROJ.exists():
    os.environ["PROJ_DATA"] = str(RASTERIO_PROJ); os.environ["PROJ_LIB"] = str(RASTERIO_PROJ)
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT))
from src.microupdate.action_space import ACTIONS, LC_BUILDING

IG = ROOT / "data/ig"
RL = ROOT / "results/ig/rl"
BT = ROOT / "results/ig/backtest"
OUT = ROOT / "results/ig/analysis"


def cluster_score(modified: np.ndarray) -> float:
    """Fraction of modified cells whose 4-neighbours are also modified (clustering)."""
    m = modified > 0
    if m.sum() == 0:
        return 0.0
    nb = np.zeros_like(m, float)
    nb[1:] += m[:-1]; nb[:-1] += m[1:]; nb[:, 1:] += m[:, :-1]; nb[:, :-1] += m[:, 1:]
    return float((nb[m] / 4.0).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene = pd.read_csv(IG / "scene_dataset.csv")

    # --- 1 action-type marginal value -------------------------------------
    hi = scene[scene.scenario_id.str.match(r"single_.*_hi")].copy()
    hi["action"] = hi.scenario_id.str.replace("single_", "").str.replace("_hi", "")
    amv = hi.pivot_table(index="action", columns="city", values="d_utci_hot")
    amv["mean"] = amv.mean(axis=1)
    amv = amv.sort_values("mean", ascending=False)
    amv.round(2).to_csv(OUT / "e2_action_marginal_value.csv")
    print("[1] action-type marginal value (hotspot ΔUTCI):\n", amv.round(2).to_string(), "\n")

    # --- 5 method comparison on truth (E1 headline) -----------------------
    if (BT / "truth_comparison.csv").exists():
        truth = pd.read_csv(BT / "truth_comparison.csv")
        piv = truth.pivot_table(index="method", columns="city", values="true_d_utci_hot")
        piv["mean"] = piv.mean(axis=1)
        piv = piv.sort_values("mean", ascending=False)
        piv.round(3).to_csv(OUT / "e1_truth_method_comparison.csv")
        print("[5] SOLWEIG-truth hotspot ΔUTCI by method:\n", piv.round(3).to_string(), "\n")
        # surrogate fidelity scatter
        sub = truth[truth.pred_d_utci_hot.notna() & (truth.method != "no_action")]
        if len(sub):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(sub.pred_d_utci_hot, sub.true_d_utci_hot, c="tab:blue", alpha=0.7)
            lim = max(sub.pred_d_utci_hot.max(), sub.true_d_utci_hot.max()) * 1.1
            ax.plot([0, lim], [0, lim], "k--", lw=1)
            ax.set_xlabel("surrogate-predicted hotspot ΔUTCI"); ax.set_ylabel("SOLWEIG-true hotspot ΔUTCI")
            ax.set_title("Surrogate fidelity (two-stage validation)")
            fig.tight_layout(); fig.savefig(OUT / "surrogate_fidelity.png", dpi=130); plt.close(fig)

    # --- 2 & 4 optimal-plan anatomy + spatial pattern (RL) ----------------
    rows = []
    if RL.exists():
        for sdir in sorted(RL.iterdir()):
            rlp = sdir / "RL_DQN" / "placements.json"
            if not rlp.exists():
                continue
            placements = json.loads(rlp.read_text())
            counts = {a.name: 0 for a in ACTIONS}
            for li, ti in placements:
                counts[ACTIONS[ti].name] += 1
            mod = rasterio.open(sdir / "RL_DQN" / "cdsm_tree_canopy.tif").read(1) > 0
            lc = rasterio.open(sdir / "RL_DQN" / "landcover.tif").read(1)
            lc_base = rasterio.open(IG / sdir.name / "landcover_baseline.tif").read(1)
            mat_mod = (lc != lc_base) & (lc_base != LC_BUILDING)
            allmod = mod | mat_mod
            rows.append({"site_id": sdir.name, **counts,
                         "n_tree": counts["tree_small"] + counts["tree_medium"] + counts["tree_large"],
                         "n_material": counts["cool_pavement"] + counts["greening"] + counts["water"],
                         "cluster_score": round(cluster_score(allmod), 3)})
    if rows:
        anat = pd.DataFrame(rows)
        anat.to_csv(OUT / "e2_optimal_plan_anatomy.csv", index=False)
        print("[2/4] RL optimal-plan anatomy (action mix + cluster score):\n", anat.to_string(index=False), "\n")

    print(f"Saved E2 analysis to {OUT}")


if __name__ == "__main__":
    main()
