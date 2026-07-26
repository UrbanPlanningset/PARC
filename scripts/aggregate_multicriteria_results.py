#!/usr/bin/env python3
"""Aggregate multi-criteria micro-renewal results for representative tiles.

This script intentionally reads existing result CSVs and writes only under
results/representative_tile_subset. It combines SOLWEIG-verified scenario ranks
with provisional surrogate-only metaheuristic baselines.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "representative_tile_subset"
SITES_DIR = ROOT / "results" / "sites"

PLANNER_RANK = OUT_DIR / "representative_tile_planner_backtest_rank.csv"
METAHEURISTIC_SUMMARY = OUT_DIR / "metaheuristic_baseline_surrogate_summary.csv"

ACTION_COSTS = {
    "add_tree": 8.0,
    "add_shade": 10.0,
    "cool_pavement": 5.0,
}

FIXED_ACTION_COUNTS = {
    "tree_20": {"add_tree": 20},
    "tree_50": {"add_tree": 50},
    "tree_100": {"add_tree": 100},
    "mixed_12_10_25": {"add_tree": 12, "add_shade": 10, "cool_pavement": 25},
    "mixed_large_40_30_120": {"add_tree": 40, "add_shade": 30, "cool_pavement": 120},
    "rand_large_000": {"add_tree": 80, "add_shade": 30, "cool_pavement": 160},
    "rand_large_001": {"add_tree": 45, "add_shade": 70, "cool_pavement": 220},
}

CITY_ORDER = ["Beijing", "New York", "Seoul", "Shanghai"]
METHOD_PRIORITY = [
    "tree_100",
    "catalog_plan_budget800",
    "plan_q_budget800",
    "plan_greedy_budget800",
    "ga_budget800",
    "sa_budget800",
    "nsga2_budget800",
    "rand_large_000",
    "mixed_large_40_30_120",
]


def entropy_from_counts(counts: dict[str, float] | None) -> float:
    if not counts:
        return 0.0
    values = np.array([float(v) for v in counts.values() if float(v) > 0.0])
    if values.size == 0:
        return 0.0
    probs = values / values.sum()
    return float(-(probs * np.log(probs)).sum())


def normalized_entropy(counts: dict[str, float] | None) -> float:
    if not counts:
        return 0.0
    active = sum(1 for value in counts.values() if float(value) > 0.0)
    if active <= 1:
        return 0.0
    return entropy_from_counts(counts) / math.log(active)


def budget_from_counts(counts: dict[str, float] | None) -> float:
    if not counts:
        return float("nan")
    return float(sum(ACTION_COSTS.get(action, 0.0) * float(n) for action, n in counts.items()))


def parse_counts(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def classify_method(scenario_id: str) -> str:
    if scenario_id == "tree_100":
        return "thermal_upper_reference"
    if scenario_id == "catalog_plan_budget800":
        return "ours_catalog_constrained"
    if scenario_id.startswith("tree_"):
        return "fixed_tree_baseline"
    if scenario_id.startswith("mixed"):
        return "fixed_mixed_baseline"
    if scenario_id.startswith("rand"):
        return "random_baseline"
    if scenario_id.startswith("plan_q"):
        return "ours_rl"
    if scenario_id.startswith("plan_greedy"):
        return "greedy_baseline"
    if scenario_id.startswith("ga_"):
        return "metaheuristic_ga"
    if scenario_id.startswith("sa_"):
        return "metaheuristic_sa"
    if scenario_id.startswith("nsga2_"):
        return "metaheuristic_nsga2"
    return "other"


def load_plan_counts(site_id: str, scenario_id: str) -> dict[str, float]:
    if scenario_id == "plan_q_budget800":
        path = SITES_DIR / site_id / "representative_multitime_plans" / "multitime_q_plan_max120.csv"
    elif scenario_id == "plan_greedy_budget800":
        path = SITES_DIR / site_id / "representative_multitime_plans" / "multitime_greedy_plan_max120.csv"
    elif scenario_id == "catalog_plan_budget800":
        path = SITES_DIR / site_id / "catalog_multitime_plans" / "catalog_constrained_greedy_plan.csv"
    else:
        return {}
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "action" not in df.columns:
        return {}
    return {str(k): float(v) for k, v in df["action"].value_counts().to_dict().items()}


def enrich_counts(site_id: str, scenario_id: str) -> dict[str, float]:
    if scenario_id in FIXED_ACTION_COUNTS:
        return dict(FIXED_ACTION_COUNTS[scenario_id])
    if scenario_id.startswith("plan_") or scenario_id == "catalog_plan_budget800":
        return load_plan_counts(site_id, scenario_id)
    return {}


def load_verified_rows() -> pd.DataFrame:
    frames = []
    for path in sorted(SITES_DIR.glob("*/representative_subset_multitime/multitime_scenario_rank.csv")):
        site_id = path.parts[-3]
        df = pd.read_csv(path)
        if "city" not in df.columns:
            city = site_id.split("_")[0].title()
            if site_id.startswith("nyc_"):
                city = "New York"
            elif site_id.startswith("seoul_"):
                city = "Seoul"
            df.insert(0, "city", city)
        if "site_id" not in df.columns:
            df.insert(1, "site_id", site_id)
        frames.append(df)

    if not frames and PLANNER_RANK.exists():
        frames.append(pd.read_csv(PLANNER_RANK))

    if not frames:
        raise FileNotFoundError("No SOLWEIG-verified rank CSV files found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "mean_delta_tmrt_vs_baseline_c": "mean_delta",
            "min_time_delta_tmrt_c": "weak_hour_delta",
        }
    )
    df["scenario_id"] = df["scenario_id"].astype(str)
    df["method"] = df["scenario_id"]
    df["method_type"] = df["method"].map(classify_method)
    df["verification_status"] = "solweig_verified"
    df["is_tree100_reference"] = df["scenario_id"].eq("tree_100")
    df["action_counts"] = [
        enrich_counts(site_id, scenario_id)
        for site_id, scenario_id in zip(df["site_id"], df["scenario_id"])
    ]
    df["budget"] = df["action_counts"].map(budget_from_counts)
    df["n_actions"] = df["action_counts"].map(lambda d: float(sum(d.values())) if d else np.nan)
    df["action_entropy"] = df["action_counts"].map(entropy_from_counts)
    df["action_entropy_norm"] = df["action_counts"].map(normalized_entropy)
    df["source"] = "SOLWEIG backtest"
    df["surrogate_pred_delta"] = np.nan
    df["surrogate_pred_weak_delta"] = np.nan
    return df


def load_surrogate_rows() -> pd.DataFrame:
    if not METAHEURISTIC_SUMMARY.exists():
        return pd.DataFrame()
    df = pd.read_csv(METAHEURISTIC_SUMMARY)
    df = df.rename(columns={"method": "scenario_id", "used_budget": "budget"})
    df["method"] = df["scenario_id"]
    df["method_type"] = df["method"].map(classify_method)
    df["verification_status"] = "surrogate_only"
    df["is_tree100_reference"] = False
    df["action_counts"] = df["action_counts"].map(parse_counts)
    df["action_entropy"] = df["action_counts"].map(entropy_from_counts)
    df["action_entropy_norm"] = df["action_counts"].map(normalized_entropy)
    df["mean_delta"] = np.nan
    df["weak_hour_delta"] = np.nan
    df["worsened_cells"] = np.nan
    df["improved_cells"] = np.nan
    df["source"] = "MLP surrogate only"
    df["surrogate_pred_delta"] = df.get("pred_delta_tmrt_c", np.nan)
    df["surrogate_pred_weak_delta"] = df.get("pred_weak_delta_tmrt_c", np.nan)
    return df


def merge_verified_and_surrogate_metadata(verified: pd.DataFrame, surrogate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach surrogate action metadata to physically verified metaheuristic rows.

    GA/SA/NSGA-II plans are now SOLWEIG-backtested, but their action counts and
    surrogate-predicted objectives still live in the surrogate summary. This
    merge keeps one verified row per method while preserving that useful
    planning metadata.
    """
    if surrogate.empty:
        return verified, surrogate

    verified = verified.copy()
    surrogate = surrogate.copy()
    meta_cols = [
        "action_counts",
        "action_entropy",
        "action_entropy_norm",
        "budget",
        "n_actions",
        "surrogate_pred_delta",
        "surrogate_pred_weak_delta",
        "pred_delta_tmrt_c",
        "pred_weak_delta_tmrt_c",
        "pred_shade_gain",
        "surrogate_score",
        "spatial_spread",
    ]
    surrogate_keyed = surrogate.set_index(["city", "site_id", "scenario_id"], drop=False)
    verified_keys = set(zip(verified["city"], verified["site_id"], verified["scenario_id"]))

    for idx, row in verified.iterrows():
        key = (row["city"], row["site_id"], row["scenario_id"])
        if key not in surrogate_keyed.index:
            continue
        meta = surrogate_keyed.loc[key]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        for col in meta_cols:
            if col in meta.index and col in verified.columns:
                current = verified.at[idx, col]
                if (isinstance(current, dict) and not current) or pd.isna(current):
                    verified.at[idx, col] = meta[col]

    surrogate = surrogate[
        ~surrogate.apply(lambda row: (row["city"], row["site_id"], row["scenario_id"]) in verified_keys, axis=1)
    ].copy()
    return verified, surrogate


def normalize_group(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = values.astype(float)
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index)
    min_v = values.min(skipna=True)
    max_v = values.max(skipna=True)
    if math.isclose(float(max_v), float(min_v)):
        out = pd.Series(1.0, index=values.index)
    else:
        out = (values - min_v) / (max_v - min_v)
    if not higher_is_better:
        out = 1.0 - out
    return out.clip(0.0, 1.0)


def add_normalized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    verified = df["verification_status"].eq("solweig_verified")

    tree_ref = (
        df[verified & df["scenario_id"].eq("tree_100")]
        .set_index("city")[["mean_delta", "weak_hour_delta"]]
        .rename(columns={"mean_delta": "tree100_mean_delta", "weak_hour_delta": "tree100_weak_hour_delta"})
    )
    df = df.merge(tree_ref, left_on="city", right_index=True, how="left")

    df["mean_vs_tree100"] = df["mean_delta"] / df["tree100_mean_delta"]
    df["weak_vs_tree100"] = df["weak_hour_delta"] / df["tree100_weak_hour_delta"]
    df["mean_vs_tree100"] = df["mean_vs_tree100"].replace([np.inf, -np.inf], np.nan)
    df["weak_vs_tree100"] = df["weak_vs_tree100"].replace([np.inf, -np.inf], np.nan)

    df["safety_score"] = np.nan
    df["budget_efficiency"] = np.nan
    df["surrogate_pred_norm"] = np.nan
    for city, idx in df.groupby("city").groups.items():
        city_idx = list(idx)
        df.loc[city_idx, "safety_score"] = normalize_group(
            df.loc[city_idx, "worsened_cells"], higher_is_better=False
        )
        verified_idx = df.index[df["city"].eq(city) & verified].tolist()
        df.loc[verified_idx, "budget_efficiency"] = normalize_group(
            df.loc[verified_idx, "mean_delta"] / df.loc[verified_idx, "budget"],
            higher_is_better=True,
        )
        surrogate_idx = df.index[df["city"].eq(city) & df["verification_status"].eq("surrogate_only")].tolist()
        df.loc[surrogate_idx, "surrogate_pred_norm"] = normalize_group(
            df.loc[surrogate_idx, "surrogate_pred_delta"], higher_is_better=True
        )

    df["thermal_norm"] = df["mean_vs_tree100"].clip(lower=0.0, upper=1.25)
    df["weak_hour_norm"] = df["weak_vs_tree100"].clip(lower=0.0, upper=1.25)
    df["diversity_norm"] = df["action_entropy_norm"].clip(0.0, 1.0)
    df["budget_norm"] = np.nan
    for city, idx in df.groupby("city").groups.items():
        df.loc[list(idx), "budget_norm"] = normalize_group(df.loc[list(idx), "budget"], higher_is_better=False)

    df["multicriteria_score_verified"] = (
        0.35 * df["thermal_norm"].fillna(0.0)
        + 0.25 * df["weak_hour_norm"].fillna(0.0)
        + 0.15 * df["safety_score"].fillna(0.0)
        + 0.15 * df["diversity_norm"].fillna(0.0)
        + 0.10 * df["budget_norm"].fillna(0.0)
    )
    df.loc[~verified, "multicriteria_score_verified"] = np.nan

    return df


def method_label(method: str) -> str:
    labels = {
        "tree_100": "tree_100\nthermal ref.",
        "catalog_plan_budget800": "Ours catalog",
        "plan_q_budget800": "Ours Q",
        "plan_greedy_budget800": "Greedy",
        "ga_budget800": "GA\nsurrogate",
        "sa_budget800": "SA\nsurrogate",
        "nsga2_budget800": "NSGA-II\nsurrogate",
        "rand_large_000": "Random-large",
        "mixed_large_40_30_120": "Mixed-large",
    }
    return labels.get(method, method)


def plot_normalized_bars(df: pd.DataFrame, path: Path) -> None:
    plot_df = df[df["scenario_id"].isin(METHOD_PRIORITY)].copy()
    plot_df["scenario_order"] = plot_df["scenario_id"].map({m: i for i, m in enumerate(METHOD_PRIORITY)})
    plot_df = plot_df.sort_values(["city", "scenario_order"])

    metrics = [
        ("thermal_norm", "Mean Tmrt"),
        ("weak_hour_norm", "Weak hour"),
        ("safety_score", "Low worsening"),
        ("diversity_norm", "Diversity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.ravel()
    colors = {
        "thermal_upper_reference": "#2a9d8f",
        "ours_catalog_constrained": "#1f9e89",
        "ours_rl": "#e76f51",
        "greedy_baseline": "#f4a261",
        "metaheuristic_ga": "#6a4c93",
        "metaheuristic_sa": "#8a6fb0",
        "metaheuristic_nsga2": "#4b6cb7",
        "random_baseline": "#8d99ae",
        "fixed_mixed_baseline": "#a7c957",
    }

    for ax, city in zip(axes, CITY_ORDER):
        city_df = plot_df[plot_df["city"].eq(city)]
        x = np.arange(len(city_df))
        bottom = np.zeros(len(city_df))
        for metric, label in metrics:
            values = city_df[metric].fillna(0.0).clip(0, 1.25).to_numpy()
            ax.bar(x, values, bottom=bottom, width=0.72, label=label)
            bottom += values
        for i, (_, row) in enumerate(city_df.iterrows()):
            if row["verification_status"] == "surrogate_only":
                ax.text(i, bottom[i] + 0.04, "S", ha="center", va="bottom", fontsize=9, color="#555")
            if row["scenario_id"] == "tree_100":
                ax.text(i, bottom[i] + 0.04, "REF", ha="center", va="bottom", fontsize=9, color="#0f766e")
        ax.set_title(city)
        ax.set_xticks(x)
        ax.set_xticklabels([method_label(m) for m in city_df["scenario_id"]], rotation=30, ha="right")
        ax.set_ylim(0, 4.5)
        ax.grid(axis="y", alpha=0.25)
        for tick, (_, row) in zip(ax.get_xticklabels(), city_df.iterrows()):
            tick.set_color(colors.get(row["method_type"], "#333333"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        "Multi-criteria comparison: tree_100 is a thermal reference; Ours/GA/NSGA-II are planning candidates",
        y=0.98,
        fontsize=14,
    )
    fig.text(0.5, 0.02, "S = surrogate-only provisional baseline; SOLWEIG verification pending", ha="center")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_radar(df: pd.DataFrame, path: Path) -> None:
    focus = ["tree_100", "plan_q_budget800", "ga_budget800", "nsga2_budget800"]
    metrics = [
        ("thermal_norm", "Mean"),
        ("weak_hour_norm", "Weak"),
        ("safety_score", "Safety"),
        ("diversity_norm", "Diversity"),
        ("budget_norm", "Budget"),
    ]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": "polar"})
    axes = axes.ravel()
    palette = {
        "tree_100": "#2a9d8f",
        "plan_q_budget800": "#e76f51",
        "ga_budget800": "#6a4c93",
        "nsga2_budget800": "#4b6cb7",
    }
    linestyles = {
        "tree_100": "-",
        "plan_q_budget800": "-",
        "ga_budget800": "--",
        "nsga2_budget800": "--",
    }

    for ax, city in zip(axes, CITY_ORDER):
        city_df = df[df["city"].eq(city)].set_index("scenario_id", drop=False)
        for method in focus:
            if method not in city_df.index:
                continue
            row = city_df.loc[method]
            if isinstance(row, pd.DataFrame):
                verified_rows = row[row["verification_status"].eq("solweig_verified")]
                row = verified_rows.iloc[0] if not verified_rows.empty else row.iloc[0]
            values = [float(row[m]) if pd.notna(row[m]) else 0.0 for m, _ in metrics]
            values = [min(max(v, 0.0), 1.0) for v in values]
            values += values[:1]
            label = method_label(method).replace("\n", " ")
            if row["verification_status"] == "surrogate_only":
                label += " (surrogate)"
            ax.plot(angles, values, color=palette[method], linestyle=linestyles[method], linewidth=2, label=label)
            ax.fill(angles, values, color=palette[method], alpha=0.08)
        ax.set_title(city, y=1.10)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label for _, label in metrics])
        ax.set_ylim(0, 1)
        ax.set_yticklabels([])
        ax.grid(alpha=0.28)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Radar view of verified and provisional multi-objective candidates", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_brief(df: pd.DataFrame, path: Path) -> None:
    verified = df[df["verification_status"].eq("solweig_verified")]
    best_verified = (
        verified.sort_values(["city", "multicriteria_score_verified"], ascending=[True, False])
        .groupby("city")
        .head(1)
    )
    lines = [
        "# Multi-criteria result note",
        "",
        "This note is generated by `scripts/aggregate_multicriteria_results.py`.",
        "",
        "Interpretation:",
        "- `tree_100` is treated as a thermal upper-reference baseline, not as the desired planning solution.",
        "- GA/SA/NSGA-II rows are marked `solweig_verified` when their generated plans have been converted into SOLWEIG scenarios and physically backtested.",
        "- The verified multi-criteria score combines mean cooling, weak-hour cooling, low worsened-cells, action diversity, and budget parsimony.",
        "",
        "Current best SOLWEIG-verified multi-criteria row by city:",
        "",
        "| city | method | mean_delta | weak_hour_delta | worsened_cells | action_entropy_norm | score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best_verified.iterrows():
        lines.append(
            f"| {row['city']} | {row['scenario_id']} | {row['mean_delta']:.4f} | "
            f"{row['weak_hour_delta']:.4f} | {row['worsened_cells']:.1f} | "
            f"{row['action_entropy_norm']:.3f} | {row['multicriteria_score_verified']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Recommended next step: rerun Ours/GA/NSGA-II with the expanded action catalog and explicit spatial-integrity constraints.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verified = load_verified_rows()
    surrogate = load_surrogate_rows()
    verified, surrogate = merge_verified_and_surrogate_metadata(verified, surrogate)
    common_columns = sorted(set(verified.columns) | set(surrogate.columns))
    df = pd.concat(
        [verified.reindex(columns=common_columns), surrogate.reindex(columns=common_columns)],
        ignore_index=True,
    )
    df = add_normalized_metrics(df)

    sort_key = {city: i for i, city in enumerate(CITY_ORDER)}
    method_key = {method: i for i, method in enumerate(METHOD_PRIORITY)}
    df["_city_order"] = df["city"].map(sort_key).fillna(99)
    df["_method_order"] = df["scenario_id"].map(method_key).fillna(50)
    df = df.sort_values(["_city_order", "_method_order", "verification_status", "scenario_id"])
    df = df.drop(columns=["_city_order", "_method_order"])

    csv_path = OUT_DIR / "representative_tile_multicriteria_summary.csv"
    norm_path = OUT_DIR / "representative_tile_multicriteria_normalized.csv"
    bar_path = OUT_DIR / "representative_tile_multicriteria_normalized_bars.png"
    radar_path = OUT_DIR / "representative_tile_multicriteria_radar.png"
    note_path = OUT_DIR / "representative_tile_multicriteria_note.md"

    df.to_csv(csv_path, index=False)
    norm_cols = [
        "city",
        "site_id",
        "scenario_id",
        "method_type",
        "verification_status",
        "mean_delta",
        "weak_hour_delta",
        "worsened_cells",
        "budget",
        "action_entropy_norm",
        "thermal_norm",
        "weak_hour_norm",
        "safety_score",
        "diversity_norm",
        "budget_norm",
        "surrogate_pred_norm",
        "multicriteria_score_verified",
    ]
    df[norm_cols].to_csv(norm_path, index=False)

    plot_normalized_bars(df, bar_path)
    plot_radar(df, radar_path)
    write_brief(df, note_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {norm_path}")
    print(f"Wrote {bar_path}")
    print(f"Wrote {radar_path}")
    print(f"Wrote {note_path}")


if __name__ == "__main__":
    main()
