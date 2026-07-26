"""Stratified train/val/test split over the benchmark tiles (corrected 微更新 design):
RL trains on the TRAIN split of the FULL benchmark (reward = the per-city surrogate); methods
are compared on the held-out TEST split. Stratified by city so every split has all 4 cities.

  .venv/bin/python scripts/make_splits.py --in data/candidate_tiles/fullscale_tiles.csv
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/candidate_tiles/fullscale_tiles.csv")
    ap.add_argument("--out", default="data/candidate_tiles/splits.csv")
    ap.add_argument("--train", type=float, default=0.6)
    ap.add_argument("--val", type=float, default=0.2)   # test = 1 - train - val
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.inp)
    rng = np.random.default_rng(args.seed)
    rows = []
    for city, g in df.groupby("city"):
        sids = list(g.site_id)
        rng.shuffle(sids)
        n = len(sids)
        ntr = int(round(n * args.train))
        nva = int(round(n * args.val))
        for i, s in enumerate(sids):
            split = "train" if i < ntr else ("val" if i < ntr + nva else "test")
            rows.append({"site_id": s, "city": city, "split": split})

    out = pd.DataFrame(rows)
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / args.out, index=False)
    print(out.groupby(["city", "split"]).size().unstack(fill_value=0).to_string())
    print(f"\nsaved {args.out}  ({len(out)} tiles, seed={args.seed}, "
          f"{args.train:.0%}/{args.val:.0%}/{1-args.train-args.val:.0%})")


if __name__ == "__main__":
    main()
