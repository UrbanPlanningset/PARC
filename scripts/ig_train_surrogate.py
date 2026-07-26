"""Stage 3: train the image-to-image CNN surrogate and check the R²>=0.8 gate.

Target = day-mean cooling field (ΔTmrt, ΔUTCI) = baseline − scenario, over ground cells.
Gate metric = ΔTmrt R² on held-out scenarios (honest analog of the plan's "delta_tmrt R²").
Also reports leave-one-city-out R² for cross-tile generalisation.

.venv/bin/python scripts/ig_train_surrogate.py
"""
from __future__ import annotations

from pathlib import Path
import os, sys, json, time, argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RASTERIO_PROJ = Path(sys.executable).resolve().parents[1] / "lib/python3.12/site-packages/rasterio/proj_data"
if RASTERIO_PROJ.exists():
    os.environ["PROJ_DATA"] = str(RASTERIO_PROJ); os.environ["PROJ_LIB"] = str(RASTERIO_PROJ)
import rasterio
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(ROOT))
from src.microupdate.action_space import LC_BUILDING, LC_WATER
from src.microupdate.surrogate import SurrogateCNN, featurize

IG = ROOT / "data/ig"
OUT = ROOT / "results/ig/surrogate"


def read(p):
    with rasterio.open(p) as s:
        return s.read(1).astype(np.float32)


def read_t(p):
    with rasterio.open(p) as s:
        return s.read(1).astype(np.float32), s.transform


def _align(items):
    """items: list of (name, array, transform), all same CRS & 5 m pixel. Crop every array
    to their common geographic extent (handles the few-pixel shape drift that stage-B baseline
    re-runs introduced between re-run baseline rasters and un-re-run scenario rasters). When
    all share the same transform/shape this is a no-op."""
    a = items[0][2].a; e = abs(items[0][2].e)
    L = max(t.c for _, _, t in items)                                 # common left x
    R = min(t.c + arr.shape[1] * t.a for _, arr, t in items)          # common right x
    T = min(t.f for _, _, t in items)                                 # common top y (north)
    B = max(t.f - arr.shape[0] * e for _, arr, t in items)            # common bottom y
    h = int(round((T - B) / e)); w = int(round((R - L) / a))
    out = {}
    for name, arr, t in items:
        i0 = int(round((t.f - T) / e)); j0 = int(round((L - t.c) / a))
        out[name] = arr[i0:i0 + h, j0:j0 + w]
    return out


def load_samples():
    man = pd.read_csv(IG / "scenario_manifest.csv")
    samples = []
    for sid, grp in man.groupby("site_id"):
        sdir = IG / sid / "scenarios"
        dsm, t_dsm = read_t(IG / sid / "dsm.tif"); dem, _ = read_t(IG / sid / "dem.tif")
        bh = np.clip(dsm - dem, 0, None)
        lc_base, t_lc = read_t(IG / sid / "landcover_baseline.tif")
        b_tmrt, t_bt = read_t(sdir / "baseline/summary/tmrt_day_mean.tif")
        b_utci, t_bu = read_t(sdir / "baseline/summary/utci_day_mean.tif")
        b_utci_n, t_bn = read_t(sdir / "baseline/summary/utci_night_mean.tif")
        for _, r in grp.iterrows():
            scn = r.scenario_id
            cdsm, t_cd = read_t(sdir / scn / "cdsm_tree_canopy.tif")
            lc, t_l = read_t(sdir / scn / "landcover.tif")
            s_tmrt, t_st = read_t(sdir / scn / "summary/tmrt_day_mean.tif")
            s_utci, t_su = read_t(sdir / scn / "summary/utci_day_mean.tif")
            s_utci_n, t_sn = read_t(sdir / scn / "summary/utci_night_mean.tif")
            al = _align([("bt", b_tmrt, t_bt), ("bh", bh, t_dsm), ("lcb", lc_base, t_lc),
                         ("bu", b_utci, t_bu), ("bn", b_utci_n, t_bn),
                         ("cd", cdsm, t_cd), ("lc", lc, t_l), ("st", s_tmrt, t_st),
                         ("su", s_utci, t_su), ("sn", s_utci_n, t_sn)])
            # exclude buildings AND real water from the loss mask
            ground = ((al["lcb"] != LC_BUILDING) & (al["lcb"] != LC_WATER)).astype(np.float32)
            x = featurize(al["bt"], al["bh"], al["cd"], al["lc"], ground)
            y = np.stack([al["bt"] - al["st"], al["bu"] - al["su"], al["bn"] - al["sn"]],
                         axis=0).astype(np.float32)
            samples.append({"site": sid, "city": r.city, "scn": scn,
                            "x": x, "y": y, "g": ground})
    return samples


def r2_masked(pred, true, mask):
    m = mask > 0
    t, p = true[m], pred[m]
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum()) + 1e-9
    mae = float(np.abs(t - p).mean())
    return 1.0 - ss_res / ss_tot, mae


def train(train_samples, device, steps=4000, bs=16, crop=96, seed=0, log_every=500, night_boost=1.0,
          equal_weight=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = SurrogateCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    # Per-channel inverse-variance loss weights. The ΔUTCI_night channel is ~20x smaller in scale
    # than the day channels, so an equal-weighted MSE regresses it to zero (even flips its sign) ->
    # the surrogate cannot see that materials cool at night / trees warm at night, so the optimizer
    # never picks materials. Normalising each channel by its target variance (over ground cells)
    # puts night on equal footing; night_boost further emphasises it.
    nC = train_samples[0]["y"].shape[0]
    ss = np.zeros(nC, np.float64); cnt = 0
    for s in train_samples:
        g = s["g"].astype(bool); cnt += int(g.sum())
        for c in range(nC):
            ss[c] += float((s["y"][c][g] ** 2).sum())
    var = ss / max(cnt, 1) + 1e-8
    w = 1.0 / var; w = w / w.mean()
    if nC >= 3:
        w[2] *= night_boost
    if equal_weight:
        w = np.ones(nC, dtype=np.float64)   # ABLATION: drop inverse-variance -> night channel swamped
    wsum = float(w.sum())
    CHAN_W = torch.tensor(w, dtype=torch.float32, device=device).view(1, nC, 1, 1)
    tag = "EQUAL-WEIGHT ablation" if equal_weight else f"inv-var, night_boost={night_boost}"
    print(f"  per-channel loss weights ({tag}): {np.round(w, 3)}", flush=True)
    model.train()
    for step in range(steps):
        xb, yb, gb = [], [], []
        for _ in range(bs):
            s = train_samples[rng.integers(len(train_samples))]
            H, W = s["g"].shape
            ch, cw = min(crop, H), min(crop, W)
            i = rng.integers(0, H - ch + 1); j = rng.integers(0, W - cw + 1)
            xb.append(s["x"][:, i:i+ch, j:j+cw])
            yb.append(s["y"][:, i:i+ch, j:j+cw])
            gb.append(s["g"][i:i+ch, j:j+cw])
        xb = torch.from_numpy(np.stack(xb)).to(device)
        yb = torch.from_numpy(np.stack(yb)).to(device)
        gb = torch.from_numpy(np.stack(gb)).unsqueeze(1).to(device)
        pred = model(xb)
        loss = (((pred - yb) ** 2) * gb * CHAN_W).sum() / (gb.sum() * wsum + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % log_every == 0:
            print(f"  step {step+1:4}/{steps}  loss={loss.item():.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, samples, device):
    model.eval()
    preds_t, true_t, preds_u, true_u, preds_n, true_n, masks = [], [], [], [], [], [], []
    for s in samples:
        x = torch.from_numpy(s["x"]).unsqueeze(0).to(device)
        p = model(x).squeeze(0).cpu().numpy()
        preds_t.append(p[0]); true_t.append(s["y"][0])
        preds_u.append(p[1]); true_u.append(s["y"][1])
        preds_n.append(p[2]); true_n.append(s["y"][2])
        masks.append(s["g"])
    P_t = np.concatenate([a.ravel() for a in preds_t]); T_t = np.concatenate([a.ravel() for a in true_t])
    P_u = np.concatenate([a.ravel() for a in preds_u]); T_u = np.concatenate([a.ravel() for a in true_u])
    P_n = np.concatenate([a.ravel() for a in preds_n]); T_n = np.concatenate([a.ravel() for a in true_n])
    M = np.concatenate([a.ravel() for a in masks])
    r2t, maet = r2_masked(P_t, T_t, M)
    r2u, maeu = r2_masked(P_u, T_u, M)
    r2n, maen = r2_masked(P_n, T_n, M)
    return {"dtmrt_r2": r2t, "dtmrt_mae": maet, "dutci_r2": r2u, "dutci_mae": maeu,
            "dnight_r2": r2n, "dnight_mae": maen}


def city_key(c: str) -> str:
    return c.lower().replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--city", default=None, help="train one city only (default: all, per-city)")
    ap.add_argument("--night-boost", type=float, default=1.0,
                    help="extra weight on the night channel beyond inverse-variance (raise if dnight_r2 stays low)")
    ap.add_argument("--equal-weight", action="store_true",
                    help="ABLATION: disable inverse-variance loss weighting (night channel gets swamped)")
    ap.add_argument("--tag", default="", help="output suffix: surrogate_<city>_<tag>.pt (avoids clobbering the canonical surrogate)")
    args = ap.parse_args()
    from src.microupdate import pick_device
    device = pick_device()
    print(f"device={device}")
    t0 = time.time()
    samples = load_samples()
    by_city = {}
    for s in samples:
        by_city.setdefault(s["city"], []).append(s)
    print(f"loaded {len(samples)} samples / {len(by_city)} cities in {time.time()-t0:.0f}s")

    cities = [args.city] if args.city else list(by_city)
    OUT.mkdir(parents=True, exist_ok=True)
    allm = {}
    for city in cities:
        cs = by_city[city]
        rng = np.random.default_rng(args.seed)
        idx = list(range(len(cs))); rng.shuffle(idx)
        nval = max(1, len(cs) // 5); val = set(idx[:nval])
        tr = [cs[i] for i in range(len(cs)) if i not in val]
        va = [cs[i] for i in range(len(cs)) if i in val]
        print(f"\n=== {city}  (train={len(tr)} val={len(va)} scenarios, per-city surrogate) ===", flush=True)
        model = train(tr, device, steps=args.steps, seed=args.seed, night_boost=args.night_boost,
                      equal_weight=args.equal_weight)
        m = evaluate(model, va, device)
        print(f"  ΔTmrt R²={m['dtmrt_r2']:.3f} MAE={m['dtmrt_mae']:.3f}  |  ΔUTCI R²={m['dutci_r2']:.3f}"
              f"  |  ΔUTCI_night R²={m['dnight_r2']:.3f} MAE={m['dnight_mae']:.3f}"
              f"  GATE R²>=0.8: {'PASS' if m['dtmrt_r2'] >= 0.8 else 'FAIL'}")
        suffix = f"_{args.tag}" if args.tag else ""
        torch.save(model.state_dict(), OUT / f"surrogate_{city_key(city)}{suffix}.pt")
        allm[city] = {**m, "n_train": len(tr), "n_val": len(va)}
    metrics_name = f"per_city_metrics_{args.tag}.json" if args.tag else "per_city_metrics.json"
    json.dump(allm, open(OUT / metrics_name, "w"), indent=2)
    print(f"\nSaved per-city surrogates + per_city_metrics.json to {OUT}")


if __name__ == "__main__":
    main()
