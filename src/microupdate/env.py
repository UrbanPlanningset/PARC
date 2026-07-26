"""Micro-update MDP environment over the CNN surrogate (Stage 4).

A *plan* is a set of placements (candidate-location, action-type). The surrogate scores
any layout's cooling field in milliseconds, so RL and all search baselines share one fast
evaluator (`TileProblem.evaluate`). The env exposes the incremental, scene-level reward
of 微更新研究方案说明.md §2.4: scene-mean ΔUTCI (captures shade saturation because the
surrogate is a field model) + hotspot targeting − extreme-heat penalty, under a budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os, sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RASTERIO_PROJ = Path(sys.executable).resolve().parents[1] / "lib/python3.12/site-packages/rasterio/proj_data"
if RASTERIO_PROJ.exists():
    os.environ["PROJ_DATA"] = str(RASTERIO_PROJ); os.environ["PROJ_LIB"] = str(RASTERIO_PROJ)
import rasterio

from .action_space import (ACTIONS, N_PLACEMENT_ACTIONS, LC_BUILDING, LC_WATER,
                           LC_GRASS, LC_BARE, FOOTPRINT_RADIUS)
from .surrogate import featurize, lc_to_props

EXTREME_UTCI = 38.0
STATE_HW = 64   # downsampled state grid (was 96); smaller -> lighter replay buffer (~0.3GB)
# reward weights (scene-level objective); tuned so trees/hotspots dominate the day terms.
# W_NIGHT prices NIGHT ground cooling (urban-heat-island release): trees trap outgoing
# longwave at night (slightly negative) while cool pavement/greening reduce stored heat
# (positive) — this is the physical trade-off that makes non-tree actions genuinely
# valuable, instead of trees-only plans. Weight disclosed; sensitivity in the ablation.
W_HOT, W_GROUND, W_PEN, W_NIGHT = 1.0, 0.3, 1.5, 1.0
# The night ground-cooling metric is ~20-30x smaller in scale than the day hotspot metric (a tree
# shades several °C by day; materials release ~0.1°C less stored heat per cell by night, diluted
# over all ground). Normalise night ONTO the day scale before weighting, so the genuine night
# co-benefit of materials (greening/permeable cool; trees/water warm) is not drowned by day shade
# -> this is what lets the optimiser pick a comprehensive mix instead of trees-only. It is a
# scale-normaliser (disclosed, swept in a sensitivity ablation), NOT a hand-tuned preference.
NIGHT_NORM = float(os.environ.get("MICROUPDATE_NIGHT_NORM", "20"))
# Multi-criteria co-benefits beyond thermal comfort — each gives a DISTINCT intervention its own
# necessity so no single action sweeps (see action_ok street-section routing): stormwater = pervious
# area created (permeable/greening/water infiltrate), surface = ground surface-temp / UHI reduction
# (materials lower Ts; trees don't change LC). Disclosed weights, swept in the sensitivity ablation.
W_STORM = float(os.environ.get("MICROUPDATE_W_STORM", "0.5"))
W_SURF = float(os.environ.get("MICROUPDATE_W_SURF", "1.0"))
# Potential-based shaping that credits cooling delivered to the HOTTEST cells (heat-excess weighted),
# de-diluting the day-hotspot signal (a tree's per-step ΔUTCI is otherwise averaged over hundreds of
# hot cells -> ~0.01 noisy -> the agent mis-targets/clusters shade). Φ(s) is a state potential, so the
# shaping γΦ(s')-Φ(s) does NOT change the optimal policy — only guides training toward hotspot coverage.
SHAPE_SCALE = float(os.environ.get("MICROUPDATE_SHAPE", "2.0"))
# Ablation toggles (env-var driven, so ablations are one-line runs; the Q-net keeps its shape — the
# ablated channel is just zeroed/uninformative, isolating the INFORMATION's contribution).
ABLATE_COVERAGE = os.environ.get("MICROUPDATE_NO_COVERAGE", "0") == "1"        # zero per-need coverage
ABLATE_BUDGET_RASTER = os.environ.get("MICROUPDATE_NO_BUDGET_RASTER", "0") == "1"  # zero budget channel
# Cost was enforced by the hard budget constraint only (COST_COEF=0); but with the
# multi-criteria objective the policy spends budget on cheap low-value materials (carriageway
# storm bonus) that crowd out high-value trees where budget binds -> hurts truth in shade-
# dominated cities (Shanghai). A small COST_COEF makes each action earn its budget
# (cost-effectiveness): low-value materials get squeezed out where they don't pay, kept where
# they do -> value-driven, city-adaptive diversity. Env-tunable so we can sweep/version it.
COST_COEF = float(os.environ.get("MICROUPDATE_COST_COEF", "0.0"))
REWARD_SCALE = 10.0   # per-step Δobjective is ~0.01-0.05; scale up for stable TD targets


def _read(p):
    with rasterio.open(p) as s:
        return s.read(1).astype(np.float32)


@dataclass
class Layout:
    cdsm: np.ndarray
    lc: np.ndarray
    modified: np.ndarray


class TileProblem:
    """Holds one tile + surrogate; scores layouts; defines the candidate action set."""

    def __init__(self, site_id: str, model, device="cpu", budget: float | None = None,
                 k_max: int = 120, ig_root: Path | None = None):
        self.site_id = site_id
        self.model = model.to(device).eval()
        self.device = device
        ig = ig_root or (ROOT / "data/ig")
        sdir = ig / site_id / "scenarios"
        dsm = _read(ig / site_id / "dsm.tif"); dem = _read(ig / site_id / "dem.tif")
        self.building_h = np.clip(dsm - dem, 0, None)
        self.lc_base = _read(ig / site_id / "landcover_baseline.tif").astype(np.int32)
        # ground = retrofittable cells: not building, not water (real water bodies from OSM
        # are not plantable/pavable and are excluded from candidates AND from metrics)
        self.ground = (self.lc_base != LC_BUILDING) & (self.lc_base != LC_WATER)
        self.baseline_tmrt = _read(sdir / "baseline/summary/tmrt_day_mean.tif")
        self.baseline_utci = _read(sdir / "baseline/summary/utci_day_mean.tif")
        self.baseline_utci_night = _read(sdir / "baseline/summary/utci_night_mean.tif")
        self.cdsm0 = np.zeros_like(self.baseline_tmrt, dtype=np.float32)
        self.H, self.W = self.baseline_tmrt.shape
        if int(self.ground.sum()) < 25:
            raise ValueError(f"degenerate tile {site_id}: only {int(self.ground.sum())} "
                             f"plantable cells after water/building masking (skip)")
        bu = self.baseline_utci[self.ground]
        self.hot_mask = self.ground & (self.baseline_utci >= np.quantile(bu, 0.75))
        self.base_frac38 = float((self.baseline_utci[self.ground] > EXTREME_UTCI).mean())
        # heat-excess weights over the hot cells (sum=1): the potential Φ = Σ ΔUTCI·hot_w concentrates
        # credit on the hottest cells, so shading the right hotspots is rewarded sharply (not diluted).
        ex = np.clip(self.baseline_utci - EXTREME_UTCI, 0.0, None) * self.hot_mask
        self.hot_w = (ex / ex.sum()).astype(np.float32) if ex.sum() > 0 else \
            (self.hot_mask / max(1, int(self.hot_mask.sum()))).astype(np.float32)

        # candidate locations: a strided grid over ground (spread across the tile),
        # stride grown until <= k_max candidates. ranked by baseline heat for top-K display.
        gi, gj = np.where(self.ground)
        stride = 2
        while True:
            sel = (gi % stride == 0) & (gj % stride == 0)
            if sel.sum() <= k_max or stride > 16:
                break
            stride += 1
        cr, cc = gi[sel], gj[sel]
        order = np.argsort(-self.baseline_utci[cr, cc])  # hottest first
        cand = np.stack([cr[order], cc[order]], axis=1)[:k_max]
        self.K = len(cand)                # actual candidate count
        self.K_max = k_max                # fixed action-space size (tiles interchangeable)
        # pad to K_max so ONE generalist policy applies to any tile; padded slots are
        # flagged invalid (legal_mask blocks them) so they're never selected.
        self.valid = np.zeros(k_max, dtype=bool); self.valid[:self.K] = True
        if self.K < k_max:
            pad = np.repeat(cand[-1:], k_max - self.K, axis=0) if self.K else np.zeros((k_max, 2), int)
            cand = np.concatenate([cand, pad], axis=0)
        self.cand = cand                  # (K_max, 2)
        self.stride = stride
        # default budget ~ enough for ~35% of candidates as medium trees
        self.budget = budget if budget is not None else 0.35 * self.K * 5.0

        # ---- plantability constraints (微更新现实约束) -------------------------
        # Crown-clearance rule from canyon width: a tree's crown must fit the open space
        # around it, proxied by the distance to the nearest building facade:
        #   d >= 15 m: all actions;  d >= 10 m: + medium tree;  d >= 5 m: small tree;
        #   materials (cool pavement / greening) allowed on any ground; the water action
        #   needs open space like a large tree. Padded slots are all-False.
        # Street cross-section ROUTING (功能可行域): each action is feasible only where it
        # realistically belongs, so no single type can fill everything (the all-trees / all-greening
        # failure mode). Proxy by distance-to-facade d: the near-facade strip (d<=7.5 m) is the
        # sidewalk/verge frontage zone (soft, greenable, awnings); d>7.5 m is the central
        # carriageway/plaza zone (must stay trafficable -> permeable/cool pavement, not soft soil).
        #   trees: crown clearance (5/10/15 m);  awning: facade (d<=7.5);
        #   greening: frontage verge only (d<=7.5);  permeable/cool pavement: carriageway only (d>7.5);
        #   water: open space (d>=15).  (OSM road geometry is left as future refinement.)
        from scipy.ndimage import distance_transform_edt
        building = self.lc_base == LC_BUILDING
        dist_m = distance_transform_edt(~building) * 5.0          # m to nearest facade
        self.action_ok = np.zeros((k_max, N_PLACEMENT_ACTIONS), dtype=bool)
        for i in range(self.K):
            r, c = int(self.cand[i, 0]), int(self.cand[i, 1])
            d = float(dist_m[r, c])
            self.action_ok[i, 0] = d >= 5.0    # tree_small  (crown 4 m, needs soil)
            self.action_ok[i, 1] = d >= 10.0   # tree_medium (crown 8 m)
            self.action_ok[i, 2] = d >= 15.0   # tree_large  (crown 12 m)
            self.action_ok[i, 3] = d > 7.5     # cool_pavement      -> carriageway/central paved
            self.action_ok[i, 4] = d <= 7.5    # greening           -> frontage/verge (soft, near walk)
            self.action_ok[i, 5] = d >= 15.0   # water feature needs open space
            self.action_ok[i, 6] = d <= 7.5    # building awning: facade-adjacent cells only
            self.action_ok[i, 7] = d > 7.5     # permeable pavement -> carriageway/central paved
        # baseline surface temperature (for the multi-criteria surface/UHI co-benefit)
        self.ts_base = lc_to_props(self.lc_base)[1]

    # ---- layout construction & scoring -------------------------------------
    def empty_layout(self) -> Layout:
        return Layout(self.cdsm0.copy(), self.lc_base.copy().astype(np.int32),
                      np.zeros((self.H, self.W), np.uint8))

    def place(self, layout: Layout, loc_idx: int, type_idx: int) -> None:
        r, c = int(self.cand[loc_idx, 0]), int(self.cand[loc_idx, 1])
        a = ACTIONS[type_idx]
        rad = FOOTPRINT_RADIUS
        r0, r1 = max(0, r - rad), min(self.H, r + rad + 1)
        c0, c1 = max(0, c - rad), min(self.W, c + rad + 1)
        sub = layout.lc[r0:r1, c0:c1]
        free = (sub != LC_BUILDING) & (sub != LC_WATER)
        if a.kind == "tree":
            sub = layout.cdsm[r0:r1, c0:c1]
            np.maximum(sub, np.where(free, a.cdsm_height, sub), out=sub)
        else:
            layout.lc[r0:r1, c0:c1][free] = a.new_lc
        layout.modified[r0:r1, c0:c1][free] = 1

    def _delta_fields(self, cdsm: np.ndarray, lc: np.ndarray) -> np.ndarray:
        x = featurize(self.baseline_tmrt, self.building_h, cdsm, lc, self.ground)
        with torch.no_grad():
            t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            d = self.model(t).squeeze(0).cpu().numpy()
        return d  # (2,H,W) = [ΔTmrt, ΔUTCI]

    def _cobenefits(self, lc: np.ndarray) -> tuple[float, float]:
        """Geometric multi-criteria co-benefits (no surrogate): stormwater = fraction of ground
        turned PERVIOUS (greening/permeable/water infiltrate; asphalt/cobble seal), surface = mean
        ground surface-temperature reduction (materials lower Ts: asphalt 0.58 -> grass 0.21 /
        bare / cobble 0.37; trees keep the LC so Ts unchanged -> shade is scored thermally instead).
        These give pavements/greening a genuine necessity distinct from shade."""
        g = self.ground
        pervious = (lc == LC_GRASS) | (lc == LC_BARE) | (lc == LC_WATER)
        storm = float(pervious[g].mean())
        ts_new = lc_to_props(lc)[1]
        surface = float((self.ts_base[g] - ts_new[g]).mean())
        return storm, surface

    def evaluate(self, layout: Layout) -> dict:
        d = self._delta_fields(layout.cdsm, layout.lc)
        dutci = d[1]
        dnight = d[2]
        cur_utci = self.baseline_utci - dutci
        mean_hot = float(dutci[self.hot_mask].mean())
        mean_grd = float(dutci[self.ground].mean())
        mean_night = float(dnight[self.ground].mean())
        frac38 = float((cur_utci[self.ground] > EXTREME_UTCI).mean())
        storm, surface = self._cobenefits(layout.lc)
        objective = (W_HOT * mean_hot + W_GROUND * mean_grd
                     + W_PEN * (self.base_frac38 - frac38) + W_NIGHT * NIGHT_NORM * mean_night
                     + W_STORM * storm + W_SURF * surface)
        return {"mean_cool_hot": mean_hot, "mean_cool_ground": mean_grd,
                "mean_cool_night": mean_night, "storm": storm, "surface": surface,
                "frac38": frac38, "d_n38_frac": self.base_frac38 - frac38,
                "objective": objective, "dutci_field": dutci}

    def evaluate_batch(self, layouts: list[Layout]) -> list[dict]:
        xs = np.stack([featurize(self.baseline_tmrt, self.building_h, L.cdsm, L.lc, self.ground)
                       for L in layouts])
        outs = []
        with torch.no_grad():
            for i in range(0, len(xs), 64):
                t = torch.from_numpy(xs[i:i+64]).to(self.device)
                outs.append(self.model(t).cpu().numpy())
        d = np.concatenate(outs, axis=0)
        res = []
        for k, L in enumerate(layouts):
            dutci = d[k, 1]
            cur = self.baseline_utci - dutci
            mh = float(dutci[self.hot_mask].mean()); mg = float(dutci[self.ground].mean())
            mn = float(d[k, 2][self.ground].mean())
            f38 = float((cur[self.ground] > EXTREME_UTCI).mean())
            storm, surface = self._cobenefits(L.lc)
            res.append({"mean_cool_hot": mh, "mean_cool_ground": mg, "mean_cool_night": mn,
                        "storm": storm, "surface": surface, "frac38": f38,
                        "objective": W_HOT * mh + W_GROUND * mg
                                     + W_PEN * (self.base_frac38 - f38) + W_NIGHT * NIGHT_NORM * mn
                                     + W_STORM * storm + W_SURF * surface})
        return res

    def plan_cost(self, placements: list[tuple[int, int]]) -> float:
        return float(sum(ACTIONS[t].cost for _, t in placements))


class MicroUpdateEnv:
    """Incremental episode wrapper around TileProblem for DQN training."""

    def __init__(self, problem: TileProblem, max_steps: int = 40):
        self.p = problem
        self.max_steps = max_steps
        self.n_actions = problem.K_max * N_PLACEMENT_ACTIONS + 1   # + STOP (fixed across tiles)
        self.STOP = self.n_actions - 1

    def cand_grid(self, state_hw: int = STATE_HW) -> np.ndarray:
        """Candidate cells mapped to the downsampled state grid — fed to the tile-agnostic
        Q-net per forward so one generalist policy works on any tile. (K_max, 2)."""
        r = np.clip(np.round(self.p.cand[:, 0] * state_hw / self.p.H), 0, state_hw - 1)
        c = np.clip(np.round(self.p.cand[:, 1] * state_hw / self.p.W), 0, state_hw - 1)
        return np.stack([r, c], axis=1).astype(np.int64)

    def reset(self):
        self.layout = self.p.empty_layout()
        self.used_loc = np.zeros(self.p.K_max, dtype=bool)
        self.cost = 0.0
        self.steps = 0
        self.type_counts = np.zeros(N_PLACEMENT_ACTIONS, np.float32)
        ev = self.p.evaluate(self.layout)
        self.prev_obj = ev["objective"]
        self.prev_phi = float((ev["dutci_field"] * self.p.hot_w).sum())
        self._cov = self._cov_vec(ev)
        return self._state()

    def _cov_vec(self, ev) -> np.ndarray:
        """Per-NEED coverage = the current WEIGHTED contribution of each criterion to the objective
        (day comfort / night heat / stormwater / surface-UHI). Fed into the state so the policy can
        SEE which needs are already met and switch ('day is covered -> now do materials') instead of
        piling on trees. This is what lets RL coordinate the 4 retrofit categories, not over-shade."""
        if ABLATE_COVERAGE:
            return np.zeros(4, dtype=np.float32)
        return np.array([W_HOT * ev["mean_cool_hot"], W_NIGHT * NIGHT_NORM * ev["mean_cool_night"],
                         W_STORM * ev["storm"], W_SURF * ev["surface"]], dtype=np.float32)

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        remaining = self.p.budget - self.cost
        for li in range(self.p.K_max):
            if not self.p.valid[li] or self.used_loc[li]:   # skip padded/used locations
                continue
            for ti, a in enumerate(ACTIONS):
                if a.cost <= remaining and self.p.action_ok[li, ti]:   # plantability mask
                    mask[li * N_PLACEMENT_ACTIONS + ti] = True
        mask[self.STOP] = True
        return mask

    def step(self, action: int):
        done = False
        if action == self.STOP:
            return self._state(), 0.0, True, {"stop": True}
        li, ti = divmod(action, N_PLACEMENT_ACTIONS)
        self.p.place(self.layout, li, ti)
        self.used_loc[li] = True
        a = ACTIONS[ti]
        self.cost += a.cost
        self.type_counts[ti] += 1
        self.steps += 1
        ev = self.p.evaluate(self.layout)
        obj = ev["objective"]
        phi = float((ev["dutci_field"] * self.p.hot_w).sum())
        reward = (((obj - self.prev_obj) - COST_COEF * a.cost) * REWARD_SCALE
                  + SHAPE_SCALE * (phi - self.prev_phi))
        self.prev_obj = obj; self.prev_phi = phi
        self._cov = self._cov_vec(ev)
        if self.steps >= self.max_steps or (self.p.budget - self.cost) < min(x.cost for x in ACTIONS):
            done = True
        return self._state(), float(reward), done, {}

    def _state(self, state_hw: int = STATE_HW):
        """Downsampled raster channels (C,state_hw,state_hw) + scalar vector.
        Downsample (plan §2.1: state ~100×100) keeps the replay buffer small.
        No surrogate forward here (the canopy/material layout + baseline heat already
        carry the spatial info) — only the reward calls the surrogate, halving forwards."""
        alb, _ = lc_to_props(self.layout.lc)
        budget_frac = 0.0 if ABLATE_BUDGET_RASTER else float(self.p.budget - self.cost) / max(float(self.p.budget), 1e-6)
        rast = np.stack([
            self.layout.cdsm / 12.0,
            self.p.building_h / 100.0,
            (self.p.baseline_utci - 38.0) / 6.0,
            (alb - 0.20) / 0.10,
            self.layout.modified.astype(np.float32),
            self.p.hot_mask.astype(np.float32),
            np.full(self.layout.cdsm.shape, budget_frac, np.float32),   # remaining-budget fraction (spatial)
        ], axis=0).astype(np.float32)
        t = torch.from_numpy(rast).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(state_hw, state_hw),
                                            mode="bilinear", align_corners=False)
        rast = t.squeeze(0).numpy().astype(np.float16)
        scal = np.concatenate([
            [(self.p.budget - self.cost) / self.p.budget],
            [self.steps / self.max_steps],
            self.type_counts / max(1.0, self.type_counts.sum()),
            self._cov,                                   # per-need coverage (day/night/storm/surface)
        ]).astype(np.float32)
        return {"raster": rast, "scalar": scal}
