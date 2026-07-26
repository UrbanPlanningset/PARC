"""Image-to-image CNN surrogate: intervention layout -> cooling field (ΔTmrt, ΔUTCI).

The scale diagnostic showed cooling is spatial (a tree shades a footprint+neighbourhood,
not its own pixel), so a per-pixel MLP cannot capture it — that is exactly why the old
surrogate stalled at R²=0.53. A fully-convolutional net whose receptive field spans a
shade footprint learns SOLWEIG's input->output map directly (cf. Briegel 2024).

Predicts the DELTA field (baseline − scenario) from (baseline thermal field + the new
canopy/material layout). Δ is near-zero except around interventions, so it is highly
learnable. Used both for offline R² evaluation and as the RL env's fast evaluator.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Land-cover ID -> (albedo, Ts_deg) from solweig default_materials.json.
LC_ALBEDO = {0: 0.20, 1: 0.18, 2: 0.18, 5: 0.16, 6: 0.25, 7: 0.05}
LC_TS = {0: 0.37, 1: 0.58, 2: 0.58, 5: 0.21, 6: 0.33, 7: 0.0}
IN_CHANNELS = 6
OUT_CHANNELS = 3  # ΔTmrt_day, ΔUTCI_day, ΔUTCI_night — the night channel carries the
                  # day/night trade-off (trees trap outgoing longwave at night; cool
                  # pavement/greening reduce stored heat), enabling honest multi-objective
                  # micro-update instead of trees-only plans.


def lc_to_props(landcover: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alb = np.full(landcover.shape, 0.18, np.float32)
    ts = np.full(landcover.shape, 0.58, np.float32)
    for k, v in LC_ALBEDO.items():
        alb[landcover == k] = v
    for k, v in LC_TS.items():
        ts[landcover == k] = v
    return alb, ts


def featurize(baseline_tmrt, building_height, cdsm, landcover, ground_mask) -> np.ndarray:
    """Stack & normalise the 6 input channels -> (6, H, W) float32."""
    alb, ts = lc_to_props(landcover)
    x = np.stack([
        (baseline_tmrt - 45.0) / 15.0,
        building_height / 100.0,
        cdsm / 12.0,
        (alb - 0.20) / 0.10,
        (ts - 0.40) / 0.20,
        ground_mask.astype(np.float32),
    ], axis=0).astype(np.float32)
    return x


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.c2(self.act(self.c1(x))))


class SurrogateCNN(nn.Module):
    """Fully-convolutional residual net. Receptive field ~26 px (130 m) covers shade
    footprints. Output masked to ground at the loss/aggregation level."""

    def __init__(self, in_ch=IN_CHANNELS, hidden=64, out_ch=OUT_CHANNELS, n_blocks=6):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Conv2d(hidden, out_ch, 3, padding=1)

    def forward(self, x):
        return self.head(self.blocks(torch.relu(self.stem(x))))


@torch.no_grad()
def predict_delta(model, x_np: np.ndarray, device="cpu") -> np.ndarray:
    """x_np: (6,H,W) -> (2,H,W) predicted [ΔTmrt, ΔUTCI]."""
    model.eval()
    t = torch.from_numpy(x_np).unsqueeze(0).to(device)
    return model(t).squeeze(0).cpu().numpy()
