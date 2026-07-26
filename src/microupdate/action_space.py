"""SOLWEIG-native micro-update action space (shared by data-gen + RL env).

Every action maps onto SOLWEIG's native input channels so the simulator can
score it (执行计划: "凡是能写成 SOLWEIG 原生输入通道里的参数,都可纳入动作空间").

UMEP land-cover IDs (from solweig default_materials.json):
    0 Cobble_stone (albedo .20, Ts .37)   1 Dark_asphalt (.18 / .58, baseline hot)
    2 Roofs/buildings   5 Grass_unmanaged (.16 / .21)   6 Bare_soil   7 Water (.05 / .0)

A pixel's ground material = its land-cover ID -> albedo/emissivity/Ts.
A tree = canopy height written into the CDSM raster (casts shade via SOLWEIG's
transmissivity, Tree_settings default 0.03 leaf-on).

v1 realises the dominant, certain-to-run levers (trees of 3 heights; cool/permeable
pavement; greening; water). High-albedo reflective materials (the reflection-paradox
subclass) are deferred — they need a custom-materials pass — and are documented as
such rather than faked.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Land-cover IDs we use (UMEP standard, no custom JSON needed for v1).
LC_COBBLE = 0
LC_ASPHALT = 1      # investment-grade baseline ground (hottest)
LC_BUILDING = 2
LC_GRASS = 5
LC_BARE = 6
LC_WATER = 7


# A single placement stamps a small disk (a tree crown / material patch), not one
# 5 m pixel — the scale diagnostic showed single-pixel trees give negligible shade
# because the canopy is too small and the shadow falls on neighbours. radius=1 -> a
# 3x3 (~15 m) footprint, a realistic street-tree crown / small grove.
FOOTPRINT_RADIUS = 1


@dataclass(frozen=True)
class Action:
    aid: int
    name: str
    kind: str           # "tree" | "material"
    cost: float         # unit cost per placement (budget is summed over placements)
    cdsm_height: float  # tree canopy height (m); 0 for material actions
    new_lc: int         # target land-cover ID for material actions; -1 for trees


# Action 0 is reserved for STOP in the RL env; placement actions are 1..N here.
ACTIONS: tuple[Action, ...] = (
    Action(0, "tree_small",   "tree",     3.0, 4.0,  -1),
    Action(1, "tree_medium",  "tree",     5.0, 8.0,  -1),
    Action(2, "tree_large",   "tree",     8.0, 12.0, -1),
    Action(3, "cool_pavement", "material", 2.0, 0.0, LC_COBBLE),
    Action(4, "greening",      "material", 2.0, 0.0, LC_GRASS),
    Action(5, "water",         "material", 6.0, 0.0, LC_WATER),
    # BUILDING-ATTACHED awning (建筑挑檐/骑楼遮阳): the classic building-side micro-update —
    # a 4 m engineered canopy cantilevered from the facade. Exactly simulable in SOLWEIG
    # (CDSM shading), needs no tree pit, mounts on the existing structure (hence cheaper
    # than a small tree), and is ONLY placeable on facade-adjacent cells (env mask d<=7.5 m).
    Action(6, "building_awning", "tree", 2.5, 4.0, -1),
    # Permeable pavement (透水铺装): SOLWEIG standard bare-soil class (6) — cooler surface-
    # temperature curve than sealed asphalt; completes the retrofittable ground-class set.
    Action(7, "permeable_pavement", "material", 2.5, 0.0, LC_BARE),
)
N_PLACEMENT_ACTIONS = len(ACTIONS)
ACTION_BY_NAME = {a.name: a for a in ACTIONS}
ACTION_BY_ID = {a.aid: a for a in ACTIONS}


def candidate_mask(landcover: np.ndarray, cdsm: np.ndarray) -> np.ndarray:
    """Cells eligible for intervention: ground (non-building) with no tree yet."""
    return (landcover != LC_BUILDING) & (cdsm <= 0.0)


def apply_action(
    cdsm: np.ndarray,
    landcover: np.ndarray,
    modified: np.ndarray,
    row: int,
    col: int,
    action: Action,
    radius: int = FOOTPRINT_RADIUS,
) -> None:
    """Apply one action in-place over a (2r+1)² disk at (row,col). Buildings are
    never overwritten (interventions are street-level)."""
    r0, r1 = max(0, row - radius), min(landcover.shape[0], row + radius + 1)
    c0, c1 = max(0, col - radius), min(landcover.shape[1], col + radius + 1)
    sub_lc = landcover[r0:r1, c0:c1]
    free = (sub_lc != LC_BUILDING) & (sub_lc != LC_WATER)   # never build on water either
    if action.kind == "tree":
        sub_cd = cdsm[r0:r1, c0:c1]
        np.maximum(sub_cd, np.where(free, action.cdsm_height, sub_cd), out=sub_cd)
    else:  # material swap
        sub_lc[free] = action.new_lc
    modified[r0:r1, c0:c1][free] = 1


def build_ig_baseline_landcover(landcover_pilot: np.ndarray) -> np.ndarray:
    """Pilot landcover uses {0=ground, 2=building, 7=water(if mapped)}. Investment-grade
    baseline maps non-building, non-water ground -> Dark_asphalt (1), the hottest UMEP
    class, so cooling actions have realistic headroom. Real water bodies (from OSM
    hydrography, see apply_water_mask.py) are PRESERVED as water — they are not
    retrofittable ground and must not become asphalt."""
    lc = landcover_pilot.copy().astype(np.uint8)
    lc[(lc != LC_BUILDING) & (lc != LC_WATER)] = LC_ASPHALT
    return lc
