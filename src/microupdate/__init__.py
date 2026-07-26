"""Investment-grade micro-update pipeline (v2).

Clean reimplementation matching 执行计划与终止准则.md / 微更新研究方案说明.md:
SOLWEIG-native action space, scene-level reward, surrogate + Dueling Double DQN.
Supersedes the old 3-action scaffold (src/rl/actions.py, scripts/generate_intervention_scenarios.py).
"""


def pick_device() -> str:
    """Best available torch device: CUDA (server GPU) > MPS (Apple) > CPU.

    Override with env MICROUPDATE_DEVICE=cpu|cuda|mps.
    """
    import os
    forced = os.environ.get("MICROUPDATE_DEVICE")
    if forced:
        return forced
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
