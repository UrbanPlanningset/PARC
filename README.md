# PARC: Anonymous Code Release

This repository contains the source code used to implement the PARC pipeline for
physics-grounded adaptation of frozen urban thermal retrofit optimizers under
natural-language constraints.


## Components

- `src/microupdate/`
  - eight-type intervention catalog and admissibility rules;
  - six-channel raster construction and CNN thermal surrogate;
  - tile environment, reward shaping, DQN, and generalist policy;
  - search and optimization baselines.
- `scripts/ig_*.py`
  - tile/scenario preparation, SOLWEIG orchestration, CNN training, RL training,
    full-scale evaluation, backtesting, and ablations.
- `scripts/agent_*.py`
  - LLM planner, deterministic executor, physical grounding/refinement,
    constraint battery, open-vocabulary evaluation, code baseline, reoptimization
    baselines, multi-agent extension, and SOLWEIG truth audit.
- `configs/`
  - normalized intervention costs, intervention catalog, and RL defaults.

## Installation

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA installation, install the matching PyTorch build first, then install
the remaining requirements.

## Required External Assets

The source release does not redistribute urban GIS rasters, SOLWEIG outputs, or
model checkpoints. Place locally obtained assets under:

```text
data/
  candidate_tiles/
  ig/<site_id>/
results/
  ig/surrogate/
  ig/split_experiment/plans/
```

The expected per-tile files and scenario layout are documented in
`data/README.md`. Generated data and results are excluded by `.gitignore`.

## Typical Pipeline

The following commands show the main execution order. Arguments should be
adjusted to the locally prepared benchmark.

```bash
# Prepare raster inputs and intervention scenarios.
python scripts/ig_build_tile_inputs.py --help
python scripts/ig_build_scene_dataset.py --help
python scripts/ig_generate_and_run_solweig.py --help

# Train the per-city CNN surrogate.
python scripts/ig_train_surrogate.py --help

# Train/evaluate the frozen RL optimizer and search baselines.
python scripts/ig_percity_run.py --help
python scripts/ig_fullscale.py --help

# Evaluate language-constrained adaptation.
python scripts/agent_eval.py --help
python scripts/agent_eval_gen.py --help

# Re-evaluate selected plans with SOLWEIG.
python scripts/agent_truth_gen.py --help
python scripts/agent_truth_score.py --help
```

## LLM Configuration

Agent scripts use an OpenAI-compatible chat-completions endpoint. Copy
`.env.example` to `.env` and fill in local values, or export the same variables
in the shell:

```bash
LLM_API_KEY=...
LLM_BASE_URL=https://example.com/v1
AGENT_LLM_MODEL=...
```

The `.env` file is ignored and must never be committed.



